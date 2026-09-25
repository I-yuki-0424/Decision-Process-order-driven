"""Pluggable Candidate-Model Benchmark Driver for Craftax-Classic.

Extends src/pipeline/craftax_benchmark.py's env setup / eval-loop pattern
(reusing CraftaxEnvAdapter and calculate_crafter_score directly) rather than
forking it, so any docs/DPOD.ipynb candidate under src/model/candidates/ can
be trained and evaluated against the real environment the same way the
existing 4th/5th-idea models are, through one shared driver and a small
{name: CandidateSpec} registry (CANDIDATE_REGISTRY below) instead of a
duplicated training loop per candidate.

All metrics come from real env steps / model forward passes (see
CraftaxEnvAdapter.step and each candidate's forward_* function) -- no
synthetic scores. Training uses src/model/checkpoint.py's
AsyncCheckpointManager for periodic, non-blocking, resumable checkpointing
under output/checkpoints/<candidate_name>/.

CPU-feasible by default (small d_model, few episodes/steps): no GPU is
required to run this. A full-scale run needs separate operator authorization
per ADR-002 (see CLAUDE.md / docs/DECISIONS).
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.craftax_env_adapter import (
    ACHIEVEMENT_NAMES,
    NUM_ACHIEVEMENTS,
    CraftaxEnvAdapter,
    calculate_crafter_score,
)
from src.model.checkpoint import AsyncCheckpointManager
from src.model.logger_utils import get_logger
from src.model.candidates import mdp_branch as mdpb
from src.model.candidates import transformer_branch as tfb
from src.model.candidates import variant_5_1_shared_attention as v51
from src.model.candidates import variant_5_2_channel_split as v52
from src.model.candidates import variant_5_3_channel_split_v2 as v53
from src.model.candidates import variant_5_4_single_shot as v54

logger = get_logger("CandidateBenchmark")


class CandidateMetrics(NamedTuple):
    """Metrics container for a benchmarked candidate. Mirrors
    src/pipeline/craftax_benchmark.py's CraftaxMetrics so the two suites'
    output JSON / plotting code can be shared."""
    model_name: str
    crafter_score: float
    achievement_unlock_rates: List[float]
    avg_unlocked_count: float
    avg_steps: float
    execution_ms_per_step: float


class CandidateSpec(NamedTuple):
    """One registry entry.

    init_fn(key, d_model, num_actions, action_feat_dim, num_costs, num_resources, target_dim) -> params
    step_fn(params, input_n, history_state) -> (DecisionVectorD, new_history_state)
        history_state is None and passed through unchanged for the
        variant_5_1..5_4 candidates (no explicit running buffer); for
        mdp_branch/transformer_branch it is the (n_hist, d_model)
        compressed-history buffer threaded across steps (stage 6 of
        IdealStructurePipeline).
    stateful: whether history_state carries real state between steps.
    """
    init_fn: Callable
    step_fn: Callable
    stateful: bool


def _flat_step(forward_fn: Callable, **fixed_kwargs) -> Callable:
    """Wraps a stateless forward_* (variant_5_1..5_4) in the (params, input_n,
    history_state) -> (decision_d, history_state) interface every registry
    entry shares."""
    def step_fn(params, input_n, history_state):
        return forward_fn(params, input_n, **fixed_kwargs), history_state
    return step_fn


def _stateful_step(forward_fn: Callable, **fixed_kwargs) -> Callable:
    """Wraps a (params, input_n, history_buffer, ...) -> (decision_d,
    updated_history_buffer) forward_* (mdp_branch / transformer_branch)."""
    def step_fn(params, input_n, history_state):
        return forward_fn(params, input_n, history_state, **fixed_kwargs)
    return step_fn


CANDIDATE_REGISTRY: Dict[str, CandidateSpec] = {
    "variant_5_1": CandidateSpec(
        v51.init_variant_5_1_parameters,
        _flat_step(v51.forward_variant_5_1, num_l=2, num_heads=8, k_mdp=3),
        stateful=False,
    ),
    "variant_5_2": CandidateSpec(
        v52.init_variant_5_2_parameters,
        _flat_step(v52.forward_variant_5_2, num_l=2, num_heads=8),
        stateful=False,
    ),
    "variant_5_3": CandidateSpec(
        v53.init_variant_5_3_parameters,
        _flat_step(v53.forward_variant_5_3, num_l=2, num_heads=8),
        stateful=False,
    ),
    "variant_5_4": CandidateSpec(
        v54.init_variant_5_4_parameters,
        _flat_step(v54.forward_variant_5_4, num_heads=8, k_mdp=3),
        stateful=False,
    ),
    "mdp_branch": CandidateSpec(
        mdpb.init_mdp_branch_parameters,
        _stateful_step(mdpb.forward_mdp_branch, num_heads=8, k_mdp=3),
        stateful=True,
    ),
    "transformer_branch": CandidateSpec(
        tfb.init_transformer_branch_parameters,
        _stateful_step(tfb.forward_transformer_branch, num_heads=8),
        stateful=True,
    ),
}


def _init_history_state(stateful: bool, d_model: int) -> Optional[jnp.ndarray]:
    return jnp.zeros((1, d_model)) if stateful else None


def train_candidate_agent(
    spec_name: str,
    spec: CandidateSpec,
    adapter: CraftaxEnvAdapter,
    params: Any,
    rng_key: jax.random.PRNGKey,
    num_episodes: int = 20,
    max_steps_per_ep: int = 50,
    checkpoint_manager: Optional[AsyncCheckpointManager] = None,
    d_model: int = 64,
    log_fn=print,
) -> Any:
    """REINFORCE policy gradient over a candidate's action_logits, terminated-
    gated value bootstrap -- same loss shape as
    src/pipeline/craftax_benchmark.py's train_craftax_rl_agent, generalized to
    any CandidateSpec.step_fn via real env steps (adapter.step)."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=1e-3))
    opt_state = optimizer.init(params)
    curr_params, curr_opt_state = params, opt_state
    keys = jax.random.split(rng_key, num_episodes)
    global_step = 0

    def loss_fn(p, input_n, history_state, action_idx, reward, terminated):
        decision_d, _ = spec.step_fn(p, input_n, history_state)
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=decision_d.action_logits[None, :], labels=action_idx[None],
        )[0]
        gamma = 0.99
        bootstrap_mask = 1.0 - terminated.astype(jnp.float32)
        bootstrapped_return = reward + gamma * bootstrap_mask * decision_d.progress_rate_pred
        return policy_loss * (-bootstrapped_return) + 0.1 * jnp.mean((decision_d.estimated_costs - 1.0) ** 2)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    for ep in range(num_episodes):
        input_n, env_state, actions_data = adapter.reset(keys[ep])
        history_state = _init_history_state(spec.stateful, d_model)
        ep_keys = jax.random.split(keys[ep], max_steps_per_ep)

        for step in range(max_steps_per_ep):
            step_key = ep_keys[step]
            _, k_env = jax.random.split(step_key)

            decision_d, next_history_state = spec.step_fn(curr_params, input_n, history_state)
            act_idx = int(jnp.argmax(decision_d.action_logits))

            next_input_n, env_state, reward, done, info = adapter.step(
                k_env, env_state, act_idx, actions_data, step_count=step, prev_history=input_n.history,
            )

            _, grads = grad_fn(
                curr_params, input_n, history_state,
                jnp.array(act_idx, dtype=jnp.int32), reward, info.get("terminated", jnp.array(False)),
            )
            updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
            curr_params = optax.apply_updates(curr_params, updates)

            input_n = next_input_n
            history_state = next_history_state
            global_step += 1
            if checkpoint_manager is not None:
                checkpoint_manager.maybe_save(curr_params, global_step, config={"candidate": spec_name})
            if done:
                break

        report_every = max(1, num_episodes // 5)
        if (ep + 1) % report_every == 0 or (ep + 1) == num_episodes:
            log_fn(f"  [{spec_name}] Completed {ep + 1} / {num_episodes} training episodes.")

    if checkpoint_manager is not None:
        checkpoint_manager.maybe_save(curr_params, global_step, config={"candidate": spec_name}, force=True)
        checkpoint_manager.wait_for_pending()

    return curr_params


def evaluate_candidate_agent(
    spec_name: str,
    spec: CandidateSpec,
    adapter: CraftaxEnvAdapter,
    params: Any,
    rng_key: jax.random.PRNGKey,
    d_model: int,
    num_episodes: int = 10,
    max_steps_per_ep: int = 50,
) -> Tuple[CandidateMetrics, List[np.ndarray]]:
    """Real env-based evaluation (mirrors
    src/pipeline/craftax_benchmark.py's evaluate_craftax_agent). Returns
    (metrics, predicted_transitions) where predicted_transitions is every
    step's real DecisionVectorD.predicted_next_state -- there is no
    ground-truth W_res for Craftax, so this is what
    src/pipeline/plotter.py's W-distribution plot draws from."""
    keys = jax.random.split(rng_key, num_episodes)
    episode_unlocked_counts = []
    achievement_matrix = np.zeros((num_episodes, NUM_ACHIEVEMENTS))
    total_steps = []
    total_time_ms = 0.0
    total_step_counts = 0
    predicted_transitions: List[np.ndarray] = []

    for ep in range(num_episodes):
        input_n, env_state, actions_data = adapter.reset(keys[ep])
        history_state = _init_history_state(spec.stateful, d_model)
        ep_steps = 0
        done = False

        while not done and ep_steps < max_steps_per_ep:
            ep_key = jax.random.fold_in(keys[ep], ep_steps)
            t0 = time.perf_counter()
            decision_d, history_state = spec.step_fn(params, input_n, history_state)
            action_idx = int(jnp.argmax(decision_d.action_logits))
            predicted_transitions.append(np.asarray(decision_d.predicted_next_state))
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0
            total_step_counts += 1

            input_n, env_state, reward, done, info = adapter.step(
                ep_key, env_state, action_idx, actions_data, step_count=ep_steps, prev_history=input_n.history,
            )
            ep_steps += 1

        if hasattr(env_state, "achievements"):
            ach_unlocked = np.array(env_state.achievements, dtype=np.float32)
            achievement_matrix[ep, :] = ach_unlocked
            episode_unlocked_counts.append(float(np.sum(ach_unlocked)))
        else:
            episode_unlocked_counts.append(0.0)
        total_steps.append(ep_steps)

    achievement_rates = [float(np.mean(achievement_matrix[:, i]) * 100.0) for i in range(NUM_ACHIEVEMENTS)]
    crafter_score = calculate_crafter_score(achievement_rates)

    metrics = CandidateMetrics(
        model_name=spec_name,
        crafter_score=crafter_score,
        achievement_unlock_rates=achievement_rates,
        avg_unlocked_count=float(np.mean(episode_unlocked_counts)),
        avg_steps=float(np.mean(total_steps)),
        execution_ms_per_step=total_time_ms / max(1, total_step_counts),
    )
    return metrics, predicted_transitions


def run_candidate_benchmark_suite(
    candidate_names: Optional[List[str]] = None,
    output_json_path: str = "output/benchmark_candidates.json",
    train_episodes: int = 20,
    eval_episodes: int = 10,
    max_steps_per_ep: int = 50,
    d_model: int = 64,
    checkpoint_root: str = "output/checkpoints",
    checkpoint_every: int = 50,
    log_fn=print,
) -> Tuple[List[CandidateMetrics], Dict[str, List[np.ndarray]]]:
    """Trains + evaluates each requested candidate (default: every registry
    entry) against the real CraftaxEnvAdapter. Resumes from the latest
    checkpoint under checkpoint_root/<name>/ if one exists (crash/timeout
    recovery). Returns (metrics_list, {candidate_name: predicted_transitions})
    for src/pipeline/plotter.py."""
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    names = candidate_names or list(CANDIDATE_REGISTRY.keys())
    adapter = CraftaxEnvAdapter()
    rng_key = jax.random.PRNGKey(2026)

    results: List[CandidateMetrics] = []
    transitions_by_model: Dict[str, List[np.ndarray]] = {}

    for name in names:
        spec = CANDIDATE_REGISTRY[name]
        k_init, k_train, k_eval, rng_key = jax.random.split(rng_key, 4)
        params = spec.init_fn(
            k_init, d_model=d_model, num_actions=adapter.num_actions, action_feat_dim=adapter.action_feat_dim,
            num_costs=adapter.num_costs, num_resources=adapter.num_resources, target_dim=8,
        )

        ckpt_mgr = AsyncCheckpointManager(os.path.join(checkpoint_root, name), save_every=checkpoint_every)
        resumed = ckpt_mgr.resume()
        if resumed is not None:
            resumed_step, _, params = resumed
            log_fn(f"[{name}] Resumed from checkpoint at step {resumed_step}.")

        log_fn(f"Training candidate '{name}' ({train_episodes} episodes)...")
        trained_params = train_candidate_agent(
            name, spec, adapter, params, k_train, num_episodes=train_episodes, max_steps_per_ep=max_steps_per_ep,
            checkpoint_manager=ckpt_mgr, d_model=d_model, log_fn=log_fn,
        )
        ckpt_mgr.close()

        log_fn(f"Evaluating candidate '{name}' ({eval_episodes} episodes)...")
        metrics, transitions = evaluate_candidate_agent(
            name, spec, adapter, trained_params, k_eval, d_model=d_model,
            num_episodes=eval_episodes, max_steps_per_ep=max_steps_per_ep,
        )
        results.append(metrics)
        transitions_by_model[name] = transitions

    export_data = [r._asdict() for r in results]
    with open(output_json_path, "w", encoding="utf-8") as jf:
        json.dump({"metrics": export_data, "achievement_names": ACHIEVEMENT_NAMES}, jf, indent=2)
    log_fn(f"Candidate benchmark dataset saved to: {output_json_path}")

    return results, transitions_by_model


if __name__ == "__main__":
    from src.pipeline.plotter import (
        plot_candidate_achievement_breakdown,
        plot_craftax_benchmark_results,
        plot_predicted_transition_distribution,
    )

    res, transitions = run_candidate_benchmark_suite()
    export = [r._asdict() for r in res]
    plot_craftax_benchmark_results(export, run_seq="Candidate-Suite")
    plot_candidate_achievement_breakdown(export)
    plot_predicted_transition_distribution(transitions)
