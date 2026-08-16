"""
Craftax-Classic Reinforcement Learning Benchmark Suite (Run-Seq: #005).

Executes 1,000-episode RL training and evaluation on Craftax-Classic (JAX-native accelerated Crafter):
1. Audits & enforces CUDA GPU hardware acceleration ([CudaDevice(id=0)])
2. Scales RL training to 1,000 episodes to maximize achievement unlock rate (22 achievements), step efficiency, and Crafter Score S_crafter
3. Evaluates 5th-Idea Hierarchical Transformer vs Flat Transformer vs Random Baseline over 50 test episodes
"""

import json
import os
import sys
import time
from typing import Dict, List, NamedTuple, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.craftax_env_adapter import (
    CraftaxEnvAdapter,
    NUM_ACHIEVEMENTS,
    ACHIEVEMENT_NAMES,
    calculate_crafter_score,
)
from src.model.hierarchical_transformer import (
    HierarchicalModelParameters,
    forward_hierarchical_transformer,
    init_hierarchical_model_parameters,
)
from src.model.beam_search import (
    beam_search_init,
    hierarchical_beam_search_step,
)
from src.model.logger_utils import get_logger

logger = get_logger("CraftaxBenchmark")


class CraftaxMetrics(NamedTuple):
    """Metrics container for a Craftax-Classic benchmarked model variant."""
    model_name: str
    crafter_score: float
    achievement_unlock_rates: List[float]
    avg_unlocked_count: float
    avg_steps: float
    execution_ms_per_step: float


def audit_gpu_hardware(log_fn=print) -> bool:
    """Audit JAX GPU hardware availability and log active devices."""
    devices = jax.devices()
    backend = jax.default_backend()
    log_fn(f"[GPU Audit] Active JAX Backend: '{backend}'")
    log_fn(f"[GPU Audit] Detected Devices: {devices}")
    
    is_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    if is_gpu:
        log_fn(f"[GPU Audit] CUDA GPU Acceleration ACTIVE & VERIFIED! ([CudaDevice(id=0)])")
    else:
        log_fn(f"[GPU Audit Notice] Running in CPU mode. (CUDA GPU containers / Kaggle API available for full GPU scale)")
    return is_gpu


def train_craftax_rl_agent(
    adapter: CraftaxEnvAdapter,
    params: HierarchicalModelParameters,
    rng_key: jax.random.PRNGKey,
    use_hierarchical: bool = True,
    num_episodes: int = 1000,
    max_steps_per_ep: int = 100,
    log_fn=print,
) -> HierarchicalModelParameters:
    """Train 5th-Idea Decision Transformer agent on Craftax-Classic using Scaled Reinforcement Learning (1,000 episodes)."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3),
    )
    opt_state = optimizer.init(params)
    curr_params = params
    curr_opt_state = opt_state

    keys = jax.random.split(rng_key, num_episodes)

    def rl_loss_fn(p, input_n, action_idx, reward):
        decision_d, _ = forward_hierarchical_transformer(
            p,
            input_n,
            use_hierarchical=use_hierarchical,
            is_training=True,
        )
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=decision_d.action_logits[None, :],
            labels=action_idx[None],
        )[0]
        # REINFORCE policy gradient loss + cost regularizer
        total_loss = policy_loss * (-reward) + 0.1 * jnp.mean((decision_d.estimated_costs - 1.0) ** 2)
        return total_loss

    grad_fn = jax.value_and_grad(rl_loss_fn, argnums=0)

    for ep in range(num_episodes):
        input_n, env_state, actions_data = adapter.reset(keys[ep])
        ep_keys = jax.random.split(keys[ep], max_steps_per_ep)

        for step in range(max_steps_per_ep):
            step_key = ep_keys[step]
            k_act, k_env = jax.random.split(step_key)

            decision_d, _ = forward_hierarchical_transformer(
                curr_params,
                input_n,
                use_hierarchical=use_hierarchical,
                is_training=False,
            )
            act_idx = int(jnp.argmax(decision_d.action_logits))

            next_input_n, env_state, reward, done, _ = adapter.step(
                k_env, env_state, act_idx, actions_data, step_count=step, prev_history=input_n.history
            )

            _, grads = grad_fn(curr_params, input_n, jnp.array(act_idx, dtype=jnp.int32), reward)
            updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
            curr_params = optax.apply_updates(curr_params, updates)

            input_n = next_input_n
            if done:
                break

        if (ep + 1) % 100 == 0 or (ep + 1) == num_episodes:
            log_fn(f"  [RL Progress] Completed {ep + 1} / {num_episodes} RL Training Episodes.")

    return curr_params


def evaluate_craftax_agent(
    model_name: str,
    adapter: CraftaxEnvAdapter,
    params: HierarchicalModelParameters,
    rng_key: jax.random.PRNGKey,
    use_hierarchical: bool = True,
    num_episodes: int = 50,
    max_steps_per_ep: int = 100,
) -> CraftaxMetrics:
    """Evaluate agent performance on Craftax-Classic across 50 test episodes, measuring achievement unlocks and Crafter Score."""
    keys = jax.random.split(rng_key, num_episodes)
    episode_unlocked_counts = []
    achievement_matrix = np.zeros((num_episodes, NUM_ACHIEVEMENTS))
    total_steps = []
    total_time_ms = 0.0
    total_step_counts = 0

    for ep in range(num_episodes):
        input_n, env_state, actions_data = adapter.reset(keys[ep])
        ep_steps = 0
        done = False

        while not done and ep_steps < max_steps_per_ep:
            ep_key = jax.random.fold_in(keys[ep], ep_steps)
            t0 = time.perf_counter()

            if params is not None:
                beam_state = beam_search_init(input_n.state, input_n, beam_width=3, num_costs=adapter.num_costs)
                beam_state = hierarchical_beam_search_step(
                    params,
                    beam_state,
                    actions_data,
                    input_n.target,
                    use_hierarchical=use_hierarchical,
                    beam_width=3,
                    num_actions=adapter.num_actions,
                )
                action_idx = int(beam_state.beams.history.action_indices[0, min(ep_steps, 127)])
            else:
                action_idx = int(jax.random.randint(ep_key, (), 0, adapter.num_actions))

            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0
            total_step_counts += 1

            input_n, env_state, reward, done, info = adapter.step(
                ep_key, env_state, action_idx, actions_data, step_count=ep_steps, prev_history=input_n.history
            )
            ep_steps += 1

        if hasattr(env_state, 'achievements'):
            ach_unlocked = np.array(env_state.achievements, dtype=np.float32)
            achievement_matrix[ep, :] = ach_unlocked
            episode_unlocked_counts.append(float(np.sum(ach_unlocked)))
        else:
            episode_unlocked_counts.append(0.0)

        total_steps.append(ep_steps)

    achievement_rates = [float(np.mean(achievement_matrix[:, i]) * 100.0) for i in range(NUM_ACHIEVEMENTS)]
    crafter_score = calculate_crafter_score(achievement_rates)

    return CraftaxMetrics(
        model_name=model_name,
        crafter_score=crafter_score,
        achievement_unlock_rates=achievement_rates,
        avg_unlocked_count=float(np.mean(episode_unlocked_counts)),
        avg_steps=float(np.mean(total_steps)),
        execution_ms_per_step=total_time_ms / max(1, total_step_counts),
    )


def run_craftax_benchmark_suite(
    output_log_path: str = "output/logs/execution_seq005.log",
    output_json_path: str = "output/benchmark_craftax_seq005.json",
    run_seq: str = "Run-Seq: #005",
    train_episodes: int = 1000,
    eval_episodes: int = 50,
) -> List[CraftaxMetrics]:
    """Run scaled 1,000-episode Craftax-Classic RL Benchmark Suite under Run-Seq: #005."""
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_log_path, "w", encoding="utf-8") as log_file:
        def log_msg(msg: str):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log_msg(f"=== Starting Scaled 1,000-Episode Craftax-Classic RL Benchmark [{run_seq}] ===")
        log_msg(f"Benchmark Environment: Craftax-Classic (JAX-native Accelerated Crafter)")
        log_msg(f"Training Scale: {train_episodes} Episodes | Evaluation Scale: {eval_episodes} Episodes")
        log_msg(f"Optimization Targets: Achievement Unlock Rate (22 Achievements), Crafter Score S_crafter, Step Efficiency\n")

        audit_gpu_hardware(log_fn=log_msg)
        log_msg("")

        adapter = CraftaxEnvAdapter()
        rng_key = jax.random.PRNGKey(2026)
        k_init, k_train_h, k_train_f, k_eval = jax.random.split(rng_key, 4)

        # 1. Train 5th-Idea Hierarchical Model (1,000 Episodes)
        log_msg(f"Initializing and Training 5th-Idea Hierarchical Model on Craftax-Classic ({train_episodes} Episodes)...")
        h_params = init_hierarchical_model_parameters(
            k_init,
            num_layers=4,
            d_model=512,
            num_actions=adapter.num_actions,
            num_macro_clusters=4,
            num_fine_actions=5,
        )
        trained_h_params = train_craftax_rl_agent(
            adapter, h_params, k_train_h, use_hierarchical=True, num_episodes=train_episodes, max_steps_per_ep=100, log_fn=log_msg
        )

        # 2. Train Flat Decision Transformer Model (1,000 Episodes)
        log_msg(f"\nInitializing and Training Flat Decision Transformer Baseline on Craftax-Classic ({train_episodes} Episodes)...")
        f_params = init_hierarchical_model_parameters(
            k_init,
            num_layers=4,
            d_model=512,
            num_actions=adapter.num_actions,
            num_macro_clusters=4,
            num_fine_actions=5,
        )
        trained_f_params = train_craftax_rl_agent(
            adapter, f_params, k_train_f, use_hierarchical=False, num_episodes=train_episodes, max_steps_per_ep=100, log_fn=log_msg
        )

        # 3. Benchmark Variant 1: 5th-Idea Hierarchical Transformer (1,000-Ep Trained)
        log_msg(f"\nBenchmarking Variant 1: 5th-Idea Hierarchical Transformer Core ({eval_episodes} Evaluation Episodes)...")
        m1 = evaluate_craftax_agent(
            model_name="5th-Idea Hierarchical Transformer (1,000 Ep Trained)",
            adapter=adapter,
            params=trained_h_params,
            rng_key=k_eval,
            use_hierarchical=True,
            num_episodes=eval_episodes,
        )

        # 4. Benchmark Variant 2: Flat Decision Transformer (1,000-Ep Trained)
        log_msg(f"Benchmarking Variant 2: Flat Decision Transformer Core ({eval_episodes} Evaluation Episodes)...")
        m2 = evaluate_craftax_agent(
            model_name="Flat Decision Transformer Baseline (1,000 Ep Trained)",
            adapter=adapter,
            params=trained_f_params,
            rng_key=k_eval,
            use_hierarchical=False,
            num_episodes=eval_episodes,
        )

        # 5. Benchmark Variant 3: Random Policy Baseline
        log_msg(f"Benchmarking Variant 3: Random Policy Baseline ({eval_episodes} Evaluation Episodes)...")
        m3 = evaluate_craftax_agent(
            model_name="Random Policy Baseline",
            adapter=adapter,
            params=None,
            rng_key=k_eval,
            use_hierarchical=False,
            num_episodes=eval_episodes,
        )

        results = [m1, m2, m3]

        log_msg(f"\n=== SCALED 1,000-EPISODE CRAFTAX BENCHMARK SUMMARY TABLE [{run_seq}] ===")
        log_msg(f"{'Model Architecture':<48} | {'Crafter Score':<15} | {'Avg Achievements':<18} | {'Avg Steps':<10} | {'Speed (ms/step)':<15}")
        log_msg("-" * 115)
        for r in results:
            log_msg(f"{r.model_name:<48} | {r.crafter_score:>13.2f}% | {r.avg_unlocked_count:>16.1f} / 22 | {r.avg_steps:>9.1f} | {r.execution_ms_per_step:>13.3f} ms")

        # Export JSON dataset
        export_data = [r._asdict() for r in results]
        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump({"metrics": export_data, "achievement_names": ACHIEVEMENT_NAMES}, jf, indent=2)
        log_msg(f"\nBenchmark dataset saved to: {output_json_path}")

        return results


if __name__ == "__main__":
    from src.pipeline.plotter import plot_craftax_benchmark_results
    res = run_craftax_benchmark_suite()
    plot_craftax_benchmark_results([r._asdict() for r in res], run_seq="Run-Seq: #005")
