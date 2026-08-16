"""
Phase II GPU-Side Runner for Kaggle — v2 (Reliable Execution).

Key architectural fixes over v1:
- eval_fn is NOT compiled eagerly at module level; compiled only when called after training
- eval uses a Python for-loop (not lax.scan) to avoid XLA graph explosion
- No int() conversions on JAX tracers inside JIT-traced code
- forward pass for action selection is JIT-compiled with static args
- Checkpoints every 100K steps; logs every 10K steps for clearer progress visibility
- Graceful OOM/error handling per section

Config (read from config.json written by Cell 2):
{
  "run_id":      "phase2_001",
  "n_layers":    6,
  "beam_width":  5,
  "z_step":      64,
  "is_causal":   true,
  "train_steps": 1000000,
  "eval_steps":  500,
  "seed":        42,
  "d_model":     512,
  "num_heads":   8,
  "lr":          3e-4,
  "noise_prob":  0.15
}
"""

import datetime
import functools
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, ".")

# ── GPU assertion ─────────────────────────────────────────────────────────────
print(f"[{datetime.datetime.now().isoformat()}] JAX Backend : {jax.default_backend().upper()}")
print(f"[{datetime.datetime.now().isoformat()}] Devices     : {jax.devices()}")
assert jax.default_backend() in ("gpu", "tpu"), (
    f"Expected GPU/TPU backend, got: {jax.default_backend()}"
)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.environment.craftax_env_adapter import (
    CraftaxEnvAdapter,
    ACHIEVEMENT_NAMES,
    NUM_ACHIEVEMENTS,
    calculate_crafter_score,
)
from src.model.transformer_decision_core import (
    init_model_parameters,
    forward_decision_transformer,
)
from src.model.beam_search import beam_search_init, beam_search_step
from src.pipeline.trainer import compute_loss

# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if os.path.exists("config.json"):
        with open("config.json") as f:
            cfg = json.load(f)
            assert isinstance(cfg, dict), f"config.json must be a JSON object, got {type(cfg)}"
            return cfg
    raw = os.environ.get("PHASE2_CONFIG", "{}")
    return json.loads(raw)


CFG         = load_config()
RUN_ID      = CFG.get("run_id",      "phase2_run")
N_LAYERS    = int(CFG.get("n_layers",    6))
BEAM_WIDTH  = int(CFG.get("beam_width",  5))
Z_STEP      = int(CFG.get("z_step",      64))
IS_CAUSAL   = bool(CFG.get("is_causal",  True))
TRAIN_STEPS = int(CFG.get("train_steps", 1_000_000))
EVAL_STEPS  = int(CFG.get("eval_steps",  500))
SEED        = int(CFG.get("seed",        42))
D_MODEL     = int(CFG.get("d_model",     512))
NUM_HEADS   = int(CFG.get("num_heads",   8))
LR          = float(CFG.get("lr",        3e-4))
NOISE_PROB  = float(CFG.get("noise_prob",0.15))

# Phase II milestones (Decision 2: coal replaces gold)
MILESTONES = {}
for name in ["collect_wood", "collect_stone", "collect_coal",
             "collect_iron", "collect_diamond"]:
    if name in ACHIEVEMENT_NAMES:
        MILESTONES[name] = ACHIEVEMENT_NAMES.index(name)

print(f"\n{'='*60}")
print(f"  Phase II Run: {RUN_ID}")
print(f"  N={N_LAYERS}, k={BEAM_WIDTH}, Z={Z_STEP}, causal={IS_CAUSAL}")
print(f"  train_steps={TRAIN_STEPS:,}, eval_steps={EVAL_STEPS}")
print(f"  d_model={D_MODEL}, num_heads={NUM_HEADS}, lr={LR}")
print(f"{'='*60}\n")

os.makedirs("output/checkpoints", exist_ok=True)

# ── Environment & Model ───────────────────────────────────────────────────────
adapter = CraftaxEnvAdapter()
rng = jax.random.PRNGKey(SEED)
rng, init_rng = jax.random.split(rng)

params = init_model_parameters(
    init_rng,
    num_layers=N_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_actions=adapter.num_actions,
    action_feat_dim=adapter.action_feat_dim,
    num_costs=adapter.num_costs,
    num_resources=adapter.num_resources,
    target_dim=8,
)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Trainable Parameters: {param_count:,}")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=LR),
)
opt_state = optimizer.init(params)

# ── JIT-compiled training step ────────────────────────────────────────────────
@jax.jit
def jit_train_step(params, opt_state, input_n, target_action,
                   target_cost, target_progress, rng_key):
    (loss, metrics), grads = jax.value_and_grad(
        compute_loss, argnums=0, has_aux=True
    )(params, input_n, target_action, target_cost, target_progress, rng_key)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, metrics


# ── JIT-compiled action selection (static args for JIT caching) ───────────────
@functools.partial(jax.jit, static_argnames=("is_causal", "z_compression_interval"))
def jit_action_select(params, input_n, is_causal, z_compression_interval):
    """Forward pass for greedy action selection — compiled once per unique (is_causal, z) pair."""
    decision_d, _ = forward_decision_transformer(
        params, input_n,
        is_training=False,
        is_causal=is_causal,
        z_compression_interval=z_compression_interval,
    )
    return jnp.argmax(decision_d.action_logits)   # returns JAX int32 array, no host sync




# ── Training loop ─────────────────────────────────────────────────────────────
print(f"[{datetime.datetime.now().isoformat()}] Compiling jit_train_step + jit_action_select...")
LOG_EVERY  = 10_000   # console print every 10K steps
CKPT_EVERY = 100_000  # checkpoint every 100K steps

training_log = []
step = 0
t_start = time.perf_counter()

rng, reset_rng = jax.random.split(rng)
input_n, env_state, actions_data = adapter.reset(reset_rng)

print(f"[{datetime.datetime.now().isoformat()}] Starting training for {TRAIN_STEPS:,} steps...")

while step < TRAIN_STEPS:
    rng, step_rng, train_rng = jax.random.split(rng, 3)

    # Greedy action via JIT-compiled forward pass.
    # act_idx is a JAX DeviceArray — no host sync until we need Python int for adapter.step.
    act_idx_dev = jit_action_select(params, input_n, IS_CAUSAL, Z_STEP)
    act_idx = int(act_idx_dev)   # host sync only ONCE per step (needed for adapter.step)

    # Step environment
    next_input_n, env_state, reward, done, info = adapter.step(
        step_rng, env_state, act_idx, actions_data,
        step_count=step % 256,
        prev_history=input_n.history,
    )

    # Train
    target_action   = jnp.array(act_idx, dtype=jnp.int32)
    target_cost     = actions_data.costs[act_idx]
    target_progress = input_n.state.progress_rate

    params, opt_state, loss, metrics = jit_train_step(
        params, opt_state, input_n,
        target_action, target_cost, target_progress, train_rng,
    )

    if done:
        rng, reset_rng = jax.random.split(rng)
        input_n, env_state, actions_data = adapter.reset(reset_rng)
    else:
        input_n = next_input_n

    step += 1

    if step % LOG_EVERY == 0 or step == TRAIN_STEPS:
        elapsed = time.perf_counter() - t_start
        sps = step / max(elapsed, 1e-9)
        log_entry = {
            "step": step,
            "loss": float(loss),
            "policy_loss":   float(metrics.policy_loss),
            "validity_loss": float(metrics.validity_loss),
            "progress_loss": float(metrics.progress_loss),
            "sps": sps,
            "elapsed_s": elapsed,
        }
        training_log.append(log_entry)
        print(
            f"[{datetime.datetime.now().isoformat()}] "
            f"step={step:>8,} | loss={loss:.4f} | "
            f"policy={metrics.policy_loss:.4f} | sps={sps:,.0f}"
        )

    if step % CKPT_EVERY == 0:
        ckpt_path = f"output/checkpoints/{RUN_ID}_step{step}.pkl"
        np_tree = jax.tree_util.tree_map(lambda x: np.array(x), params)
        with open(ckpt_path, "wb") as f:
            pickle.dump(np_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [Checkpoint] {ckpt_path}")

t_train = time.perf_counter() - t_start
print(f"\nTraining complete in {t_train:.1f}s | SPS={TRAIN_STEPS/max(t_train,1e-9):,.0f}")

# ── Final weights checkpoint (W) ──────────────────────────────────────────────
final_ckpt = f"output/checkpoints/{RUN_ID}_final.pkl"
np_tree = jax.tree_util.tree_map(lambda x: np.array(x), params)
with open(final_ckpt, "wb") as f:
    pickle.dump(np_tree, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Final weights W saved: {final_ckpt}")

# ── Evaluation — Python for-loop (avoids lax.scan compilation overhead) ───────
print(f"\n[{datetime.datetime.now().isoformat()}] Running evaluation ({EVAL_STEPS} steps)...")
rng, eval_rng = jax.random.split(rng)

# JIT-compile beam_search_step separately for reuse across steps
@functools.partial(jax.jit, static_argnames=("beam_width", "num_actions",
                                               "is_causal", "z_compression_interval"))
def jit_beam_step(params, beam_state, actions_data, target,
                  beam_width, num_actions, is_causal, z_compression_interval):
    return beam_search_step(
        params, beam_state, actions_data, target,
        beam_width=beam_width, num_actions=num_actions,
        is_causal=is_causal, z_compression_interval=z_compression_interval,
    )


eval_input_n, eval_env_state, eval_actions = adapter.reset(eval_rng)
beam_state = beam_search_init(
    eval_input_n.state, eval_input_n,
    beam_width=BEAM_WIDTH, num_costs=adapter.num_costs,
)
cum_ach = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
rewards_log  = []
progress_log = []

t_eval_start = time.perf_counter()
for i in range(EVAL_STEPS):
    rng, step_rng, env_rng = jax.random.split(rng, 3)

    # Beam search step (JIT-compiled, no Python control flow inside)
    beam_state = jit_beam_step(
        params, beam_state, eval_actions, eval_input_n.target,
        beam_width=BEAM_WIDTH, num_actions=adapter.num_actions,
        is_causal=IS_CAUSAL, z_compression_interval=Z_STEP,
    )

    # Extract best action index (host sync needed for env step)
    hist_idx   = int(jnp.clip(beam_state.step_count - 1, 0, 255))
    action_idx = int(beam_state.beams.history.action_indices[0, hist_idx])

    # Step the raw Craftax environment
    _, next_env_state, reward, done, _ = adapter.raw_env.step(
        env_rng, eval_env_state,
        jnp.array(action_idx, dtype=jnp.int32),
        adapter.raw_env.default_params,
    )

    # Accumulate achievements
    if hasattr(next_env_state, "achievements"):
        step_ach = np.array(next_env_state.achievements, dtype=np.float32)
        np.maximum(cum_ach, step_ach, out=cum_ach)

    rewards_log.append(float(reward))
    progress_log.append(float(beam_state.beams.progress_rate.mean()))

    if done:
        eval_rng, reset_rng = jax.random.split(eval_rng)
        eval_input_n, eval_env_state, eval_actions = adapter.reset(reset_rng)
        beam_state = beam_search_init(
            eval_input_n.state, eval_input_n,
            beam_width=BEAM_WIDTH, num_costs=adapter.num_costs,
        )
    else:
        eval_env_state = next_env_state
        # Advance input context using raw env step output only (no adapter.step needed)
        eval_input_n, _, _, _, _ = adapter.step(
            step_rng, next_env_state, action_idx, eval_actions,
            step_count=i % 256, prev_history=eval_input_n.history,
        )

t_eval = time.perf_counter() - t_eval_start
eval_sps = EVAL_STEPS / max(t_eval, 1e-9)

# ── Compute metrics ───────────────────────────────────────────────────────────
achievement_rates = [float(cum_ach[i] * 100.0) for i in range(NUM_ACHIEVEMENTS)]
crafter_score     = calculate_crafter_score(achievement_rates)
mean_reward       = float(np.mean(rewards_log))  if rewards_log  else 0.0
mean_progress     = float(np.mean(progress_log)) if progress_log else 0.0
context_util      = (256 // Z_STEP) / 256.0 if Z_STEP > 0 else 1.0

milestone_results = {
    name: bool(cum_ach[idx] > 0)
    for name, idx in MILESTONES.items()
}

# Print eval summary
print(f"\n{'='*60}")
print(f"  EVALUATION RESULTS  {RUN_ID}")
print(f"{'='*60}")
print(f"  Crafter Score   : {crafter_score:.4f}")
print(f"  Mean Reward     : {mean_reward:.4f}")
print(f"  Mean Progress   : {mean_progress:.4f}")
print(f"  Eval SPS        : {eval_sps:,.0f} steps/sec")
for name, unlocked in milestone_results.items():
    print(f"  {name:<20}: {'YES' if unlocked else 'NO'}")
print(f"{'='*60}\n")

print("Achievement Unlock Rates:")
for i, (name, rate) in enumerate(zip(ACHIEVEMENT_NAMES, achievement_rates)):
    bar = "#" * int(rate / 5)
    print(f"  [{i:>2}] {name:<25} {rate:>6.1f}%  {bar}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "run_id": RUN_ID,
    "timestamp": datetime.datetime.now().isoformat(),
    "config": CFG,
    "hardware": {
        "jax_backend": jax.default_backend(),
        "devices": str(jax.devices()),
        "param_count": param_count,
    },
    "training": {
        "total_steps": TRAIN_STEPS,
        "total_time_s": t_train,
        "final_sps": TRAIN_STEPS / max(t_train, 1e-9),
        "log": training_log,
    },
    "evaluation": {
        "eval_steps":          EVAL_STEPS,
        "eval_time_s":         t_eval,
        "eval_sps":            eval_sps,
        "crafter_score":       crafter_score,
        "mean_reward":         mean_reward,
        "mean_progress_rate":  mean_progress,
        "context_util_ratio":  context_util,
        "milestone_results":   milestone_results,
        "achievement_rates": {
            ACHIEVEMENT_NAMES[i]: achievement_rates[i]
            for i in range(NUM_ACHIEVEMENTS)
        },
    },
    "checkpoint_path": final_ckpt,
}

results_path = f"output/{RUN_ID}_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {results_path}")

# Markdown summary
md_path = f"output/{RUN_ID}_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Phase II Run: {RUN_ID}\n\n")
    f.write(f"**Timestamp:** {results['timestamp']}  \n")
    f.write(f"**Backend:** {jax.default_backend().upper()}  \n\n")
    f.write("## Configuration\n\n| Param | Value |\n|---|---|\n")
    for k, v in CFG.items():
        f.write(f"| {k} | `{v}` |\n")
    f.write("\n## Results\n\n| Metric | Value |\n|---|---|\n")
    f.write(f"| Crafter Score | {crafter_score:.4f} |\n")
    f.write(f"| Mean Reward | {mean_reward:.4f} |\n")
    f.write(f"| Mean Progress Rate | {mean_progress:.4f} |\n")
    f.write(f"| Eval SPS | {eval_sps:,.0f} |\n")
    f.write(f"| Training SPS | {TRAIN_STEPS/max(t_train,1e-9):,.0f} |\n")
    f.write(f"| Context Util | {context_util:.3f} |\n")
    f.write("\n## Milestone Checklist\n\n")
    for name, unlocked in milestone_results.items():
        f.write(f"- [{'x' if unlocked else ' '}] {name}\n")
    f.write("\n## Achievement Unlock Rates\n\n| Achievement | Rate (%) |\n|---|---|\n")
    for name, rate in zip(ACHIEVEMENT_NAMES, achievement_rates):
        f.write(f"| {name} | {rate:.1f} |\n")
print(f"Summary saved: {md_path}")

print(f"\n[{datetime.datetime.now().isoformat()}] Phase II run {RUN_ID} complete.")
