import json
import os
import sys
import time
import datetime
import jax
import jax.numpy as jnp
import optax
import numpy as np

sys.path.insert(0, '.')

from src.environment.craftax_env_adapter import CraftaxEnvAdapter, calculate_crafter_score
from src.model.hierarchical_transformer import (
    init_hierarchical_model_parameters,
    forward_hierarchical_transformer,
)

def count_parameters(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))

def run_kaggle_verification():
    print(f"[{datetime.datetime.now().isoformat()}] ===========================================================")
    print(f"[{datetime.datetime.now().isoformat()}]   KAGGLE GPU JAX-ACCELERATED RE-TRAINING & BENCHMARKING SUITE  ")
    print(f"[{datetime.datetime.now().isoformat()}] ===========================================================")

    devices = jax.devices()
    print(f"JAX Backend Accelerator : {jax.default_backend().upper()}")
    print(f"Available Devices       : {devices}")

    os.makedirs("output", exist_ok=True)
    env = CraftaxEnvAdapter()
    base_rng = jax.random.PRNGKey(42)

    cfg = {"name": "5th-Idea Hierarchical (L=8)", "num_layers": 8, "d_model": 256, "num_heads": 4, "d_ff": 512}
    print(f"\n[{datetime.datetime.now().isoformat()}] Model: {cfg['name']} | Config: L={cfg['num_layers']}, d_model={cfg['d_model']}")

    base_rng, init_rng = jax.random.split(base_rng)
    params = init_hierarchical_model_parameters(
        init_rng, num_layers=cfg["num_layers"], d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], num_actions=env.num_actions
    )
    param_count = count_parameters(params)
    print(f"[{datetime.datetime.now().isoformat()}] Trainable Parameters: {param_count:,}")

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(p, o):
        dec, _ = forward_hierarchical_transformer(p, o, is_training=True)
        return jnp.mean(jnp.square(dec.action_logits))

    @jax.jit
    def train_step(p, opt_st, o):
        loss_val, grads = jax.value_and_grad(loss_fn)(p, o)
        updates, new_opt_st = optimizer.update(grads, opt_st, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_st, loss_val

    @jax.jit
    def rollout_chunk(carry, _):
        # carry: (params, opt_state, obs, env_state, act_data, rng)
        p, opt_st, o, e_st, a_data, r_key = carry
        
        p, opt_st, loss_val = train_step(p, opt_st, o)
        
        r_key, step_key = jax.random.split(r_key)
        dec, _ = forward_hierarchical_transformer(p, o, is_training=False)
        action = jnp.argmax(dec.action_logits)
        
        next_o, next_e_st, reward, done, _ = env.step(step_key, e_st, action, a_data)
        
        # Simple auto-reset logic for continuous execution
        r_key, reset_key = jax.random.split(r_key)
        reset_o, reset_e_st, reset_a_data = env.reset(reset_key)
        
        # If done, take reset state, else take next state
        next_o = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_o, next_o)
        next_e_st = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_e_st, next_e_st)
        a_data = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_a_data, a_data)

        return (p, opt_st, next_o, next_e_st, a_data, r_key), (loss_val, reward, done)

    @jax.jit
    def execute_chunk(carry):
        # Run 1000 steps completely compiled in JAX
        return jax.lax.scan(rollout_chunk, carry, None, length=1000)

    # Initialize environment
    base_rng, reset_rng = jax.random.split(base_rng)
    obs, env_state, act_data = env.reset(reset_rng)
    
    carry = (params, opt_state, obs, env_state, act_data, base_rng)
    
    # Active Tracking Loop
    print(f"[{datetime.datetime.now().isoformat()}] Starting Accelerated Training (10,000 steps)...")
    t_start = time.perf_counter()
    
    total_steps = 0
    all_rewards = []
    
    for epoch in range(10):  # 10 epochs * 1000 steps = 10,000 steps
        carry, (losses, rewards, dones) = execute_chunk(carry)
        total_steps += 1000
        mean_loss = jnp.mean(losses)
        mean_reward = jnp.mean(rewards)
        all_rewards.extend(np.array(rewards).tolist())
        print(f"[{datetime.datetime.now().isoformat()}] Epoch {epoch+1}/10 | Steps: {total_steps} | Mean Loss: {mean_loss:.4f} | Mean Step Reward: {mean_reward:.4f}")
    
    t_elapsed = time.perf_counter() - t_start
    sps = total_steps / max(t_elapsed, 1e-6)
    
    params = carry[0] # Updated parameters
    
    print(f"[{datetime.datetime.now().isoformat()}] Accelerated Training Complete. SPS: {sps:,.2f}")
    
    # Save Results
    output_path = "output/kaggle_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "jax_backend": jax.default_backend(),
            "model_name": cfg["name"],
            "trainable_parameters": param_count,
            "total_steps": total_steps,
            "sps_throughput": float(sps),
            "mean_reward": float(np.mean(all_rewards)),
        }, f, indent=2)

    print(f"[{datetime.datetime.now().isoformat()}] Metrics saved to: {output_path}")

if __name__ == "__main__":
    run_kaggle_verification()
