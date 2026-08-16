"""
Main Entry Point for 4th-Idea JAX Decision Model & Gymnax Harness.

Orchestrates:
1. Model Parameter Initialization
2. Gymnax Environment Setup
3. Training with Noise Injection & Optax
4. Evaluation with Beam Search & Greedy Single-Pass
5. Result Collection & Plot Generation
"""

import sys
import jax
import jax.numpy as jnp
import optax

from src.environment.gymnax_decision_env import DecisionProcessEnv, EnvParams
from src.model.logger_utils import format_error, get_logger, validate_input_context
from src.model.transformer_decision_core import init_model_parameters
from src.pipeline.evaluator import evaluate_beam_search, evaluate_greedy
from src.pipeline.plotter import plot_evaluation_comparison, plot_training_curves
from src.pipeline.trainer import train_step

logger = get_logger("4thIdeaMain")


def main():
    logger.info("Initializing 4th-Idea JAX Decision Process Framework...")

    # 1. Setup PRNG Keys & Environment
    rng_key = jax.random.PRNGKey(42)
    env_params = EnvParams(
        max_steps=150,
        num_actions=16,
        action_feat_dim=32,
        num_costs=4,
        num_resources=8,
        target_dim=8,
        history_len=128,
        goal_tolerance=0.05,
    )
    env = DecisionProcessEnv(params=env_params)

    # 2. Initialize Model Parameters
    k_init, k_train, k_eval = jax.random.split(rng_key, 3)
    params = init_model_parameters(
        k_init,
        num_layers=4,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_actions=env_params.num_actions,
        action_feat_dim=env_params.action_feat_dim,
        num_costs=env_params.num_costs,
        num_resources=env_params.num_resources,
        target_dim=env_params.target_dim,
    )
    logger.info("Model Parameters initialized successfully (d_model=512, num_heads=8, 4 layers).")

    # 3. Setup Optax Optimizer
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    # 4. Training Loop Simulation (Gymnax)
    logger.info("Starting Training Loop with Noise Injection...")
    total_losses = []
    policy_losses = []
    validity_losses = []

    obs, env_state, actions_data = env.reset(k_train)
    
    # Verify input context structure statically
    is_valid, msg = validate_input_context(obs)
    if not is_valid:
        logger.error(format_error("Main", "InputValidationError", msg))
        sys.exit(1)
    logger.info("Input context static validation passed: " + msg)

    current_params = params
    current_opt_state = opt_state

    # Perform mock training steps over Gymnax environment
    train_keys = jax.random.split(k_train, 20)
    for i in range(20):
        target_action = jnp.array(i % env_params.num_actions)
        target_cost = jnp.array([5.0, 2.0, 1.0, 0.5])
        target_progress = jnp.array(min(1.0, (i + 1) / 20.0))

        current_params, current_opt_state, metrics = train_step(
            current_params,
            current_opt_state,
            optimizer,
            obs,
            target_action,
            target_cost,
            target_progress,
            train_keys[i],
        )

        total_losses.append(float(metrics.total_loss))
        policy_losses.append(float(metrics.policy_loss))
        validity_losses.append(float(metrics.validity_loss))

    logger.info(f"Training completed. Final Total Loss: {total_losses[-1]:.4f}")

    # 5. Evaluation Loop (Greedy vs 4th-Idea Beam Search)
    logger.info("Running Evaluation Harness (Greedy vs. 4th-Idea Beam Search)...")
    greedy_res = evaluate_greedy(current_params, env, k_eval, num_episodes=5)
    beam_res = evaluate_beam_search(current_params, env, k_eval, beam_width=5, num_episodes=5)

    logger.info(f"Greedy Single-Pass Success Rate: {greedy_res.success_rate:.1%}")
    logger.info(f"4th-Idea Beam Search Success Rate: {beam_res.success_rate:.1%}")

    # 6. Plot Results
    logger.info("Generating Plot Visualizations...")
    plot_training_curves(total_losses, policy_losses, validity_losses)
    plot_evaluation_comparison(greedy_res, beam_res)
    logger.info("All plots generated successfully in output/plots/.")

    return 0


if __name__ == "__main__":
    main()
