"""Diagnostic (throwaway, not part of the pipeline): why is greedy eval_return << stochastic train_return at
the same checkpoint (TASK-20260928-013's flagged finding)? Hypothesis: eval always does a FRESH reset_envs()
and episode_summary only averages over done==True rows within a fixed `eval_cycles` window, while train's
`ws` persists across the whole run so completions are a steady, well-mixed stream. If most eval envs simply
haven't died yet by tick 64, only the fastest (likely worst) subset gets counted -> biased low mean_return.
Test: reload the real trained checkpoint and re-run real greedy evaluation at several eval_cycles budgets,
using true env steps only (no synthetic scores).
"""
import sys
sys.path.insert(0, ".")
import jax
from src.model.checkpoint import load_pytree_checkpoint
from src.pipeline.chunk_ppo import ChunkPPO, ChunkPPOConfig

CKPT = "output/latency_chunking/craftax_calibration/s0/checkpoints/step_15.pkl"

payload = load_pytree_checkpoint(CKPT)
cfg_dict = payload["config"]["cfg"]
cfg = ChunkPPOConfig(**cfg_dict)
st = payload["params"]["st"]
print("loaded checkpoint at step", payload["step"], "num_envs", cfg.num_envs)

for ec in (64, 128, 256):
    runner = ChunkPPO(cfg._replace(eval_cycles=ec))
    key = jax.random.PRNGKey(12345)
    ev = runner.evaluate(st, key, greedy=True)
    done_frac = ev["episodes"] / cfg.num_envs
    print(f"[greedy]     eval_cycles={ec:4d}  mean_return={ev['mean_return']:.4f}  "
          f"episodes={ev['episodes']:4d}  done_frac={done_frac:.4f}  terminal_rate={ev['terminal_rate']:.4f}")

for ec in (64, 128, 256):
    runner = ChunkPPO(cfg._replace(eval_cycles=ec))
    key = jax.random.PRNGKey(12345)
    ev = runner.evaluate(st, key, greedy=False)
    done_frac = ev["episodes"] / cfg.num_envs
    print(f"[stochastic] eval_cycles={ec:4d}  mean_return={ev['mean_return']:.4f}  "
          f"episodes={ev['episodes']:4d}  done_frac={done_frac:.4f}  terminal_rate={ev['terminal_rate']:.4f}")
