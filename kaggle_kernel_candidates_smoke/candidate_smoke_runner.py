"""
GPU-side smoke-verification runner for the DPOD.ipynb candidate architectures
(src/model/candidates/*.py), reusing src/pipeline/candidate_benchmark.py's own
train/eval functions rather than re-deriving a training loop.

This is a SMALL VERIFICATION RUN, not a Phase II-scale run (ADR-002 does not
apply here): a handful of candidates, tens of episodes, few dozen steps per
episode, small d_model. It exists to prove the candidate pipeline -- training,
real Craftax eval, AsyncCheckpointManager, and the new
plot_predicted_transition_distribution / plot_candidate_achievement_breakdown
plots -- runs correctly end-to-end on real Kaggle GPU hardware, with real
numbers (no synthetic scores; see CLAUDE.md's data-integrity rules).

Config (read from config.json written by the notebook builder):
{
  "candidates":       ["transformer_branch", "variant_5_1", "mdp_branch"],
  "train_episodes":   20,
  "eval_episodes":    5,
  "max_steps_per_ep": 20,
  "d_model":          32,
  "checkpoint_every": 25,
  "seed":             2026
}
"""

import datetime
import json
import os
import sys

import jax

sys.path.insert(0, ".")

# -- GPU assertion -----------------------------------------------------------
print(f"[{datetime.datetime.now().isoformat()}] JAX Backend : {jax.default_backend().upper()}")
print(f"[{datetime.datetime.now().isoformat()}] Devices     : {jax.devices()}")
assert jax.default_backend() in ("gpu", "tpu"), (
    f"Expected GPU/TPU backend, got: {jax.default_backend()}"
)

from src.pipeline.candidate_benchmark import (
    CANDIDATE_REGISTRY,
    run_candidate_benchmark_suite,
)
from src.pipeline.plotter import (
    plot_candidate_achievement_breakdown,
    plot_craftax_benchmark_results,
    plot_predicted_transition_distribution,
)


def load_config() -> dict:
    if os.path.exists("config.json"):
        with open("config.json") as f:
            cfg = json.load(f)
            assert isinstance(cfg, dict), f"config.json must be a JSON object, got {type(cfg)}"
            return cfg
    return json.loads(os.environ.get("SMOKE_CONFIG", "{}"))


CFG = load_config()
CANDIDATES = CFG.get("candidates", ["transformer_branch", "variant_5_1", "mdp_branch"])
TRAIN_EPISODES = int(CFG.get("train_episodes", 20))
EVAL_EPISODES = int(CFG.get("eval_episodes", 5))
MAX_STEPS_PER_EP = int(CFG.get("max_steps_per_ep", 20))
D_MODEL = int(CFG.get("d_model", 32))
CHECKPOINT_EVERY = int(CFG.get("checkpoint_every", 25))
SEED = int(CFG.get("seed", 2026))

unknown = [n for n in CANDIDATES if n not in CANDIDATE_REGISTRY]
assert not unknown, f"Unknown candidate name(s) not in CANDIDATE_REGISTRY: {unknown}"

print(f"\n{'=' * 60}")
print("  DPOD Candidate Smoke-Verification Run")
print(f"  candidates={CANDIDATES}")
print(f"  train_episodes={TRAIN_EPISODES}, eval_episodes={EVAL_EPISODES}, "
      f"max_steps_per_ep={MAX_STEPS_PER_EP}")
print(f"  d_model={D_MODEL}, checkpoint_every={CHECKPOINT_EVERY}, seed={SEED}")
print(f"{'=' * 60}\n")

os.makedirs("output", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

results, transitions_by_model = run_candidate_benchmark_suite(
    candidate_names=CANDIDATES,
    output_json_path="output/candidate_smoke_results.json",
    train_episodes=TRAIN_EPISODES,
    eval_episodes=EVAL_EPISODES,
    max_steps_per_ep=MAX_STEPS_PER_EP,
    d_model=D_MODEL,
    checkpoint_root="output/checkpoints",
    checkpoint_every=CHECKPOINT_EVERY,
    log_fn=print,
)

export = [r._asdict() for r in results]
plot_craftax_benchmark_results(export, run_seq="Candidate-Smoke-Verify")
plot_candidate_achievement_breakdown(export, run_seq="Candidate-Smoke-Verify")
plot_predicted_transition_distribution(transitions_by_model, run_seq="Candidate-Smoke-Verify")

print(f"\n{'=' * 60}")
print("  SMOKE-VERIFY RESULTS")
print(f"{'=' * 60}")
for r in results:
    print(f"  [{r.model_name}] crafter_score={r.crafter_score:.4f} "
          f"avg_unlocked={r.avg_unlocked_count:.2f} avg_steps={r.avg_steps:.1f} "
          f"ms/step={r.execution_ms_per_step:.3f}")
print(f"{'=' * 60}\n")

summary = {
    "run_type": "candidate_smoke_verify",
    "timestamp": datetime.datetime.now().isoformat(),
    "config": CFG,
    "hardware": {
        "jax_backend": jax.default_backend(),
        "devices": str(jax.devices()),
    },
    "results": export,
}
with open("output/candidate_smoke_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[{datetime.datetime.now().isoformat()}] Candidate smoke-verification run complete.")
