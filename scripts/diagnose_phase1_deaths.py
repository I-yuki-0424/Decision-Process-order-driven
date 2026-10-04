"""Death-cause and crafting-readiness diagnostic for policies saved by `run_epa_mini.py --save-params` (TASK-20261004-023).

  python scripts/diagnose_phase1_deaths.py output/phase1/recipe_t023/diag/params/tf_gru__diag__s1002.pkl --out output/phase1/recipe_t023/diag/tf_gru_s1002_deaths.json

Evaluates the first episode of 256 fresh envs with the same key stream as the headline evaluator of that run (test seed in the
pickle), so the masked achievements reproduce the run's own evaluation; adds cause of death, light level / sleep at death,
vitals, and how often the agent stood next to a table holding wood + stone. Diagnostic only: never used for selection.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, ".")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.model.epa_policies import ARM_BUILDERS  # noqa: E402
from src.pipeline.epa_diagnostics import diagnose_first_episodes, summarize  # noqa: E402
from src.pipeline.epa_harness import run_provenance, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", help="pickle written by run_epa_mini.py --save-params")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-envs", type=int, default=256)
    a = ap.parse_args()
    with open(a.params, "rb") as f:
        saved = pickle.load(f)
    arm = ARM_BUILDERS[saved["arm_key"]](**saved["arm_kwargs"])
    if arm.name != saved["arm"]:
        raise SystemExit(f"arm mismatch: rebuilt {arm.name}, saved {saved['arm']}")
    params = jax.tree_util.tree_map(jnp.asarray, saved["params"])
    key = jax.random.fold_in(jax.random.PRNGKey(saved["test_seed"]), 0xE7A1)   # = Trainer.evaluate's key for that test seed
    diag = diagnose_first_episodes(arm, params, key, a.eval_envs)
    summary, causes = summarize(diag)
    out = dict(kind="phase1_death_diagnostic", params_file=a.params, arm=saved["arm"], arm_key=saved["arm_key"],
               arm_kwargs=saved["arm_kwargs"], config=saved["config"], seed=saved["seed"], test_seed=saved["test_seed"],
               training_provenance=saved["provenance"], provenance=run_provenance(extra_files=[os.path.relpath(__file__)]),
               summary=summary,
               per_episode=dict(length=diag["length"].tolist(), cause=causes, light_at_death=diag["death_light"].tolist(),
                                asleep_at_death=diag["death_asleep"].tolist(), vitals_before_death=diag["death_vitals"].tolist(),
                                near_table_steps=diag["near_table"].tolist(), near_table_with_both_steps=diag["near_table_both"].tolist(),
                                max_wood=diag["max_wood"].tolist(), max_stone=diag["max_stone"].tolist()))
    write_json(a.out, out)
    s = summary
    print(f"{saved['arm']} seed {saved['seed']} (test {saved['test_seed']}): mean length {s['length_mean']:.0f}, causes {s['cause_counts']}, "
          f"night deaths {100 * s['death_at_night_frac']:.0f}%, asleep {100 * s['death_asleep_frac']:.0f}%, "
          f"near-table steps with wood+stone {100 * s['near_table_with_both_frac']:.1f}%, "
          f"reward_pct masked {s['headline_masked']['reward_pct']:.2f} / incl. death tick {s['incl_death_tick']['reward_pct']:.2f}")


if __name__ == "__main__":
    main()
