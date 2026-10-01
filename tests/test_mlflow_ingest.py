"""Ingest/report round trip on tiny hand-written fixture files and a temp SQLite store (no network, no GPU)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import mlflow_ingest as mi  # noqa: E402

try:
    import mlflow  # noqa: F401
    HAVE_MLFLOW = True
except ImportError:
    HAVE_MLFLOW = False

PHASE1 = {"arm_key": "ppo_x", "arm": "x64", "protocol": "EP-A", "role": "final", "config": {"lr": 0.001}, "git_commit": "abc",
          "env_steps_total": 100, "params_total": 7, "params_deployed": 7, "seeds": [0, 1], "reward_pct_mean": 10.0,
          "reward_pct_se": 1.0, "score_pct_mean": 2.0, "score_pct_se": 0.5,
          "per_seed": [{"seed": 0, "reward_pct": 9.0, "score_pct": 1.5, "achievement_rates_pct": [1.0, 2.0],
                        "train_curve_return": [0.1, float("nan"), 0.3]},
                       {"seed": 1, "reward_pct": 11.0, "score_pct": 2.5, "achievement_rates_pct": [1.0, 2.0]}]}
CAND = {"candidate": "tb", "tag": "tb__wm_act__ac__s3", "learner": "actor_critic", "config": {"seed": 3},
        "final_sampled": {"crafter_score": 1.0, "achievement_rates": [5.0]}, "eval_history": [{"update": 1, "crafter_score": 0.5}]}
LAT = {"cfg": {"env": "e", "arm": "oracle", "actor": "transformer", "delta": 2}, "records": [
    {"update": i, "env_ticks": i * 10, "train_return": float(i), "pg": 1.0} for i in range(1, 301)],
    "n_updates": 300, "train_return_settled_mean": 5.0}


class TestMlflowIngest(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(mi.detect(PHASE1), "phase1")
        self.assertEqual(mi.detect(CAND), "result")
        self.assertEqual(mi.detect(LAT), "latency")
        self.assertIsNone(mi.detect({"x": 1}))
        self.assertIsNone(mi.detect([1]))

    def test_builders(self):
        _, p = mi.build_phase1(PHASE1, "phase1/final/x.json", 0)
        self.assertEqual(len(p.children), 2)
        self.assertEqual(p.children[0].series["train_curve_return"], [(0, 0.1), (2, 0.3)])  # NaN dropped
        _, c = mi.build_result(CAND, "a/b.result.json", 0)
        self.assertEqual(c.tags["variant"], "wm_act__ac")
        self.assertEqual(c.tags["seed"], "3")
        _, l = mi.build_latency(LAT, "lat/d2/s0/summary.json", 0)
        self.assertEqual(l.tags["oracle_control"], "true")
        self.assertLessEqual(len(l.series["train_return"]), mi.MAX_STEP_POINTS)
        self.assertNotIn("pg", l.series)

    @unittest.skipUnless(HAVE_MLFLOW, "mlflow not installed")
    def test_idempotent_ingest(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root, store = Path(td) / "out", Path(td) / "store"
            (root / "a").mkdir(parents=True)
            for name, obj in (("p.json", PHASE1), ("c.result.json", CAND), ("l.json", LAT)):
                (root / "a" / name).write_text(json.dumps(obj), encoding="utf-8")
            (root / "a" / "ptr.json").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
            cmd = [sys.executable, mi.__file__, "--root", str(root), "--store", str(store)]
            env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
            first = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
            self.assertIn("ingested 5 runs", first)  # 1 phase1 aggregate + 2 seeds + 1 result + 1 latency
            self.assertIn("'lfs_pointer': 1", first)
            second = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
            self.assertIn("ingested 0 runs", second)


if __name__ == "__main__":
    unittest.main()
