"""Ingest/report rules on tiny hand-written fixture files and a temp SQLite store (no network, no GPU)."""
import json
import os
import subprocess
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

PHASE1 = {"arm_key": "ppo_gru", "arm": "gru256", "protocol": "EP-A", "role": "final", "phase": 1,
          "config": {"lr": 0.001, "total_steps": 100, "gamma": 0.99, "ent": 0.01}, "arm_kwargs": {"d": 64},
          "git_commit": "abc", "env_steps_total": 100, "params_total": 7, "params_deployed": 7, "seeds": [0, 1],
          "per_seed": [{"seed": 0, "reward_pct": 9.0, "score_pct": 1.5, "achievement_rates_pct": [1.0, 2.0],
                        "train_curve_return": [0.1, float("nan"), 0.3]},
                       {"seed": 1, "reward_pct": 11.0, "score_pct": 2.5, "achievement_rates_pct": [1.0, 2.0]}]}
CAND = {"candidate": "tb", "tag": "tb__wm_act__ac__s3", "learner": "actor_critic", "config": {"seed": 3, "lr": 0.1, "T": 250, "zzz": 1},
        "final_sampled": {"crafter_score": 1.0, "achievement_rates": [5.0]}, "eval_history": [{"update": 1, "crafter_score": 0.5}]}
LAT = {"cfg": {"env": "e", "arm": "oracle", "actor": "transformer", "delta": 2, "k": 4}, "records": [
    {"update": i, "env_ticks": i * 10, "train_return": float(i), "pg": 1.0} for i in range(1, 301)],
    "n_updates": 300, "train_return_settled_mean": 5.0}
FAIL = {"model_id": "Base_O02_D00_S00", "phase": 1, "end_reason": "oom", "note": "x", "steps_completed": 10}


class TestMlflowIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sch = mi.Schema()

    def test_detect(self):
        self.assertEqual([mi.detect(x) for x in (PHASE1, CAND, LAT, FAIL)], ["phase1", "result", "latency", "failure"])
        self.assertIsNone(mi.detect({"x": 1}))
        self.assertIsNone(mi.detect([1]))

    def test_official_param_names(self):
        for alias in ("lr", "LR", "Learning_rate", "learning-rate"):
            self.assertEqual(self.sch.official(alias, "phase1"), "train.learning_rate")
        self.assertEqual(self.sch.official("k", "latency"), "arch.commit_length")
        self.assertEqual(self.sch.official("brand_new", "phase1"), "x.brand_new")
        self.assertEqual(self.sch.unknown["brand_new"], 1)

    def test_phase1_per_seed_runs(self):
        runs = mi.build_phase1(self.sch, PHASE1, "phase1/final/x.json", 1_700_000_000_000)
        self.assertEqual(len(runs), 2)  # one run per seed, no aggregate run
        r = runs[0]
        self.assertEqual((r.experiment, r.model_id), ("phase-1", "Base_O02_D00_S00"))
        self.assertTrue(r.name.startswith("Base_O02_D00_S00_2023"))
        self.assertEqual(r.params["train.learning_rate"], "0.001")
        self.assertEqual(r.params["arch.d_model"], "64")
        self.assertEqual(r.params["seed.train"], "0")
        self.assertEqual(r.series["train/return"], [(0, 0.1), (2, 0.3)])  # NaN dropped

    def test_legacy_runs(self):
        c = mi.build_result(self.sch, CAND, "a/b.result.json", 0)[0]
        self.assertEqual((c.experiment, c.model_id, c.tags["variant"], c.tags["seed"]), ("phase-0-legacy", "UNREGISTERED", "wm_act__ac", "3"))
        self.assertEqual(c.params["env.episode_limit"], "250")
        self.assertIn("x.zzz", c.params)
        lt = mi.build_latency(self.sch, LAT, "lat/s0/summary.json", 0)[0]
        self.assertEqual(lt.tags["oracle_control"], "true")
        self.assertLessEqual(len(lt.series["curve/train_return"]), mi.MAX_STEP_POINTS)
        self.assertNotIn("pg", " ".join(lt.series))

    def test_failure_status(self):
        f = mi.build_failure(self.sch, FAIL, "failures/x.json", 0)[0]
        self.assertEqual((f.status, f.tags["end_reason"], f.experiment), ("FAILED", "oom", "phase-1"))
        self.assertEqual((f.params["seed.train"], f.params["seed.test"]), ("42", "424"))
        k = mi.build_failure(self.sch, dict(FAIL, end_reason="interrupted"), "failures/y.json", 0)[0]
        self.assertEqual(k.status, "KILLED")
        with self.assertRaises(ValueError):
            mi.build_failure(self.sch, dict(FAIL, model_id="Nope_O01_D00_S00"), "failures/z.json", 0)
        with self.assertRaises(ValueError):
            mi.build_failure(self.sch, dict(FAIL, end_reason="completed"), "failures/z.json", 0)

    def test_registry_is_valid(self):
        for mid, m in self.sch.models.items():
            self.assertRegex(mid, r"^[A-Za-z0-9]+_O\d{2}_D\d{2}_S\d{2}$")
            if m.get("proper_name"):
                self.assertEqual(m.get("proper_name_by"), "operator")

    @unittest.skipUnless(HAVE_MLFLOW, "mlflow not installed")
    def test_idempotent_ingest(self):
        with tempfile.TemporaryDirectory() as td:
            root, store = Path(td) / "out", Path(td) / "store"
            (root / "a").mkdir(parents=True)
            for name, obj in (("p.json", PHASE1), ("c.result.json", CAND), ("l.json", LAT), ("f.json", FAIL)):
                (root / "a" / name).write_text(json.dumps(obj), encoding="utf-8")
            (root / "a" / "ptr.json").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
            cmd = [sys.executable, mi.__file__, "--root", str(root), "--store", str(store)]
            env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
            first = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
            self.assertIn("ingested 5 runs", first)  # 2 seeds + 1 result + 1 latency + 1 failure
            self.assertIn("'lfs_pointer': 1", first)
            self.assertIn("ingested 0 runs", subprocess.run(cmd, capture_output=True, text=True, env=env).stdout)


if __name__ == "__main__":
    unittest.main()
