"""MLflow tooling: the achievement index/names used for metric keys must match the env adapter; score formula must agree."""
import ast
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from mlflow_ingest import ACHIEVEMENT_NAMES, _ach, crafter_score  # noqa: E402


def adapter_names():
    tree = ast.parse((REPO / "src/environment/craftax_env_adapter.py").read_text(encoding="utf-8"))  # no jax import needed
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ACHIEVEMENT_NAMES":
            return ast.literal_eval(n.value)


class TestMlflowTools(unittest.TestCase):
    def test_names_match_adapter(self):
        self.assertEqual(ACHIEVEMENT_NAMES, adapter_names())

    def test_metric_keys_are_numbered_and_named(self):
        keys = list(_ach([float(i) for i in range(22)], None))
        self.assertEqual(keys[0], "ach_rate_pct/00_collect_wood")
        self.assertEqual(keys[21], "ach_rate_pct/21_make_iron_sword")

    def test_crafter_score_is_geometric_mean(self):
        r = [50.0] * 22
        self.assertAlmostEqual(crafter_score(r), 50.0)
        self.assertAlmostEqual(crafter_score([0.0] * 21 + [100.0]), math.exp(math.log(101) / 22) - 1)


if __name__ == "__main__":
    unittest.main()
