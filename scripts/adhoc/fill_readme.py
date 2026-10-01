"""Fill the TABLE_/ACH_ placeholders of the Phase-1 README from the aggregated result files (no hand-copied numbers)."""
import subprocess
import sys

p = "docs/experiments/2026-09-30_phase1_model_families/README.md"
s = open(p, encoding="utf-8").read()


def table(d, ref):
    out = subprocess.check_output([sys.executable, "scripts/aggregate_phase1.py", d, "--ref", ref]).decode("utf-8")
    out = out.replace(chr(13) + chr(10), chr(10))
    return out.split("\n\nPer-seed")[0].strip()


def order(tbl):
    head, rows = tbl.split("\n")[:2], tbl.split("\n")[2:]
    key = lambda r: -float(r.split("|")[6].split("±")[0])
    return "\n".join(head + sorted(rows, key=key))


s = s.replace("TABLE_100K", order(table("output/phase1/final_100k_all", "ppo_mlp")))
s = s.replace("TABLE_1M", order(table("output/phase1/final_1000k_all", "ppo_gru")))
s = s.replace("ACH_1M", open("output/phase1/final_1000k_all/ach.md", encoding="utf-8").read().strip())
open(p, "w", encoding="utf-8").write(s)
print("filled")
