"""Push ONE self-contained Kaggle kernel (own slug/notebook, so several can run in parallel) for the passive-WM v2 study.

  python scripts/kaggle_push_pwm.py --slug dpod-pwm-oracle --script scripts/run_passive_policy_v2.py \
      --extra-files <local files shipped verbatim, same relative path> -- --out /kaggle/working/exp --wm-dir wm ...

Fetch later with:  kaggle kernels output bfloat16/<slug> -p output_remote/<slug>
"""
import argparse, base64, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, ".")
import build_kaggle_kernel_candidates_smoke as bk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True); ap.add_argument("--script", required=True)
ap.add_argument("--extra-files", nargs="*", default=[], help="local=remote pairs")
ap.add_argument("script_args", nargs=argparse.REMAINDER)
a = ap.parse_args()
cli = [x for x in a.script_args if x != "--"]
kdir = Path(f"kaggle_kernel_{a.slug.replace('-', '_')}"); kdir.mkdir(exist_ok=True)
script = Path(a.script).read_text(encoding="utf-8")
extra = ["import base64, os\n"]
for pair in a.extra_files:
    loc, rem = pair.split("=")
    extra.append(f"os.makedirs(os.path.dirname({rem!r}) or '.', exist_ok=True)\n")
    extra.append(f"open({rem!r},'wb').write(base64.b64decode({base64.b64encode(Path(loc).read_bytes()).decode()!r}))\n")
cells = [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import subprocess, sys\n",
        "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
        "print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else '')\n",
        "import jax; print('JAX Backend:', jax.default_backend(), jax.devices(), jax.__version__)\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": bk.build_src_decode_cell()},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import os\nos.makedirs('scripts', exist_ok=True)\n", *extra,
        f"open({a.script!r},'w',encoding='utf-8').write({script!r})\n",
        f"r = subprocess.run([sys.executable, '-u', {a.script!r}] + {cli!r})\n", "print('exit', r.returncode)\n"]},
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
(kdir / "nb.ipynb").write_text(json.dumps(nb), encoding="utf-8")
(kdir / "kernel-metadata.json").write_text(json.dumps({
    "id": f"bfloat16/{a.slug}", "title": a.slug.replace("-", " ").title(), "code_file": "nb.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2), encoding="utf-8")
print(subprocess.run(["kaggle", "kernels", "push", "-p", str(kdir)], capture_output=True, text=True).stdout)
