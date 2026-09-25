"""
Kaggle Notebook Builder for the DPOD Candidate Smoke-Verification Run.

Adapts build_kaggle_kernel.py's src-embedding logic to a NEW, separate kernel
(bfloat16/dpod-candidates-smoke-verify) so the existing production kernels
(craftax-classic-1000-episode-rl-benchmark, 4th/5th-idea benchmarks) are never
touched. Generates a self-contained .ipynb that:
1. Installs packages (craftax, optax) and verifies the GPU backend.
2. Decodes the full src/ tree from base64.
3. Writes config.json with the candidate list + small step counts.
4. Executes kaggle_kernel_candidates_smoke/candidate_smoke_runner.py.

This is a SMALL verification run, not a Phase II-scale run -- see
kaggle_kernel_candidates_smoke/candidate_smoke_runner.py's docstring.

Usage:
    python build_kaggle_kernel_candidates_smoke.py \\
        --candidates transformer_branch variant_5_1 mdp_branch \\
        --train-episodes 20 --eval-episodes 5 --max-steps-per-ep 20 --d-model 32
"""

import argparse
import base64
import json
import os


KERNEL_DIR = "kaggle_kernel_candidates_smoke"
NOTEBOOK_OUT = os.path.join(KERNEL_DIR, "candidates_smoke_verify.ipynb")
RUNNER_PATH = os.path.join(KERNEL_DIR, "candidate_smoke_runner.py")


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_src_decode_cell() -> list:
    """Return source lines that reconstruct src/ from base64 on the Kaggle machine."""
    lines = [
        "import base64, os\n",
        "os.makedirs('src', exist_ok=True)\n",
        "with open('src/__init__.py', 'w') as _f: _f.write('')\n",
    ]
    for root, dirs, files in os.walk("src"):
        for d in sorted(dirs):
            dir_path = os.path.join(root, d).replace("\\", "/")
            lines.append(f"os.makedirs('{dir_path}', exist_ok=True)\n")
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            unix_path = fpath.replace("\\", "/")
            b64 = b64_file(fpath)
            lines.append(
                f"with open('{unix_path}', 'wb') as _f:\n"
                f"    _f.write(base64.b64decode('{b64}'))\n"
            )
    return lines


def generate_notebook(cfg: dict) -> None:
    cells = []

    # -- Cell 0: Package install + hardware check --
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# DPOD Candidate Smoke-Verify: Package Installation & GPU Verification\n",
            "# Kaggle GPU kernels have JAX (GPU build) pre-installed -- do NOT reinstall jax/jaxlib.\n",
            "import subprocess, sys\n",
            "result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
            "print(result.stdout[-2000:] if result.stdout else '')\n",
            "if result.returncode != 0:\n",
            "    print('PIP STDERR:', result.stderr[-2000:])\n",
            "    raise RuntimeError('pip install failed -- see stderr above')\n",
            "import jax, jaxlib, optax\n",
            "import craftax\n",
            "from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv\n",
            "print('craftax import OK:', CraftaxClassicSymbolicEnv)\n",
            "print('JAX Backend:', jax.default_backend())\n",
            "print('JAX Devices:', jax.devices())\n",
            "print('JAX version:', jax.__version__)\n",
        ],
    })

    # -- Cell 1: Decode src/ tree --
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": build_src_decode_cell(),
    })

    # -- Cell 2: Write config.json --
    cfg_repr = repr(cfg)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            f"cfg = {cfg_repr}\n",
            "assert isinstance(cfg, dict), f'cfg must be dict, got {type(cfg)}'\n",
            "with open('config.json', 'w') as f:\n",
            "    json.dump(cfg, f, indent=2)\n",
            "print('Config written:', cfg)\n",
        ],
    })

    # -- Cell 3: Inject and run candidate_smoke_runner.py --
    with open(RUNNER_PATH, "r", encoding="utf-8") as f:
        runner_lines = [line + "\n" for line in f.read().splitlines()]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": runner_lines,
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(os.path.dirname(NOTEBOOK_OUT), exist_ok=True)
    with open(NOTEBOOK_OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook written: {NOTEBOOK_OUT}")
    print(f"Config: {cfg}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the DPOD candidate smoke-verification Kaggle notebook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", nargs="+",
                    default=["transformer_branch", "variant_5_1", "mdp_branch"])
    p.add_argument("--train-episodes", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-steps-per-ep", type=int, default=20)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "candidates": args.candidates,
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps_per_ep": args.max_steps_per_ep,
        "d_model": args.d_model,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
    }
    generate_notebook(cfg)
