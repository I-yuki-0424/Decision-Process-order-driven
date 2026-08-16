"""
Kaggle Notebook Builder for Phase II Grid Search.

Generates a self-contained .ipynb that:
1. Installs packages (craftax, jax, optax)
2. Decodes the full src/ tree from base64 (so Kaggle gets the latest fixed code)
3. Writes config.json with the specified hyperparameters
4. Executes phase2_runner.py

Usage:
    python build_kaggle_kernel.py --run-id phase2_001 --n-layers 6 --beam-width 5 \\
        --z-step 64 --causal --train-steps 1000000 --eval-steps 500
"""

import argparse
import base64
import json
import os
import sys


KERNEL_ID    = "bfloat16/craftax-classic-1000-episode-rl-benchmark"
NOTEBOOK_OUT = "kaggle_kernel/decision_process_benchmark.ipynb"


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
        # Create directories first
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
    """Generate the Kaggle notebook ipynb file with the given Phase II config."""
    cells = []

    # ── Cell 0: Package install + hardware check ─────────────────────────────
    run_id_val = cfg["run_id"]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Phase II: Package Installation & GPU Verification\n",
            "# Kaggle GPU kernels have JAX (GPU build) pre-installed — do NOT reinstall jax/jaxlib.\n",
            "# craftax versions start at 1.0.0; pin to latest stable.\n",
            "import subprocess, sys\n",
            "result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax', 'optax'], capture_output=True, text=True)\n",
            "print(result.stdout[-2000:] if result.stdout else '')\n",
            "if result.returncode != 0:\n",
            "    print('PIP STDERR:', result.stderr[-2000:])\n",
            "    raise RuntimeError('pip install failed — see stderr above')\n",
            "# Verify critical imports before doing anything else\n",
            "import jax, jaxlib, optax\n",
            "import craftax\n",
            "# craftax does not expose __version__; verify via submodule import\n",
            "from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv\n",
            "print('craftax import OK:', CraftaxClassicSymbolicEnv)\n",
            "print('JAX Backend:', jax.default_backend())\n",
            "print('JAX Devices:', jax.devices())\n",
            "print('JAX version:', jax.__version__)\n",
            f"print('Run ID: {run_id_val}')\n",
        ],
    })

    # ── Cell 1: Decode src/ tree ──────────────────────────────────────────────
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": build_src_decode_cell(),
    })

    # ── Cell 2: Write config.json ─────────────────────────────────────────────
    # Use repr(cfg) so the Kaggle cell receives a Python dict literal,
    # NOT a JSON string wrapped in quotes (which caused 'str'.get() AttributeError).
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

    # ── Cell 3: Inject and run phase2_runner.py ───────────────────────────────
    runner_path = "kaggle_kernel/phase2_runner.py"
    with open(runner_path, "r", encoding="utf-8") as f:
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
        description="Build Kaggle Phase II benchmark notebook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",       default="phase2_run",  help="Unique run identifier")
    p.add_argument("--n-layers",     type=int, default=6,   help="Transformer layer count")
    p.add_argument("--beam-width",   type=int, default=5,   help="Beam search width k")
    p.add_argument("--z-step",       type=int, default=64,  help="Z-compression interval")
    p.add_argument("--causal",       action="store_true",   help="Use causal attention mask")
    p.add_argument("--no-causal",    dest="causal", action="store_false")
    p.add_argument("--train-steps",  type=int, default=1_000_000, help="Training steps")
    p.add_argument("--eval-steps",   type=int, default=500,  help="Eval episode steps")
    p.add_argument("--seed",         type=int, default=42,   help="JAX PRNG seed")
    p.add_argument("--d-model",      type=int, default=512,  help="Transformer d_model")
    p.add_argument("--num-heads",    type=int, default=8,    help="Number of attention heads")
    p.add_argument("--lr",           type=float, default=3e-4, help="AdamW learning rate")
    p.add_argument("--noise-prob",   type=float, default=0.15, help="Noise injection rate")
    p.set_defaults(causal=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "run_id":      args.run_id,
        "n_layers":    args.n_layers,
        "beam_width":  args.beam_width,
        "z_step":      args.z_step,
        "is_causal":   args.causal,
        "train_steps": args.train_steps,
        "eval_steps":  args.eval_steps,
        "seed":        args.seed,
        "d_model":     args.d_model,
        "num_heads":   args.num_heads,
        "lr":          args.lr,
        "noise_prob":  args.noise_prob,
    }
    generate_notebook(cfg)
