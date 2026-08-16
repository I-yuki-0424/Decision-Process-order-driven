import base64
import json
import os

def read_file_bytes(filepath: str) -> str:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode('ascii')
    return ""

files_to_embed = [
    ("src/__init__.py", read_file_bytes("src/__init__.py")),
    ("src/model/__init__.py", read_file_bytes("src/model/__init__.py")),
    ("src/environment/__init__.py", read_file_bytes("src/environment/__init__.py")),
    ("src/pipeline/__init__.py", read_file_bytes("src/pipeline/__init__.py")),
    ("src/model/types.py", read_file_bytes("src/model/types.py")),
    ("src/model/channel_encoder.py", read_file_bytes("src/model/channel_encoder.py")),
    ("src/model/baseline_model.py", read_file_bytes("src/model/baseline_model.py")),
    ("src/model/transformer_decision_core.py", read_file_bytes("src/model/transformer_decision_core.py")),
    ("src/model/hierarchical_transformer.py", read_file_bytes("src/model/hierarchical_transformer.py")),
    ("src/model/beam_search.py", read_file_bytes("src/model/beam_search.py")),
    ("src/model/checkpoint.py", read_file_bytes("src/model/checkpoint.py")),
    ("src/environment/gymnax_decision_env.py", read_file_bytes("src/environment/gymnax_decision_env.py")),
    ("src/environment/craftax_env_adapter.py", read_file_bytes("src/environment/craftax_env_adapter.py")),
    ("src/pipeline/hierarchical_pipeline.py", read_file_bytes("src/pipeline/hierarchical_pipeline.py")),
    ("kaggle_kernel/hierarchical_kaggle_runner.py", read_file_bytes("kaggle_kernel/hierarchical_kaggle_runner.py")),
]

cells = []

# Cell 1: Environment Setup & GPU check
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Kaggle Remote GPU Setup & Package Installation\n",
        "!pip install -q craftax craftax-classic jax jaxlib flax optax gymnax matplotlib numpy pandas\n",
        "import jax\n",
        "print('=== REMOTE KAGGLE EXECUTION ENVIRONMENT ===')\n",
        "print('JAX Backend:', jax.default_backend())\n",
        "print('JAX Devices:', jax.devices())\n"
    ]
})

# Cell 2: Write out module codebase using Base64 decode
setup_code_lines = ["import base64, os\n"]
for rel_path, b64_str in files_to_embed:
    dir_name = os.path.dirname(rel_path)
    if dir_name:
        setup_code_lines.append(f'os.makedirs("{dir_name}", exist_ok=True)\n')
    setup_code_lines.append(f'with open("{rel_path}", "wb") as f:\n    f.write(base64.b64decode("{b64_str}"))\n')

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code_lines
})

# Cell 3: Execute remote re-training and benchmarking suite
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run Re-Training & Comprehensive Benchmarking Suite across all Model Configurations\n",
        "import sys\n",
        "sys.path.insert(0, '.')\n",
        "from kaggle_kernel.hierarchical_kaggle_runner import run_kaggle_verification\n",
        "run_kaggle_verification()\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs("kaggle_kernel", exist_ok=True)
with open("kaggle_kernel/decision_process_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated self-contained Kaggle notebook at kaggle_kernel/decision_process_benchmark.ipynb with valid kernelspec.")
