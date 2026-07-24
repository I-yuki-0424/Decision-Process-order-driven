import json
import os

notebook = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Notebook Version & Runtime Metadata\n",
        "NOTEBOOK_VERSION = \"v1.0\"\n",
        "PROTOCOL_VERSION = \"v6\"\n",
        "GIT_COMMIT_SHA = \"7c598c6\"\n",
        "DATASET_VERSION = \"gymnax-v0.0.9\"\n",
        "\n",
        "print(f\"=== KAGGLE RUNTIME METADATA ===\")\n",
        "print(f\"Notebook Version: {NOTEBOOK_VERSION}\")\n",
        "print(f\"Protocol Version: {PROTOCOL_VERSION}\")\n",
        "print(f\"Git Commit SHA  : {GIT_COMMIT_SHA}\")\n",
        "print(f\"Dataset Version : {DATASET_VERSION}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install -q jax jaxlib flax optax gymnax matplotlib numpy pandas\n",
        "import sys\n",
        "import os\n",
        "print(\"Python Executable:\", sys.executable)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Run Gymnax Decision Process Benchmark Suite\n",
        "!python -m src.pipeline.benchmark\n"
      ]
    }
  ],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

os.makedirs("kaggle_kernel", exist_ok=True)
with open("kaggle_kernel/decision_process_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Kaggle notebook generated at kaggle_kernel/decision_process_benchmark.ipynb")
