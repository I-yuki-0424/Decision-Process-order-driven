"""
JAX / Flax PyTree Checkpoint Serialization & Deserialization Module.

Saves and loads exact HierarchicalModelParameters PyTree structure matching
src/model/hierarchical_transformer.py without structural flattening or shape mismatch.

Also provides a generic, type-agnostic PyTree checkpointer
(save_pytree_checkpoint/load_pytree_checkpoint) plus AsyncCheckpointManager,
which periodically saves an arbitrary training PyTree (any of the candidate
model params in src/model/candidates/, or the existing Hierarchical/flat
model params) on a background thread so training isn't blocked, under a
resumable output/checkpoints/<run_name>/ path scheme. The original two
functions above are kept unchanged (no existing caller depends on them, per
analyze_change_impact -- see docs/core/STATE.yaml) so nothing else needs to
be touched by this change.
"""

import glob
import json
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np

from src.model.hierarchical_transformer import HierarchicalModelParameters, init_hierarchical_model_parameters


def save_model_checkpoint(params: HierarchicalModelParameters, checkpoint_path: str) -> str:
    """Save HierarchicalModelParameters PyTree to a serialized pickle checkpoint file.

    Args:
        params: HierarchicalModelParameters PyTree instance.
        checkpoint_path: Target filepath (e.g. 'output/checkpoints/model_8L.pkl').

    Returns:
        Absolute filepath of saved checkpoint.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Convert JAX arrays to numpy float32 arrays for portable serialization
    def _to_numpy(leaf):
        if isinstance(leaf, (jnp.ndarray, jax.Array)):
            return np.array(leaf)
        return leaf

    import numpy as np
    numpy_pytree = jax.tree_util.tree_map(_to_numpy, params)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(numpy_pytree, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Checkpoint] Saved PyTree model checkpoint to: {checkpoint_path}")
    return os.path.abspath(checkpoint_path)


def load_model_checkpoint(checkpoint_path: str) -> HierarchicalModelParameters:
    """Load HierarchicalModelParameters PyTree from a serialized pickle checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        HierarchicalModelParameters PyTree instance with jnp.ndarray leaves.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        loaded_pytree = pickle.load(f)

    # Convert numpy arrays back to jnp.ndarray
    def _to_jax(leaf):
        if hasattr(leaf, '__array__'):
            return jnp.array(leaf)
        return leaf

    params = jax.tree_util.tree_map(_to_jax, loaded_pytree)
    print(f"[Checkpoint] Loaded PyTree model checkpoint from: {checkpoint_path}")
    return params


def inspect_pytree_parameters(params: HierarchicalModelParameters) -> Dict[str, Any]:
    """Inspect PyTree parameter structure, leaf count, and shape breakdown."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    total_params = sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    total_bytes = sum(leaf.nbytes for leaf in leaves if hasattr(leaf, 'nbytes'))
    
    layer_counts = []
    if hasattr(params, 'layers'):
        for i, l_params in enumerate(params.layers):
            l_leaves = jax.tree_util.tree_leaves(l_params)
            l_count = sum(p.size for p in l_leaves if hasattr(p, 'size'))
            layer_counts.append({
                "layer_index": i,
                "layer_name": f"Transformer_Layer_{i}",
                "parameter_count": l_count,
                "memory_mb": round((l_count * 4) / (1024 * 1024), 2),
                "num_pytree_tensors": len(l_leaves),
            })

    assert len(layer_counts) > 0, "No layers found during inspection of parameters."

    return {
        "num_pytree_leaves": len(leaves),
        "total_parameters": total_params,
        "memory_footprint_mb": round(total_bytes / (1024 * 1024), 2),
        "transformer_layers": layer_counts,
    }


# ---------------------------------------------------------------------------
# Generic PyTree checkpointing (any params, not just HierarchicalModelParameters).
# ---------------------------------------------------------------------------


def save_pytree_checkpoint(params: Any, checkpoint_path: str, step: int = 0, config: Optional[Dict[str, Any]] = None) -> str:
    """Save an arbitrary JAX PyTree to a pickle file, with step/config metadata
    for resuming. Blocks until `params`'s JAX arrays are ready before converting
    to numpy (call this from AsyncCheckpointManager to avoid blocking training)."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _to_numpy(leaf):
        if isinstance(leaf, (jnp.ndarray, jax.Array)):
            return np.array(leaf)
        return leaf

    numpy_pytree = jax.tree_util.tree_map(_to_numpy, params)
    payload = {"step": step, "config": config or {}, "params": numpy_pytree}

    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, checkpoint_path)  # atomic on POSIX: a crash mid-write never corrupts checkpoint_path

    return os.path.abspath(checkpoint_path)


def load_pytree_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint saved by save_pytree_checkpoint. Returns
    {"step": int, "config": dict, "params": <PyTree with jnp.ndarray leaves>}."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)

    def _to_jax(leaf):
        if hasattr(leaf, "__array__"):
            return jnp.array(leaf)
        return leaf

    payload["params"] = jax.tree_util.tree_map(_to_jax, payload["params"])
    return payload


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Find the highest-step checkpoint file under run_dir (named
    step_<N>.pkl by AsyncCheckpointManager), for resuming a crashed/timed-out
    run. Returns None if run_dir has no checkpoints yet."""
    candidates = glob.glob(os.path.join(run_dir, "step_*.pkl"))
    if not candidates:
        return None

    def _step_of(path: str) -> int:
        match = re.search(r"step_(\d+)\.pkl$", path)
        return int(match.group(1)) if match else -1

    return max(candidates, key=_step_of)


class AsyncCheckpointManager:
    """Periodically saves a training PyTree on a background thread so the
    training loop is never blocked by pickle/disk I/O.

    Usage (see src/pipeline/candidate_benchmark.py for the wired-in example):
        mgr = AsyncCheckpointManager("output/checkpoints/my_run", save_every=200)
        for step in range(num_steps):
            ...
            mgr.maybe_save(params, step, config={"model": "transformer_branch"})
        mgr.wait_for_pending()  # flush before the process exits

    Resumability: mgr.resume() returns (step, config, params) from the latest
    step_<N>.pkl under run_dir, or None if there is nothing to resume from.
    """

    def __init__(self, run_dir: str, save_every: int = 200, max_workers: int = 1):
        self.run_dir = run_dir
        self.save_every = max(1, int(save_every))
        os.makedirs(run_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending = []
        self._lock = threading.Lock()

    def maybe_save(self, params: Any, step: int, config: Optional[Dict[str, Any]] = None, force: bool = False) -> bool:
        """Saves every `save_every` steps (or when force=True). block_until_ready
        happens here, synchronously, on the caller's thread (cheap relative to
        pickling+disk I/O); the actual numpy conversion + pickle write is handed
        off to a background thread so the training loop keeps going."""
        if not force and step % self.save_every != 0:
            return False

        ready_params = jax.block_until_ready(params)
        checkpoint_path = os.path.join(self.run_dir, f"step_{step}.pkl")
        future = self._executor.submit(save_pytree_checkpoint, ready_params, checkpoint_path, step, config)
        with self._lock:
            self._pending.append(future)
        return True

    def wait_for_pending(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def resume(self):
        """Returns (step, config, params) from the latest checkpoint under
        run_dir, or None if there is nothing to resume from."""
        latest = find_latest_checkpoint(self.run_dir)
        if latest is None:
            return None
        payload = load_pytree_checkpoint(latest)
        return payload["step"], payload["config"], payload["params"]

    def close(self) -> None:
        self.wait_for_pending()
        self._executor.shutdown(wait=True)
