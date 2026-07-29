"""
JAX / Flax PyTree Checkpoint Serialization & Deserialization Module.

Saves and loads exact HierarchicalModelParameters PyTree structure matching
src/model/hierarchical_transformer.py without structural flattening or shape mismatch.
"""

import os
import pickle
from typing import Any, Dict
import jax
import jax.numpy as jnp

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
    if hasattr(params, 'transformer_layers'):
        for i, l_params in enumerate(params.transformer_layers):
            l_leaves = jax.tree_util.tree_leaves(l_params)
            l_count = sum(p.size for p in l_leaves if hasattr(p, 'size'))
            layer_counts.append({
                "layer_index": i,
                "layer_name": f"Transformer_Layer_{i}",
                "parameter_count": l_count,
                "memory_mb": round((l_count * 4) / (1024 * 1024), 2),
                "num_pytree_tensors": len(l_leaves),
            })

    return {
        "num_pytree_leaves": len(leaves),
        "total_parameters": total_params,
        "memory_footprint_mb": round(total_bytes / (1024 * 1024), 2),
        "transformer_layers": layer_counts,
    }
