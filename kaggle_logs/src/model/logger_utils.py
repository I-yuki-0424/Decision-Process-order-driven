"""
Logging, Error Display, and Validation Utilities for JAX Decision Model.

Provides structured logging, error handling, static validation helpers,
and assertion utilities.
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp


def get_logger(name: str = "DecisionModel", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a structured logger with custom formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def format_error(module_name: str, error_type: str, details: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Format a detailed error message for debugging and logging."""
    msg = f"=== ERROR IN [{module_name}] ===\n"
    msg += f"Type: {error_type}\n"
    msg += f"Details: {details}\n"
    if context:
        msg += "Context:\n"
        for k, v in context.items():
            msg += f"  - {k}: {v}\n"
    msg += "================================="
    return msg


def validate_input_context(input_n: Any) -> Tuple[bool, str]:
    """Perform static shape and sanity validation on InputContextN.

    Returns:
        (is_valid, error_reason)
    """
    try:
        actions = input_n.actions
        state = input_n.state
        history = input_n.history
        target = input_n.target

        if actions.features.ndim < 2:
            return False, f"actions.features must be at least 2D, got shape {actions.features.shape}"
        if actions.costs.ndim < 2:
            return False, f"actions.costs must be at least 2D, got shape {actions.costs.shape}"
        if state.progress_rate.ndim != 0:
            return False, f"state.progress_rate must be scalar (ndim 0), got shape {state.progress_rate.shape}"
        if history.action_indices.ndim < 1:
            return False, f"history.action_indices must be at least 1D, got shape {history.action_indices.shape}"
        if target.target_state.ndim < 1:
            return False, f"target.target_state must be at least 1D, got shape {target.target_state.shape}"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Exception during validation: {str(e)}"
