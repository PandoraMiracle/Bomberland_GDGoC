"""
checkpoint_utils.py — Actor checkpoint loading with scalar-dim migration.

When N_SCALAR grows (e.g. 22 → 28 legal-action features), the first layer of
scalar_mlp gains input columns. Older checkpoints pad with zeros so existing
weights stay intact and new features start neutral.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from agent.mappo_agent.encoder import N_SCALAR


_SCALAR_WEIGHT_KEY = "scalar_mlp.0.weight"


def adapt_actor_scalar_weights(state_dict: dict[str, Any], n_scalar: int) -> dict[str, Any]:
    """Pad or trim scalar_mlp.0.weight to match ``n_scalar`` input features."""
    if _SCALAR_WEIGHT_KEY not in state_dict:
        return state_dict

    w = state_dict[_SCALAR_WEIGHT_KEY]
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return state_dict

    in_dim = int(w.shape[1])
    if in_dim == n_scalar:
        return state_dict

    out_dim = int(w.shape[0])
    new_w = torch.zeros(out_dim, n_scalar, dtype=w.dtype, device=w.device)
    copy_cols = min(in_dim, n_scalar)
    new_w[:, :copy_cols] = w[:, :copy_cols]
    adapted = dict(state_dict)
    adapted[_SCALAR_WEIGHT_KEY] = new_w
    return adapted


def extract_actor_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict", checkpoint))


def load_actor_state_dict(
    actor: nn.Module,
    checkpoint: dict[str, Any] | str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool | None = None,
) -> dict[str, Any]:
    """
    Load actor weights from a checkpoint file or dict.

    ``strict`` defaults to False when scalar input dim changed; True when shapes match.
    Returns the full checkpoint dict.
    """
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    else:
        ckpt = checkpoint

    state = extract_actor_state_dict(ckpt)
    n_scalar = int(actor.scalar_mlp[0].in_features)
    old_in = int(state[_SCALAR_WEIGHT_KEY].shape[1]) if _SCALAR_WEIGHT_KEY in state else n_scalar
    state = adapt_actor_scalar_weights(state, n_scalar)

    if strict is None:
        strict = old_in == n_scalar

    actor.load_state_dict(state, strict=strict)
    return ckpt
