"""Activation factory."""

from __future__ import annotations

import torch.nn as nn

from .configs import ActConfig


def build_activation(cfg: ActConfig) -> nn.Module:
    """Build an activation module from a small typed config."""

    if cfg.kind == "identity" or cfg.kind is None:
        return nn.Identity()
    if cfg.kind == "relu":
        return nn.ReLU(inplace=cfg.inplace)
    if cfg.kind == "gelu":
        return nn.GELU()
    if cfg.kind == "silu":
        return nn.SiLU(inplace=cfg.inplace)
    if cfg.kind == "elu":
        return nn.ELU(inplace=cfg.inplace)
    if cfg.kind == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation kind: {cfg.kind!r}")
