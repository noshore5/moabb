"""Regularization helpers for primitive blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import RegConfig


def _validate_probability(name: str, value: float) -> None:
    if float(value) < 0.0 or float(value) >= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0).")


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        _validate_probability("drop_prob", drop_prob)
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


def build_dense_regularization(cfg: RegConfig) -> nn.Module:
    """Build dense regularization layers."""

    _validate_probability("dropout", cfg.dropout)
    _validate_probability("spatial_dropout", cfg.spatial_dropout)
    if cfg.spatial_dropout > 0.0:
        raise ValueError("spatial_dropout is only supported for convolution blocks.")
    return nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()


def build_conv_regularization(cfg: RegConfig, *, ndim: int) -> nn.Module:
    """Build convolution regularization layers for 1D or 2D conv blocks."""

    _validate_probability("dropout", cfg.dropout)
    _validate_probability("spatial_dropout", cfg.spatial_dropout)
    layers: list[nn.Module] = []
    if cfg.dropout > 0.0:
        layers.append(nn.Dropout(cfg.dropout))
    if cfg.spatial_dropout > 0.0:
        if ndim == 1:
            layers.append(nn.Dropout1d(cfg.spatial_dropout))
        elif ndim == 2:
            layers.append(nn.Dropout2d(cfg.spatial_dropout))
        else:
            raise ValueError("ndim must be 1 or 2.")
    if not layers:
        return nn.Identity()
    return nn.Sequential(*layers)
