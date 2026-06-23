"""Normalization factories and small channel-wise norm wrappers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import NormConfig


class RMSNorm(nn.Module):
    """Root mean square normalization over the final dimension."""

    def __init__(self, features: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive.")
        self.features = int(features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(features)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.features:
            raise ValueError(
                f"Expected final dimension {self.features}, got {x.shape[-1]}."
            )
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x * scale
        if self.weight is not None:
            y = y * self.weight
        return y


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channels for [B, C, T] tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T] tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for [B, C, H, W] tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")
        y = self.norm(x.permute(0, 2, 3, 1))
        return y.permute(0, 3, 1, 2)


class ChannelRMSNorm1d(nn.Module):
    """RMSNorm over channels for [B, C, T] tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.norm = RMSNorm(channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, T] tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ChannelRMSNorm2d(nn.Module):
    """RMSNorm over channels for [B, C, H, W] tensors."""

    def __init__(self, channels: int, *, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.norm = RMSNorm(channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}.")
        y = self.norm(x.permute(0, 2, 3, 1))
        return y.permute(0, 3, 1, 2)


def build_norm_dense(cfg: NormConfig, features: int) -> nn.Module | None:
    """Build a norm for dense tensors with shape [..., D]."""

    features = _validate_positive_int("features", features)
    _validate_eps(cfg.eps)
    if cfg.kind is None:
        return None
    if cfg.kind == "layer":
        return nn.LayerNorm(features, eps=cfg.eps, elementwise_affine=cfg.affine)
    if cfg.kind == "rms":
        return RMSNorm(features, eps=cfg.eps, affine=cfg.affine)
    raise ValueError(f"Norm kind {cfg.kind!r} is not supported for dense tensors.")


def build_norm_1d(cfg: NormConfig, channels: int) -> nn.Module | None:
    """Build a norm for [B, C, T] tensors."""

    channels = _validate_positive_int("channels", channels)
    _validate_eps(cfg.eps)
    if cfg.kind is None:
        return None
    if cfg.kind == "batch":
        return nn.BatchNorm1d(channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "group":
        _validate_groups(cfg.groups, channels)
        return nn.GroupNorm(cfg.groups, channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "layer":
        return ChannelLayerNorm1d(channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "rms":
        return ChannelRMSNorm1d(channels, eps=cfg.eps, affine=cfg.affine)
    raise ValueError(f"Unsupported norm kind: {cfg.kind!r}")


def build_norm_2d(cfg: NormConfig, channels: int) -> nn.Module | None:
    """Build a norm for [B, C, H, W] tensors."""

    channels = _validate_positive_int("channels", channels)
    _validate_eps(cfg.eps)
    if cfg.kind is None:
        return None
    if cfg.kind == "batch":
        return nn.BatchNorm2d(channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "group":
        _validate_groups(cfg.groups, channels)
        return nn.GroupNorm(cfg.groups, channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "layer":
        return ChannelLayerNorm2d(channels, eps=cfg.eps, affine=cfg.affine)
    if cfg.kind == "rms":
        return ChannelRMSNorm2d(channels, eps=cfg.eps, affine=cfg.affine)
    raise ValueError(f"Unsupported norm kind: {cfg.kind!r}")


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_eps(eps: float) -> None:
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive.")


def _validate_groups(groups: int, channels: int) -> None:
    groups = _validate_positive_int("groups", groups)
    if channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by groups ({groups})."
        )
