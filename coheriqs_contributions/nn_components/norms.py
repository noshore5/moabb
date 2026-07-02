"""Normalization factories and small channel-wise norm wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index

import torch
import torch.nn as nn

from .configs import NormConfig


class AxisNorm(nn.Module):
    """Normalize activations over arbitrary axes with broadcastable affine params."""

    def __init__(
        self,
        num_dims: int,
        reduce_dims: tuple[int, ...],
        stat_dims: tuple[int, ...] | None = None,
        affine_dims: tuple[int, ...] | None = None,
        dim_sizes: dict[int, int] | None = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
        batch_dim: int = 0,
        mode: str = "standard",
    ) -> None:
        super().__init__()
        self.num_dims = _validate_positive_int("num_dims", num_dims)
        if mode != "standard":
            raise ValueError("AxisNorm currently supports mode='standard' only.")
        self.mode = mode
        self.reduce_dims = _normalize_dims(
            "reduce_dims", reduce_dims, self.num_dims, allow_empty=False
        )
        complement_dims = tuple(
            dim for dim in range(self.num_dims) if dim not in self.reduce_dims
        )
        self.stat_dims = (
            complement_dims
            if stat_dims is None
            else _normalize_dims(
                "stat_dims", stat_dims, self.num_dims, allow_empty=True
            )
        )
        default_affine_dims = self.stat_dims if affine else ()
        self.affine_dims = (
            default_affine_dims
            if affine_dims is None
            else _normalize_dims(
                "affine_dims", affine_dims, self.num_dims, allow_empty=True
            )
        )
        self.dim_sizes = _normalize_dim_sizes(dim_sizes, self.num_dims)
        self.eps = float(eps)
        _validate_eps(self.eps)
        self.momentum = _validate_momentum(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        self.batch_dim = _normalize_single_dim("batch_dim", batch_dim, self.num_dims)

        stat_overlap = sorted(set(self.stat_dims).intersection(self.reduce_dims))
        if stat_overlap:
            raise ValueError(
                "stat_dims must not include reduced dimensions; "
                f"got overlap {stat_overlap}."
            )
        if self.track_running_stats and self.batch_dim not in self.reduce_dims:
            raise ValueError(
                "track_running_stats=True requires batch_dim to be included "
                "in reduce_dims."
            )
        if self.track_running_stats and set(self.stat_dims) != set(complement_dims):
            raise ValueError(
                "track_running_stats=True requires stat_dims to match the "
                "non-reduced dimensions."
            )

        if self.affine:
            affine_shape = _make_broadcast_shape(
                "affine parameters", self.affine_dims, self.num_dims, self.dim_sizes
            )
            self.weight = nn.Parameter(torch.ones(affine_shape))
            self.bias = nn.Parameter(torch.zeros(affine_shape))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            stat_shape = _make_broadcast_shape(
                "running stats", self.stat_dims, self.num_dims, self.dim_sizes
            )
            self.register_buffer("running_mean", torch.zeros(stat_shape))
            self.register_buffer("running_var", torch.ones(stat_shape))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.num_dims:
            raise ValueError(
                f"Expected input rank {self.num_dims}, got shape {tuple(x.shape)}."
            )
        self._validate_input_sizes(x)

        if self.track_running_stats and not self.training:
            if self.running_mean is None or self.running_var is None:
                raise RuntimeError("AxisNorm running stats are not initialized.")
            mean = self.running_mean
            var = self.running_var
        else:
            var, mean = torch.var_mean(
                x,
                dim=self.reduce_dims,
                keepdim=True,
                correction=0,
            )
            if self.track_running_stats:
                if self.running_mean is None or self.running_var is None:
                    raise RuntimeError("AxisNorm running stats are not initialized.")
                with torch.no_grad():
                    self.running_mean.lerp_(mean.detach(), self.momentum)
                    self.running_var.lerp_(var.detach(), self.momentum)

        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def _validate_input_sizes(self, x: torch.Tensor) -> None:
        for dim, expected_size in self.dim_sizes.items():
            actual_size = int(x.shape[dim])
            if actual_size != expected_size:
                raise ValueError(
                    f"Expected dimension {dim} to have size {expected_size}, "
                    f"got {actual_size}."
                )


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


def _validate_momentum(momentum: float) -> float:
    momentum = float(momentum)
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be between 0 and 1.")
    return momentum


def _normalize_dims(
    name: str,
    dims: Sequence[int],
    num_dims: int,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    try:
        normalized = tuple(
            _normalize_single_dim(name, dim, num_dims) for dim in tuple(dims)
        )
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integer dimensions.") from exc
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique dimensions.")
    return normalized


def _normalize_single_dim(name: str, dim: int, num_dims: int) -> int:
    try:
        raw_dim = index(dim)
    except TypeError as exc:
        raise ValueError(f"{name} must contain integer dimensions.") from exc
    if raw_dim < -num_dims or raw_dim >= num_dims:
        raise ValueError(
            f"{name} dimension {raw_dim} is out of range for num_dims={num_dims}."
        )
    return raw_dim % num_dims


def _normalize_dim_sizes(
    dim_sizes: Mapping[int, int] | None,
    num_dims: int,
) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for raw_dim, raw_size in (dim_sizes or {}).items():
        dim = _normalize_single_dim("dim_sizes", raw_dim, num_dims)
        size = _validate_positive_int(f"dim_sizes[{raw_dim!r}]", raw_size)
        if dim in normalized and normalized[dim] != size:
            raise ValueError(f"dim_sizes has conflicting sizes for dimension {dim}.")
        normalized[dim] = size
    return normalized


def _make_broadcast_shape(
    name: str,
    dims: tuple[int, ...],
    num_dims: int,
    dim_sizes: Mapping[int, int],
) -> tuple[int, ...]:
    missing_dims = [dim for dim in dims if dim not in dim_sizes]
    if missing_dims:
        raise ValueError(f"{name} require dim_sizes for dimensions {missing_dims}.")
    shape = [1] * num_dims
    for dim in dims:
        shape[dim] = dim_sizes[dim]
    return tuple(shape)


def _validate_groups(groups: int, channels: int) -> None:
    groups = _validate_positive_int("groups", groups)
    if channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by groups ({groups})."
        )
