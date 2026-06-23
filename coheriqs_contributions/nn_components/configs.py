"""Configuration dataclasses for generic neural-network primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ActivationKind = Literal["identity", "relu", "gelu", "silu", "elu", "tanh"]
InitMode = Literal[
    "torch_default",
    "auto",
    "kaiming",
    "xavier",
    "lecun",
    "normal",
    "trunc_normal",
    "orthogonal",
    "zeros",
]
BiasInitMode = Literal["zeros", "normal"]
NormKind = Literal["batch", "group", "layer", "rms"]
NormPosition = Literal["none", "pre", "post", "sandwich"]
ShortcutMode = Literal["identity", "project", "auto"]
Padding1d = Literal["same", "valid"] | int
Padding2d = Literal["same", "valid"] | int | tuple[int, int]


@dataclass(frozen=True)
class SeedConfig:
    """Reproducibility settings applied before module construction."""

    seed: int = 42
    deterministic: bool = False
    benchmark: bool = False


@dataclass(frozen=True)
class InitConfig:
    """Initialization policy for Linear and Conv modules."""

    mode: InitMode = "auto"
    gain: float | None = None
    std: float | None = None
    bias: BiasInitMode = "zeros"
    zero_last: bool = False


@dataclass(frozen=True)
class NormConfig:
    """Normalization factory settings."""

    kind: NormKind | None = None
    groups: int = 8
    eps: float = 1e-5
    affine: bool = True


@dataclass(frozen=True)
class ActConfig:
    """Activation factory settings."""

    kind: ActivationKind = "gelu"
    inplace: bool = False


@dataclass(frozen=True)
class RegConfig:
    """Regularization settings used inside primitive blocks."""

    dropout: float = 0.0
    spatial_dropout: float = 0.0


@dataclass(frozen=True)
class ResidualConfig:
    """Residual wrapping settings."""

    norm_position: NormPosition = "none"
    shortcut: ShortcutMode = "auto"
    scale: float = 1.0
    layer_scale: float | None = None
    rezero: bool = False
    drop_path: float = 0.0


@dataclass(frozen=True)
class DenseMLPConfig:
    """Configuration for a dense MLP acting on the final tensor dimension."""

    in_features: int
    hidden_features: int
    out_features: int | None = None
    depth: int = 2
    activation: ActConfig = field(default_factory=ActConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    regularization: RegConfig = field(default_factory=RegConfig)
    init: InitConfig = field(default_factory=InitConfig)


@dataclass(frozen=True)
class Conv1dConfig:
    """Configuration for a Conv1d primitive over [B, C, T] tensors."""

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    padding: Padding1d = "same"
    groups: int = 1
    separable: bool = False
    bias: bool = True
    norm: NormConfig = field(default_factory=NormConfig)
    activation: ActConfig = field(default_factory=ActConfig)
    regularization: RegConfig = field(default_factory=RegConfig)
    init: InitConfig = field(default_factory=InitConfig)


@dataclass(frozen=True)
class Conv2dConfig:
    """Configuration for a Conv2d primitive over [B, C, H, W] tensors."""

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int] = 3
    stride: int | tuple[int, int] = 1
    dilation: int | tuple[int, int] = 1
    padding: Padding2d = "same"
    groups: int = 1
    separable: bool = False
    bias: bool = True
    norm: NormConfig = field(default_factory=NormConfig)
    activation: ActConfig = field(default_factory=ActConfig)
    regularization: RegConfig = field(default_factory=RegConfig)
    init: InitConfig = field(default_factory=InitConfig)
