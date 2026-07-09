"""Configuration dataclasses for generic neural-network primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ActivationKind = Literal["identity", "relu", "gelu", "silu", "elu", "tanh"] | None
GateMode = Literal["soft", "gumbel_soft", "gumbel_hard", "argmax", "frozen"]
GateScope = Literal["layer", "channel"]
GateGradientMode = Literal["selected_only", "soft_all"]
GateEvalMode = Literal["same", "argmax", "frozen"]
AlphaUpdateSplit = Literal["train", "val"]
AlphaOptim = Literal["shared", "separate"]
LogitsInit = float | tuple[float, ...] | tuple[tuple[float, ...], ...]
BranchOutputNormKind = Literal["none", "rms", "layer"]
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

    kind: ActivationKind = None
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
class CategoricalGateConfig:
    """Learned categorical selector settings.

    ``alpha_update_split`` and ``alpha_optim`` are declared training intent.
    The trainer is responsible for optimizer grouping and data split handling.
    """

    num_choices: int
    mode: GateMode = "soft"
    temperature: float = 1.0
    scope: GateScope = "layer"
    num_features: int | None = None
    channel_dim: int = 1
    logits_init: LogitsInit = 0.0
    eval_mode: GateEvalMode = "argmax"
    exploration_epsilon: float | None = None
    cost_weight: float = 0.0
    entropy_weight: float = 0.0
    gradient_mode: GateGradientMode = "soft_all"
    frozen_index: int | None = None
    alpha_update_split: AlphaUpdateSplit = "train"
    alpha_optim: AlphaOptim = "separate"


@dataclass(frozen=True)
class SelectPathConfig:
    """Optional path-selection conveniences layered on top of a gate config."""

    include_zero_update: bool = False
    candidate_costs: tuple[float, ...] | None = None
    branch_norm: BranchOutputNormKind = "none"
    branch_norm_eps: float = 1e-5


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
