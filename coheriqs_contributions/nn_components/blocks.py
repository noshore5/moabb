"""Primitive dense and convolution block builders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import build_activation
from .configs import (
    Conv1dConfig,
    Conv2dConfig,
    DenseMLPConfig,
    NormConfig,
    ResidualConfig,
)
from .init import initialize_module, scoped_torch_init_seed
from .norms import build_norm_1d, build_norm_2d, build_norm_dense
from .regularization import build_conv_regularization, build_dense_regularization
from .residual import ResidualWrapper


class _ExpectNDims(nn.Module):
    def __init__(self, ndims: int, layout: str) -> None:
        super().__init__()
        self.ndims = ndims
        self.layout = layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.ndims:
            raise ValueError(
                f"Expected {self.layout} tensor, got shape {tuple(x.shape)}."
            )
        return x


class _ExpectLastDim(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = int(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.features:
            raise ValueError(
                f"Expected final dimension {self.features}, got {x.shape[-1]}."
            )
        return x


class SamePadConv1d(nn.Module):
    """Conv1d with TensorFlow-style dynamic same padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int,
        groups: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = int(x.shape[-1])
        out_length = math.ceil(length / self.stride)
        pad_needed = max(
            (out_length - 1) * self.stride
            + (self.kernel_size - 1) * self.dilation
            + 1
            - length,
            0,
        )
        left = pad_needed // 2
        right = pad_needed - left
        if pad_needed > 0:
            x = F.pad(x, (left, right))
        return self.conv(x)


class SamePadConv2d(nn.Module):
    """Conv2d with TensorFlow-style dynamic same padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = tuple(int(v) for v in kernel_size)
        self.stride = tuple(int(v) for v in stride)
        self.dilation = tuple(int(v) for v in dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = (int(x.shape[-2]), int(x.shape[-1]))
        pad_h = _same_pad_amount(
            height,
            self.kernel_size[0],
            self.stride[0],
            self.dilation[0],
        )
        pad_w = _same_pad_amount(
            width,
            self.kernel_size[1],
            self.stride[1],
            self.dilation[1],
        )
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            x = F.pad(x, (left, right, top, bottom))
        return self.conv(x)


def build_dense_mlp(
    cfg: DenseMLPConfig,
    residual: ResidualConfig | None = None,
    *,
    init_seed: int | None = None,
) -> nn.Module:
    """Build a dense MLP under an optional component-local initialization seed."""

    with scoped_torch_init_seed(init_seed):
        return _build_dense_mlp(cfg, residual)


def _build_dense_mlp(
    cfg: DenseMLPConfig,
    residual: ResidualConfig | None,
) -> nn.Module:
    """Build a dense MLP; wrap with ResidualWrapper when requested."""

    out_features = cfg.out_features if cfg.out_features is not None else cfg.in_features
    _validate_dense_cfg(cfg, out_features)

    layers: list[nn.Module] = [_ExpectLastDim(cfg.in_features)]
    if cfg.depth == 1:
        layers.append(nn.Linear(cfg.in_features, out_features))
    else:
        layers.append(nn.Linear(cfg.in_features, cfg.hidden_features))
        _append_dense_middle(layers, cfg, cfg.hidden_features)
        for _ in range(cfg.depth - 2):
            layers.append(nn.Linear(cfg.hidden_features, cfg.hidden_features))
            _append_dense_middle(layers, cfg, cfg.hidden_features)
        layers.append(nn.Linear(cfg.hidden_features, out_features))

    branch = nn.Sequential(*layers)
    initialize_module(
        branch,
        cfg.init,
        activation=cfg.activation.kind,
        primitive="dense",
    )

    if residual is None:
        return branch

    shortcut = _make_dense_shortcut(cfg.in_features, out_features, residual)
    _initialize_shortcut(
        shortcut,
        cfg.init,
        activation=cfg.activation.kind,
        primitive="dense",
    )
    return ResidualWrapper(
        branch,
        residual,
        shortcut=shortcut,
        norm_in=_make_dense_residual_norm(
            cfg,
            cfg.in_features,
            residual,
            input_side=True,
        ),
        norm_out=_make_dense_residual_norm(cfg, out_features, residual, input_side=False),
        scale_shape=(out_features,),
    )


def build_conv1d_block(
    cfg: Conv1dConfig,
    residual: ResidualConfig | None = None,
    *,
    init_seed: int | None = None,
) -> nn.Module:
    """Build a Conv1d block under an optional component-local init seed."""

    with scoped_torch_init_seed(init_seed):
        return _build_conv1d_block(cfg, residual)


def _build_conv1d_block(
    cfg: Conv1dConfig,
    residual: ResidualConfig | None,
) -> nn.Module:
    """Build a Conv1d block; wrap with ResidualWrapper when requested."""

    _validate_conv1d_cfg(cfg)
    layers: list[nn.Module] = [_ExpectNDims(3, "[B, C, T]")]
    if cfg.separable:
        layers.append(
            _make_conv1d(
                cfg.in_channels,
                cfg.in_channels,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                dilation=cfg.dilation,
                padding=cfg.padding,
                groups=cfg.in_channels,
                bias=cfg.bias,
            )
        )
        layers.append(
            _make_conv1d(
                cfg.in_channels,
                cfg.out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding="valid",
                groups=1,
                bias=cfg.bias,
            )
        )
    else:
        layers.append(
            _make_conv1d(
                cfg.in_channels,
                cfg.out_channels,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                dilation=cfg.dilation,
                padding=cfg.padding,
                groups=cfg.groups,
                bias=cfg.bias,
            )
        )

    _append_if_not_none(layers, build_norm_1d(cfg.norm, cfg.out_channels))
    layers.append(build_activation(cfg.activation))
    layers.append(build_conv_regularization(cfg.regularization, ndim=1))
    branch = nn.Sequential(*layers)
    initialize_module(
        branch,
        cfg.init,
        activation=cfg.activation.kind,
        primitive=_conv1d_primitive(cfg),
    )

    if residual is None:
        return branch

    shortcut = _make_conv1d_shortcut(cfg, residual)
    _initialize_shortcut(
        shortcut,
        cfg.init,
        activation=cfg.activation.kind,
        primitive="pointwise_conv1d",
    )
    return ResidualWrapper(
        branch,
        residual,
        shortcut=shortcut,
        norm_in=_make_conv1d_residual_norm(
            cfg,
            cfg.in_channels,
            residual,
            input_side=True,
        ),
        norm_out=_make_conv1d_residual_norm(
            cfg,
            cfg.out_channels,
            residual,
            input_side=False,
        ),
        scale_shape=(cfg.out_channels, 1),
    )


def build_conv2d_block(
    cfg: Conv2dConfig,
    residual: ResidualConfig | None = None,
    *,
    init_seed: int | None = None,
) -> nn.Module:
    """Build a Conv2d block under an optional component-local init seed."""

    with scoped_torch_init_seed(init_seed):
        return _build_conv2d_block(cfg, residual)


def _build_conv2d_block(
    cfg: Conv2dConfig,
    residual: ResidualConfig | None,
) -> nn.Module:
    """Build a Conv2d block; wrap with ResidualWrapper when requested."""

    _validate_conv2d_cfg(cfg)
    kernel_size = _as_pair(cfg.kernel_size)
    stride = _as_pair(cfg.stride)
    dilation = _as_pair(cfg.dilation)

    layers: list[nn.Module] = [_ExpectNDims(4, "[B, C, H, W]")]
    if cfg.separable:
        layers.append(
            _make_conv2d(
                cfg.in_channels,
                cfg.in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=cfg.padding,
                groups=cfg.in_channels,
                bias=cfg.bias,
            )
        )
        layers.append(
            _make_conv2d(
                cfg.in_channels,
                cfg.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding="valid",
                groups=1,
                bias=cfg.bias,
            )
        )
    else:
        layers.append(
            _make_conv2d(
                cfg.in_channels,
                cfg.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=cfg.padding,
                groups=cfg.groups,
                bias=cfg.bias,
            )
        )

    _append_if_not_none(layers, build_norm_2d(cfg.norm, cfg.out_channels))
    layers.append(build_activation(cfg.activation))
    layers.append(build_conv_regularization(cfg.regularization, ndim=2))
    branch = nn.Sequential(*layers)
    initialize_module(
        branch,
        cfg.init,
        activation=cfg.activation.kind,
        primitive=_conv2d_primitive(cfg),
    )

    if residual is None:
        return branch

    shortcut = _make_conv2d_shortcut(cfg, residual)
    _initialize_shortcut(
        shortcut,
        cfg.init,
        activation=cfg.activation.kind,
        primitive="pointwise_conv2d",
    )
    return ResidualWrapper(
        branch,
        residual,
        shortcut=shortcut,
        norm_in=_make_conv2d_residual_norm(
            cfg,
            cfg.in_channels,
            residual,
            input_side=True,
        ),
        norm_out=_make_conv2d_residual_norm(
            cfg,
            cfg.out_channels,
            residual,
            input_side=False,
        ),
        scale_shape=(cfg.out_channels, 1, 1),
    )


def _append_dense_middle(
    layers: list[nn.Module],
    cfg: DenseMLPConfig,
    features: int,
) -> None:
    _append_if_not_none(layers, build_norm_dense(cfg.norm, features))
    layers.append(build_activation(cfg.activation))
    layers.append(build_dense_regularization(cfg.regularization))


def _make_conv1d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: str | int,
    groups: int,
    bias: bool,
) -> nn.Module:
    if padding == "same":
        return SamePadConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    if padding == "valid":
        padding = 0
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=int(padding),
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def _make_conv2d(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: str | int | tuple[int, int],
    groups: int,
    bias: bool,
) -> nn.Module:
    if padding == "same":
        return SamePadConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    if padding == "valid":
        padding = (0, 0)
    elif isinstance(padding, int):
        padding = (padding, padding)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def _make_dense_shortcut(
    in_features: int,
    out_features: int,
    residual: ResidualConfig,
) -> nn.Module:
    if residual.shortcut == "identity":
        if in_features != out_features:
            raise ValueError("identity shortcut requires in_features == out_features.")
        return nn.Identity()
    if residual.shortcut == "project" or in_features != out_features:
        return nn.Linear(in_features, out_features)
    return nn.Identity()


def _make_conv1d_shortcut(cfg: Conv1dConfig, residual: ResidualConfig) -> nn.Module:
    needs_project = cfg.in_channels != cfg.out_channels or cfg.stride != 1
    if residual.shortcut == "identity":
        if needs_project:
            raise ValueError("identity shortcut requires matching channels and stride=1.")
        return nn.Identity()
    if residual.shortcut == "project" or needs_project:
        return nn.Conv1d(
            cfg.in_channels,
            cfg.out_channels,
            kernel_size=1,
            stride=cfg.stride,
        )
    return nn.Identity()


def _make_conv2d_shortcut(cfg: Conv2dConfig, residual: ResidualConfig) -> nn.Module:
    stride = _as_pair(cfg.stride)
    needs_project = cfg.in_channels != cfg.out_channels or stride != (1, 1)
    if residual.shortcut == "identity":
        if needs_project:
            raise ValueError(
                "identity shortcut requires matching channels and stride=(1, 1)."
            )
        return nn.Identity()
    if residual.shortcut == "project" or needs_project:
        return nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=1, stride=stride)
    return nn.Identity()


def _make_dense_residual_norm(
    cfg: DenseMLPConfig,
    features: int,
    residual: ResidualConfig,
    *,
    input_side: bool,
) -> nn.Module | None:
    if residual.norm_position == "none":
        return None
    if input_side and residual.norm_position not in {"pre", "sandwich"}:
        return None
    if not input_side and residual.norm_position not in {"post", "sandwich"}:
        return None
    norm_cfg = cfg.norm if cfg.norm.kind is not None else NormConfig(kind="layer")
    return build_norm_dense(norm_cfg, features)


def _make_conv1d_residual_norm(
    cfg: Conv1dConfig,
    channels: int,
    residual: ResidualConfig,
    *,
    input_side: bool,
) -> nn.Module | None:
    if residual.norm_position == "none":
        return None
    if input_side and residual.norm_position not in {"pre", "sandwich"}:
        return None
    if not input_side and residual.norm_position not in {"post", "sandwich"}:
        return None
    norm_cfg = cfg.norm if cfg.norm.kind is not None else NormConfig(kind="layer")
    return build_norm_1d(norm_cfg, channels)


def _make_conv2d_residual_norm(
    cfg: Conv2dConfig,
    channels: int,
    residual: ResidualConfig,
    *,
    input_side: bool,
) -> nn.Module | None:
    if residual.norm_position == "none":
        return None
    if input_side and residual.norm_position not in {"pre", "sandwich"}:
        return None
    if not input_side and residual.norm_position not in {"post", "sandwich"}:
        return None
    norm_cfg = cfg.norm if cfg.norm.kind is not None else NormConfig(kind="layer")
    return build_norm_2d(norm_cfg, channels)


def _initialize_shortcut(
    shortcut: nn.Module,
    init_cfg,
    *,
    activation: str,
    primitive: str,
) -> None:
    if isinstance(shortcut, nn.Identity):
        return
    shortcut_cfg = type(init_cfg)(
        mode=init_cfg.mode,
        gain=init_cfg.gain,
        std=init_cfg.std,
        bias=init_cfg.bias,
        zero_last=False,
    )
    initialize_module(shortcut, shortcut_cfg, activation=activation, primitive=primitive)


def _validate_dense_cfg(cfg: DenseMLPConfig, out_features: int) -> None:
    _validate_positive("in_features", cfg.in_features)
    _validate_positive("hidden_features", cfg.hidden_features)
    _validate_positive("out_features", out_features)
    _validate_positive("depth", cfg.depth)


def _validate_conv1d_cfg(cfg: Conv1dConfig) -> None:
    _validate_positive("in_channels", cfg.in_channels)
    _validate_positive("out_channels", cfg.out_channels)
    _validate_positive("kernel_size", cfg.kernel_size)
    _validate_positive("stride", cfg.stride)
    _validate_positive("dilation", cfg.dilation)
    _validate_positive("groups", cfg.groups)
    if cfg.padding not in {"same", "valid"} and not isinstance(cfg.padding, int):
        raise ValueError("padding must be 'same', 'valid', or an integer.")
    if isinstance(cfg.padding, int) and cfg.padding < 0:
        raise ValueError("integer padding must be non-negative.")
    if cfg.separable and cfg.groups != 1:
        raise ValueError("separable=True manages groups internally; keep groups=1.")
    groups = cfg.in_channels if cfg.separable else cfg.groups
    if cfg.in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups.")
    if cfg.out_channels % groups != 0 and not cfg.separable:
        raise ValueError("out_channels must be divisible by groups.")


def _validate_conv2d_cfg(cfg: Conv2dConfig) -> None:
    _validate_positive("in_channels", cfg.in_channels)
    _validate_positive("out_channels", cfg.out_channels)
    for name, pair in {
        "kernel_size": _as_pair(cfg.kernel_size),
        "stride": _as_pair(cfg.stride),
        "dilation": _as_pair(cfg.dilation),
    }.items():
        _validate_positive(name, pair[0])
        _validate_positive(name, pair[1])
    _validate_positive("groups", cfg.groups)
    if cfg.padding not in {"same", "valid"}:
        if isinstance(cfg.padding, int):
            if cfg.padding < 0:
                raise ValueError("integer padding must be non-negative.")
        else:
            pad = _as_pair(cfg.padding)
            if pad[0] < 0 or pad[1] < 0:
                raise ValueError("padding values must be non-negative.")
    if cfg.separable and cfg.groups != 1:
        raise ValueError("separable=True manages groups internally; keep groups=1.")
    groups = cfg.in_channels if cfg.separable else cfg.groups
    if cfg.in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups.")
    if cfg.out_channels % groups != 0 and not cfg.separable:
        raise ValueError("out_channels must be divisible by groups.")


def _conv1d_primitive(cfg: Conv1dConfig) -> str:
    if cfg.separable:
        return "conv1d"
    if cfg.kernel_size == 1 and cfg.groups == 1:
        return "pointwise_conv1d"
    if cfg.groups == cfg.in_channels:
        return "depthwise_conv1d"
    return "conv1d"


def _conv2d_primitive(cfg: Conv2dConfig) -> str:
    kernel_size = _as_pair(cfg.kernel_size)
    if cfg.separable:
        return "conv2d"
    if kernel_size == (1, 1) and cfg.groups == 1:
        return "pointwise_conv2d"
    if cfg.groups == cfg.in_channels:
        return "depthwise_conv2d"
    return "conv2d"


def _same_pad_amount(length: int, kernel_size: int, stride: int, dilation: int) -> int:
    out_length = math.ceil(length / stride)
    return max((out_length - 1) * stride + (kernel_size - 1) * dilation + 1 - length, 0)


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError("Expected a pair.")
    return (int(value[0]), int(value[1]))


def _append_if_not_none(layers: list[nn.Module], layer: nn.Module | None) -> None:
    if layer is not None:
        layers.append(layer)


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
