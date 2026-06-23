"""Initialization policies for primitive modules."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
import torch.nn as nn

from .configs import InitConfig
from .norms import RMSNorm


_ALLOWED_MODES = {
    "dense": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "orthogonal",
        "zeros",
    },
    "conv1d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "orthogonal",
        "zeros",
    },
    "conv2d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "orthogonal",
        "zeros",
    },
    "depthwise_conv1d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "zeros",
    },
    "depthwise_conv2d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "zeros",
    },
    "pointwise_conv1d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "orthogonal",
        "zeros",
    },
    "pointwise_conv2d": {
        "torch_default",
        "auto",
        "kaiming",
        "xavier",
        "lecun",
        "normal",
        "trunc_normal",
        "orthogonal",
        "zeros",
    },
}


@contextmanager
def scoped_torch_init_seed(seed: int | None) -> Iterator[None]:
    """Temporarily seed CPU initialization without advancing the caller's RNG."""

    if seed is None:
        yield
        return

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        yield


def initialize_module(
    module: nn.Module,
    cfg: InitConfig,
    *,
    activation: str,
    primitive: str,
) -> None:
    """Initialize Linear/Conv weights and norm affine parameters in a module tree."""

    if primitive not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported primitive: {primitive!r}")
    if cfg.mode not in _ALLOWED_MODES[primitive]:
        raise ValueError(f"Init mode {cfg.mode!r} is not supported for {primitive!r}.")
    if cfg.mode == "torch_default":
        if cfg.zero_last:
            raise ValueError("zero_last=True is not compatible with torch_default init.")
        return
    if cfg.bias not in {"zeros", "normal"}:
        raise ValueError("bias must be one of {'zeros', 'normal'}.")

    leaf_layers: list[nn.Module] = []
    for submodule in module.modules():
        if isinstance(submodule, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            _initialize_weight(submodule, cfg, activation=activation, primitive=primitive)
            _initialize_bias(submodule, cfg)
            leaf_layers.append(submodule)
        elif _has_affine_norm_parameters(submodule):
            _initialize_norm_affine(submodule)

    if cfg.zero_last:
        if not leaf_layers:
            raise ValueError("zero_last=True requires at least one Linear/Conv layer.")
        nn.init.zeros_(leaf_layers[-1].weight)
        if leaf_layers[-1].bias is not None:
            nn.init.zeros_(leaf_layers[-1].bias)


def _initialize_weight(
    module: nn.Linear | nn.Conv1d | nn.Conv2d,
    cfg: InitConfig,
    *,
    activation: str,
    primitive: str,
) -> None:
    mode = _resolve_auto_mode(cfg.mode, activation)
    gain = cfg.gain if cfg.gain is not None else _activation_gain(activation)

    if mode == "zeros":
        nn.init.zeros_(module.weight)
    elif mode == "kaiming":
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif mode == "xavier":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif mode == "lecun":
        nn.init.normal_(module.weight, mean=0.0, std=_lecun_std(module.weight))
    elif mode == "normal":
        nn.init.normal_(module.weight, mean=0.0, std=cfg.std or 0.02)
    elif mode == "trunc_normal":
        nn.init.trunc_normal_(module.weight, mean=0.0, std=cfg.std or 0.02)
    elif mode == "orthogonal":
        if isinstance(module, (nn.Conv1d, nn.Conv2d)) and module.groups != 1:
            raise ValueError(
                "orthogonal init is not supported for grouped/depthwise conv."
            )
        if primitive.startswith("depthwise"):
            raise ValueError("orthogonal init is not supported for depthwise conv.")
        nn.init.orthogonal_(module.weight, gain=gain)
    else:
        raise ValueError(f"Unsupported resolved init mode: {mode!r}")


def _initialize_bias(module: nn.Linear | nn.Conv1d | nn.Conv2d, cfg: InitConfig) -> None:
    if module.bias is None:
        return
    if cfg.bias == "zeros":
        nn.init.zeros_(module.bias)
    elif cfg.bias == "normal":
        nn.init.normal_(module.bias, mean=0.0, std=cfg.std or 0.02)


def _initialize_norm_affine(module: nn.Module) -> None:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if weight is not None:
        nn.init.ones_(weight)
    if bias is not None:
        nn.init.zeros_(bias)


def _has_affine_norm_parameters(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.GroupNorm,
            nn.LayerNorm,
            RMSNorm,
        ),
    )


def _resolve_auto_mode(mode: str, activation: str) -> str:
    if mode != "auto":
        return mode
    if activation in {"relu", "elu"}:
        return "kaiming"
    return "xavier"


def _activation_gain(activation: str) -> float:
    if activation == "tanh":
        return nn.init.calculate_gain("tanh")
    if activation == "relu":
        return nn.init.calculate_gain("relu")
    return 1.0


def _lecun_std(weight) -> float:
    fan_in = _fan_in(weight)
    return math.sqrt(1.0 / fan_in)


def _fan_in(weight) -> int:
    if weight.ndim < 2:
        raise ValueError("fan_in requires at least a 2D weight tensor.")
    if weight.ndim == 2:
        return int(weight.shape[1])
    receptive_field = 1
    for size in weight.shape[2:]:
        receptive_field *= int(size)
    return int(weight.shape[1] * receptive_field)
