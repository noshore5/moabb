"""Generic PyTorch neural-network primitive components."""

from .activations import build_activation
from .blocks import build_conv1d_block, build_conv2d_block, build_dense_mlp
from .configs import (
    ActConfig,
    CategoricalGateConfig,
    Conv1dConfig,
    Conv2dConfig,
    DenseMLPConfig,
    InitConfig,
    NormConfig,
    RegConfig,
    ResidualConfig,
    SelectPathConfig,
    SeedConfig,
)
from .init import initialize_module, scoped_torch_init_seed
from .norms import AxisNorm, build_norm_1d, build_norm_2d, build_norm_dense
from .regularization import DropPath
from .residual import ResidualWrapper
from .seed import set_reproducible_seed
from .selectable import (
    CategoricalChoice,
    CategoricalGate,
    GateInfo,
    SelectableActivation,
    SelectPath,
    ZeroUpdate,
    export_selectable_modules,
)


__all__ = [
    "ActConfig",
    "AxisNorm",
    "CategoricalChoice",
    "CategoricalGate",
    "CategoricalGateConfig",
    "Conv1dConfig",
    "Conv2dConfig",
    "DenseMLPConfig",
    "DropPath",
    "GateInfo",
    "InitConfig",
    "NormConfig",
    "RegConfig",
    "ResidualConfig",
    "ResidualWrapper",
    "SelectPath",
    "SelectPathConfig",
    "SelectableActivation",
    "SeedConfig",
    "ZeroUpdate",
    "build_activation",
    "build_conv1d_block",
    "build_conv2d_block",
    "build_dense_mlp",
    "build_norm_1d",
    "build_norm_2d",
    "build_norm_dense",
    "initialize_module",
    "scoped_torch_init_seed",
    "set_reproducible_seed",
    "export_selectable_modules",
]
