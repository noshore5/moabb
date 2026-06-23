"""Generic PyTorch neural-network primitive components."""

from .activations import build_activation
from .blocks import build_conv1d_block, build_conv2d_block, build_dense_mlp
from .configs import (
    ActConfig,
    Conv1dConfig,
    Conv2dConfig,
    DenseMLPConfig,
    InitConfig,
    NormConfig,
    RegConfig,
    ResidualConfig,
    SeedConfig,
)
from .init import initialize_module, scoped_torch_init_seed
from .norms import build_norm_1d, build_norm_2d, build_norm_dense
from .regularization import DropPath
from .residual import ResidualWrapper
from .seed import set_reproducible_seed


__all__ = [
    "ActConfig",
    "Conv1dConfig",
    "Conv2dConfig",
    "DenseMLPConfig",
    "DropPath",
    "InitConfig",
    "NormConfig",
    "RegConfig",
    "ResidualConfig",
    "ResidualWrapper",
    "SeedConfig",
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
]
