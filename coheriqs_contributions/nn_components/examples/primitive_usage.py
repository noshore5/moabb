"""Small generic usage example for nn_components primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_path = str(REPO_ROOT)
if repo_root_path not in sys.path:
    sys.path.insert(0, repo_root_path)

from coheriqs_contributions.nn_components import (
    ActConfig,
    Conv1dConfig,
    Conv2dConfig,
    DenseMLPConfig,
    NormConfig,
    RegConfig,
    ResidualConfig,
    SeedConfig,
    build_conv1d_block,
    build_conv2d_block,
    build_dense_mlp,
    set_reproducible_seed,
)


def main() -> None:
    set_reproducible_seed(SeedConfig(seed=7))

    dense = build_dense_mlp(
        DenseMLPConfig(
            in_features=32,
            hidden_features=64,
            norm=NormConfig(kind="layer"),
            regularization=RegConfig(dropout=0.1),
        ),
        residual=ResidualConfig(layer_scale=1e-3),
    )
    conv1d = build_conv1d_block(
        Conv1dConfig(
            in_channels=16,
            out_channels=16,
            kernel_size=7,
            norm=NormConfig(kind="group", groups=4),
            activation=ActConfig(kind="silu"),
        ),
        residual=ResidualConfig(norm_position="pre", drop_path=0.1),
    )
    conv2d = build_conv2d_block(
        Conv2dConfig(
            in_channels=8,
            out_channels=16,
            kernel_size=(1, 5),
            stride=(1, 2),
            norm=NormConfig(kind="batch"),
        ),
        residual=ResidualConfig(shortcut="auto"),
    )

    print(dense(torch.randn(4, 32)).shape)
    print(conv1d(torch.randn(4, 16, 128)).shape)
    print(conv2d(torch.randn(4, 8, 16, 64)).shape)


if __name__ == "__main__":
    main()
