from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coheriqs_contributions.nn_components import (
    ActConfig,
    AxisNorm,
    Conv1dConfig,
    Conv2dConfig,
    DenseMLPConfig,
    DropPath,
    InitConfig,
    NormConfig,
    RegConfig,
    ResidualConfig,
    SeedConfig,
    build_conv1d_block,
    build_conv2d_block,
    build_dense_mlp,
    build_norm_1d,
    build_norm_2d,
    build_norm_dense,
    scoped_torch_init_seed,
    set_reproducible_seed,
)


def _manual_axis_norm(
    x: torch.Tensor,
    reduce_dims: tuple[int, ...],
    eps: float = 1e-5,
) -> torch.Tensor:
    var, mean = torch.var_mean(x, dim=reduce_dims, keepdim=True, correction=0)
    return (x - mean) * torch.rsqrt(var + eps)


def test_dense_mlp_residual_projection_shape() -> None:
    block = build_dense_mlp(
        DenseMLPConfig(
            in_features=8,
            hidden_features=16,
            out_features=12,
            norm=NormConfig(kind="layer"),
        ),
        residual=ResidualConfig(shortcut="auto"),
    )

    y = block(torch.randn(3, 8))

    assert y.shape == (3, 12)


def test_conv1d_normal_depthwise_pointwise_and_separable_shapes() -> None:
    x = torch.randn(2, 4, 17)

    normal = build_conv1d_block(
        Conv1dConfig(4, 4, kernel_size=3),
        residual=ResidualConfig(),
    )
    depthwise = build_conv1d_block(Conv1dConfig(4, 4, kernel_size=5, groups=4))
    pointwise = build_conv1d_block(Conv1dConfig(4, 6, kernel_size=1))
    separable = build_conv1d_block(Conv1dConfig(4, 6, kernel_size=3, separable=True))

    assert normal(x).shape == x.shape
    assert depthwise(x).shape == x.shape
    assert pointwise(x).shape == (2, 6, 17)
    assert separable(x).shape == (2, 6, 17)


def test_conv1d_residual_projection_with_stride() -> None:
    block = build_conv1d_block(
        Conv1dConfig(4, 6, kernel_size=3, stride=2),
        residual=ResidualConfig(shortcut="auto"),
    )

    assert block(torch.randn(2, 4, 17)).shape == (2, 6, 9)


def test_conv2d_anisotropic_separable_and_residual_projection_shapes() -> None:
    x = torch.randn(2, 4, 8, 17)
    anisotropic = build_conv2d_block(Conv2dConfig(4, 4, kernel_size=(1, 5)))
    separable = build_conv2d_block(Conv2dConfig(4, 8, kernel_size=3, separable=True))
    projected = build_conv2d_block(
        Conv2dConfig(4, 8, kernel_size=(1, 5), stride=(1, 2)),
        residual=ResidualConfig(shortcut="auto"),
    )

    assert anisotropic(x).shape == x.shape
    assert separable(x).shape == (2, 8, 8, 17)
    assert projected(x).shape == (2, 8, 8, 9)


def test_norm_factories_preserve_layouts() -> None:
    dense_norm = build_norm_dense(NormConfig(kind="rms"), 5)
    norm1d = build_norm_1d(NormConfig(kind="layer"), 4)
    norm2d = build_norm_2d(NormConfig(kind="rms"), 3)

    assert dense_norm is not None
    assert norm1d is not None
    assert norm2d is not None
    assert dense_norm(torch.randn(2, 5)).shape == (2, 5)
    assert norm1d(torch.randn(2, 4, 6)).shape == (2, 4, 6)
    assert norm2d(torch.randn(2, 3, 4, 5)).shape == (2, 3, 4, 5)


def test_axis_norm_spectrogram_stats_and_running_eval() -> None:
    x = torch.arange(4 * 3 * 5 * 2, dtype=torch.float32).reshape(4, 3, 5, 2)
    x = x / 10.0
    layer = AxisNorm(
        num_dims=4,
        reduce_dims=(0, 2),
        dim_sizes={1: 3, 3: 2},
        track_running_stats=True,
    )

    y = layer(x)

    assert torch.allclose(
        y.mean(dim=(0, 2), keepdim=True),
        torch.zeros(1, 3, 1, 2),
        atol=1e-6,
    )
    assert torch.allclose(
        y.var(dim=(0, 2), keepdim=True, correction=0),
        torch.ones(1, 3, 1, 2),
        atol=1e-4,
    )
    batch_var, batch_mean = torch.var_mean(
        x,
        dim=(0, 2),
        keepdim=True,
        correction=0,
    )
    assert torch.allclose(layer.running_mean, batch_mean * layer.momentum)
    assert torch.allclose(
        layer.running_var,
        torch.ones_like(batch_var).lerp(batch_var, layer.momentum),
    )

    layer.eval()
    x_eval = x + 1.0
    expected_eval = (x_eval - layer.running_mean) * torch.rsqrt(
        layer.running_var + layer.eps
    )
    assert torch.allclose(layer(x_eval), expected_eval)


@pytest.mark.parametrize(
    ("layer", "x", "reduce_dims"),
    [
        (
            AxisNorm(
                num_dims=4,
                reduce_dims=(0, 2, 3),
                stat_dims=(1,),
                affine_dims=(1,),
                dim_sizes={1: 3},
            ),
            torch.randn(2, 3, 4, 5),
            (0, 2, 3),
        ),
        (
            AxisNorm(
                num_dims=4,
                reduce_dims=(1, 2, 3),
                stat_dims=(),
                affine_dims=(1, 2, 3),
                dim_sizes={1: 3, 2: 4, 3: 5},
            ),
            torch.randn(2, 3, 4, 5),
            (1, 2, 3),
        ),
        (
            AxisNorm(
                num_dims=4,
                reduce_dims=(2, 3),
                stat_dims=(0, 1),
                affine_dims=(1,),
                dim_sizes={1: 3},
            ),
            torch.randn(2, 3, 4, 5),
            (2, 3),
        ),
    ],
    ids=["batch2d", "layer2d", "instance2d"],
)
def test_axis_norm_matches_manual_special_cases(
    layer: AxisNorm,
    x: torch.Tensor,
    reduce_dims: tuple[int, ...],
) -> None:
    assert torch.allclose(layer(x), _manual_axis_norm(x, reduce_dims))


def test_axis_norm_rejects_invalid_configs_and_shapes() -> None:
    with pytest.raises(ValueError, match="batch_dim"):
        AxisNorm(num_dims=4, reduce_dims=(1, 2, 3), track_running_stats=True)

    with pytest.raises(ValueError, match="out of range"):
        AxisNorm(num_dims=4, reduce_dims=(4,))

    with pytest.raises(ValueError, match="unique"):
        AxisNorm(num_dims=4, reduce_dims=(0, -4))

    with pytest.raises(ValueError, match="dim_sizes"):
        AxisNorm(num_dims=4, reduce_dims=(0, 2), affine_dims=(1,), dim_sizes={})

    layer = AxisNorm(
        num_dims=4,
        reduce_dims=(0, 2),
        dim_sizes={1: 3, 3: 2},
        affine=False,
    )
    with pytest.raises(ValueError, match="dimension 3"):
        layer(torch.randn(2, 3, 4, 5))


def test_axis_norm_gradients_flow() -> None:
    x = torch.randn(2, 3, 4, requires_grad=True)
    layer = AxisNorm(num_dims=3, reduce_dims=(0, 2), dim_sizes={1: 3})

    layer(x).square().mean().backward()

    assert x.grad is not None
    assert layer.weight is not None
    assert layer.bias is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert torch.isfinite(x.grad).all()


def test_drop_path_train_eval_behavior() -> None:
    layer = DropPath(0.5)
    x = torch.ones(32, 3)

    layer.eval()
    assert torch.equal(layer(x), x)

    set_reproducible_seed(SeedConfig(seed=3))
    layer.train()
    y = layer(x)

    assert set(torch.unique(y).tolist()).issubset({0.0, 2.0})


def test_zero_last_initialization_zeros_final_projection() -> None:
    block = build_dense_mlp(
        DenseMLPConfig(
            in_features=8,
            hidden_features=16,
            init=InitConfig(mode="xavier", zero_last=True),
        )
    )
    linear_layers = [m for m in block.modules() if isinstance(m, nn.Linear)]

    assert torch.count_nonzero(linear_layers[-1].weight).item() == 0
    assert torch.count_nonzero(linear_layers[-1].bias).item() == 0


def test_torch_default_initialization_preserves_linear_defaults() -> None:
    set_reproducible_seed(SeedConfig(seed=13))
    expected = nn.Linear(4, 3)

    set_reproducible_seed(SeedConfig(seed=13))
    block = build_dense_mlp(
        DenseMLPConfig(
            in_features=4,
            hidden_features=3,
            out_features=3,
            depth=1,
            init=InitConfig(mode="torch_default"),
        )
    )
    actual = [m for m in block.modules() if isinstance(m, nn.Linear)][0]

    assert torch.equal(actual.weight, expected.weight)
    assert torch.equal(actual.bias, expected.bias)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: build_dense_mlp(DenseMLPConfig(4, 6), init_seed=11),
        lambda: build_conv1d_block(Conv1dConfig(4, 6), init_seed=11),
        lambda: build_conv2d_block(Conv2dConfig(4, 6), init_seed=11),
    ],
    ids=["dense", "conv1d", "conv2d"],
)
def test_builder_init_seed_is_reproducible(builder) -> None:
    first = builder()
    second = builder()

    for p_first, p_second in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(p_first, p_second)


def test_model_seed_stream_gives_equal_shaped_components_distinct_weights() -> None:
    cfg = DenseMLPConfig(
        in_features=4,
        hidden_features=4,
        out_features=4,
        depth=1,
        init=InitConfig(mode="torch_default"),
    )

    with scoped_torch_init_seed(17):
        first = build_dense_mlp(cfg)
        second = build_dense_mlp(cfg)
    with scoped_torch_init_seed(17):
        repeated_first = build_dense_mlp(cfg)
        repeated_second = build_dense_mlp(cfg)

    first_weight = next(first.parameters())
    second_weight = next(second.parameters())
    assert not torch.equal(first_weight, second_weight)
    assert torch.equal(first_weight, next(repeated_first.parameters()))
    assert torch.equal(second_weight, next(repeated_second.parameters()))


def test_invalid_configs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="dropout"):
        build_dense_mlp(
            DenseMLPConfig(
                in_features=4,
                hidden_features=8,
                regularization=RegConfig(dropout=1.0),
            )
        )

    with pytest.raises(ValueError, match="groups"):
        build_conv1d_block(
            Conv1dConfig(4, 4, groups=3, norm=NormConfig(kind="group", groups=3))
        )

    with pytest.raises(ValueError, match="identity shortcut"):
        build_conv2d_block(
            Conv2dConfig(4, 8, kernel_size=3),
            residual=ResidualConfig(shortcut="identity"),
        )

    with pytest.raises(ValueError, match="orthogonal"):
        build_conv1d_block(
            Conv1dConfig(
                4,
                4,
                kernel_size=3,
                groups=4,
                init=InitConfig(mode="orthogonal"),
            )
        )

    with pytest.raises(ValueError, match="rezero and layer_scale"):
        build_dense_mlp(
            DenseMLPConfig(4, 8),
            residual=ResidualConfig(rezero=True, layer_scale=1e-3),
        )
