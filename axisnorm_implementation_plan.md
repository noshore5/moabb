# AxisNorm Implementation Plan

## Goal

Implement a general activation normalization module that can normalize over arbitrary tensor axes while keeping separate statistics and/or affine parameters over selected axes.

Primary target use case:

```python
# Spectrogram tensor
x.shape == [N, C, T, F]

# Normalize over batch and time, separately per channel and frequency
reduce_dims = (0, 2)
stat_dims = (1, 3)
affine_dims = (1, 3)
```

This should produce statistics shaped:

```python
[1, C, 1, F]
```

## Scope

Implement `AxisNorm`, a configurable activation norm covering:

- BatchNorm-like behavior
- LayerNorm-like behavior
- InstanceNorm-like behavior without PyTorch's special running-stat semantics
- Spectrogram frequency-wise normalization
- Other arbitrary axis-reduction norms

Do not implement in the first version:

- GroupNorm
- SyncBatchNorm
- LocalResponseNorm
- WeightNorm / SpectralNorm / Weight Standardization
- Conditional/adaptive affine parameters

## Proposed API

```python
AxisNorm(
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
    mode: str = "standard",  # initially support "standard" only; optionally add "rms" later
)
```

## Definitions

- `reduce_dims`: axes used to compute normalization statistics.
- `stat_dims`: axes preserved in the statistic tensors. Defaults to complement of `reduce_dims`.
- `affine_dims`: axes receiving learnable affine parameters. Defaults to `stat_dims` when `affine=True`.
- `dim_sizes`: fixed sizes for any axis used by `stat_dims` or `affine_dims` and not inferable as singleton.
- `batch_dim`: axis representing batch. Usually `0`.
- `track_running_stats`: whether to maintain BatchNorm-style running mean/variance for eval mode.

## Core Forward Behavior

Training or no running stats:

```python
var, mean = torch.var_mean(
    x,
    dim=reduce_dims,
    keepdim=True,
    correction=0,
)
y = (x - mean) * torch.rsqrt(var + eps)
```

Eval with running stats:

```python
mean = running_mean
var = running_var
y = (x - mean) * torch.rsqrt(var + eps)
```

Affine:

```python
if affine:
    y = y * weight + bias
```

## Shape Construction

Build all stat/affine tensors as broadcastable full-rank shapes of length `num_dims`.

Example:

```python
num_dims = 4
stat_dims = (1, 3)
dim_sizes = {1: C, 3: F}

shape = [1, C, 1, F]
```

For every dimension in `stat_dims` or `affine_dims`, its size must be known from `dim_sizes` at construction time unless a lazy-init strategy is intentionally added.

## Running Stats Rules

`track_running_stats=True` should only be allowed when `batch_dim in reduce_dims`.

Rationale:

- BatchNorm-style running stats are meaningful when stats are estimated across examples.
- LayerNorm-style stats are per sample and should not be stored as running stats.

Validation:

```python
if track_running_stats and batch_dim not in reduce_dims:
    raise ValueError(
        "track_running_stats=True requires batch_dim to be included in reduce_dims."
    )
```

Running buffers:

```python
running_mean: shape == stat_shape
running_var:  shape == stat_shape
```

Update rule:

```python
running_mean.lerp_(batch_mean.detach(), momentum)
running_var.lerp_(batch_var.detach(), momentum)
```

Use biased variance for normalization and for the running variance initially, matching the simple implementation. Do not add unbiased correction unless the codebase needs exact BatchNorm compatibility.

## Validation Rules

Normalize all dims to positive indices:

```python
dim = dim % num_dims
```

Validate:

- `num_dims > 0`
- all dims are unique within each dim tuple
- all dims are in range after normalization
- `reduce_dims` is not empty
- `stat_dims` and `reduce_dims` may be complements by default
- `affine_dims` may differ from `stat_dims`
- every non-singleton stat/affine dimension has a known size
- input tensor rank equals `num_dims`
- runtime input shape matches configured fixed sizes
- `track_running_stats=True` requires `batch_dim in reduce_dims`

## Suggested Examples

### Frequency-wise spectrogram norm

```python
AxisNorm(
    num_dims=4,
    reduce_dims=(0, 2),
    stat_dims=(1, 3),
    affine_dims=(1, 3),
    dim_sizes={1: C, 3: F},
    track_running_stats=True,
)
```

### BatchNorm2d-like

```python
AxisNorm(
    num_dims=4,
    reduce_dims=(0, 2, 3),
    stat_dims=(1,),
    affine_dims=(1,),
    dim_sizes={1: C},
    track_running_stats=True,
)
```

### LayerNorm-like over `[C, H, W]`

```python
AxisNorm(
    num_dims=4,
    reduce_dims=(1, 2, 3),
    stat_dims=(),
    affine_dims=(1, 2, 3),
    dim_sizes={1: C, 2: H, 3: W},
    track_running_stats=False,
)
```

### InstanceNorm2d-like without running stats

```python
AxisNorm(
    num_dims=4,
    reduce_dims=(2, 3),
    stat_dims=(0, 1),
    affine_dims=(1,),
    dim_sizes={1: C},
    track_running_stats=False,
)
```

Note: `stat_dims` includes `N`, but affine only uses `C`. This is why `stat_dims` and `affine_dims` must be separate.

## Implementation Steps

1. Add `AxisNorm` module in the project normalization/layers package.
2. Implement dim normalization and validation helpers.
3. Implement broadcast-shape construction for stat buffers and affine parameters.
4. Register `weight` and `bias` only when `affine=True`.
5. Register `running_mean` and `running_var` only when `track_running_stats=True`.
6. Implement forward path using `torch.var_mean(..., keepdim=True, correction=0)`.
7. Update running stats only in training mode and only under `torch.no_grad()`.
8. Use running stats in eval mode when `track_running_stats=True`.
9. Add runtime shape checks for configured dimensions.
10. Add tests comparing special cases to equivalent manual calculations.

## Tests

Required tests:

- Spectrogram case: `[N, C, T, F]`, reduce `(0, 2)`, output has zero mean and unit variance over `N x T` for each `C x F`.
- BatchNorm2d-like case: compare training output to manual reduction over `(0, 2, 3)`.
- LayerNorm-like case: compare to manual reduction over `(1, 2, 3)`.
- InstanceNorm-like case: compare to manual reduction over `(2, 3)`.
- `affine_dims != stat_dims` works.
- eval mode uses running stats.
- running stats update in training mode.
- `track_running_stats=True` raises if `batch_dim not in reduce_dims`.
- invalid dims raise clear errors.
- fixed configured sizes are checked at runtime.
- gradients flow through input, weight, and bias.

## Performance Notes

Default implementation should not permute tensors.

Use:

```python
torch.var_mean(x, dim=reduce_dims, keepdim=True, correction=0)
```

Avoid `.permute(...).contiguous()` unless benchmarks prove it helps for a specific shape/layout.

Potential later optimization:

- add specialized `FreqBatchNorm2d` wrapper for the common spectrogram case
- benchmark with `torch.compile`
- benchmark channels-last or alternative layouts only after correctness is locked down

## Optional Future Extensions

- `mode="rms"` for RMSNorm-like behavior without mean subtraction
- lazy initialization from first input shape
- distributed stats for SyncBatchNorm-like behavior
- grouped axis normalization via reshape/group config
- specialized wrappers:
  - `FreqBatchNorm2d`
  - `AxisBatchNorm2d`
  - `AxisLayerNorm`
