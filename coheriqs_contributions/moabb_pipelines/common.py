"""Shared utilities for local MOABB-compatible classifiers."""

from __future__ import annotations

import hashlib
import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable, Sequence

import numpy as np
from scipy.signal import resample
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def validate_eeg_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_channels, n_timepoints)")
    return np.asarray(X, dtype=np.float32)


def fit_global_zscore_stats(X: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(X))
    std = float(np.std(X))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def apply_global_zscore(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (X - mean) / (std + 1e-8)


def fit_minmax_stats(X: np.ndarray) -> tuple[float, float]:
    min_value = float(np.min(X))
    max_value = float(np.max(X))
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return 0.0, 0.0
    return min_value, max_value


def apply_minmax(X: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if max_value > min_value:
        return (X - min_value) / (max_value - min_value)
    return np.zeros_like(X)


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class _SelectorTrainingSpec:
    name: str
    module: nn.Module
    gate: nn.Module
    params: tuple[nn.Parameter, ...]
    alpha_optim: str
    alpha_update_split: str


def _collect_selector_training_specs(model: nn.Module) -> list[_SelectorTrainingSpec]:
    specs: list[_SelectorTrainingSpec] = []
    seen_param_ids: set[int] = set()

    for name, module in model.named_modules():
        selection_parameters = getattr(module, "selection_parameters", None)
        if not callable(selection_parameters):
            continue

        params = []
        for param in selection_parameters():
            if not isinstance(param, nn.Parameter) or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            params.append(param)

        if not params:
            continue

        gate = getattr(module, "gate", module)
        alpha_optim = getattr(gate, "alpha_optim", "shared")
        alpha_update_split = getattr(gate, "alpha_update_split", "train")
        if alpha_optim not in {"shared", "separate"}:
            raise ValueError("selector alpha_optim must be 'shared' or 'separate'.")
        if alpha_update_split not in {"train", "val"}:
            raise ValueError("selector alpha_update_split must be 'train' or 'val'.")

        specs.append(
            _SelectorTrainingSpec(
                name=name or module.__class__.__name__,
                module=module,
                gate=gate,
                params=tuple(params),
                alpha_optim=alpha_optim,
                alpha_update_split=alpha_update_split,
            )
        )

    return specs


def _selector_spec_matches(
    spec: _SelectorTrainingSpec,
    *,
    alpha_optim: str | None = None,
    alpha_update_split: str | None = None,
) -> bool:
    if alpha_optim is not None and spec.alpha_optim != alpha_optim:
        return False
    if alpha_update_split is not None and spec.alpha_update_split != alpha_update_split:
        return False
    return True


def _filter_selector_specs(
    specs: Sequence[_SelectorTrainingSpec],
    *,
    alpha_optim: str | None = None,
    alpha_update_split: str | None = None,
) -> tuple[_SelectorTrainingSpec, ...]:
    return tuple(
        spec
        for spec in specs
        if _selector_spec_matches(
            spec,
            alpha_optim=alpha_optim,
            alpha_update_split=alpha_update_split,
        )
    )


def _selector_params(
    specs: Sequence[_SelectorTrainingSpec],
    *,
    alpha_optim: str | None = None,
    alpha_update_split: str | None = None,
) -> tuple[nn.Parameter, ...]:
    params: list[nn.Parameter] = []
    for spec in _filter_selector_specs(
        specs,
        alpha_optim=alpha_optim,
        alpha_update_split=alpha_update_split,
    ):
        params.extend(spec.params)
    return tuple(params)


def _selector_gate_modules(
    specs: Sequence[_SelectorTrainingSpec],
    *,
    alpha_optim: str | None = None,
    alpha_update_split: str | None = None,
) -> tuple[nn.Module, ...]:
    modules: list[nn.Module] = []
    seen_module_ids: set[int] = set()
    for spec in _filter_selector_specs(
        specs,
        alpha_optim=alpha_optim,
        alpha_update_split=alpha_update_split,
    ):
        module_id = id(spec.gate)
        if module_id in seen_module_ids:
            continue
        seen_module_ids.add(module_id)
        modules.append(spec.gate)
    return tuple(modules)


def _selector_extra_loss(
    specs: Sequence[_SelectorTrainingSpec],
    reference: torch.Tensor,
    *,
    alpha_optim: str | None = None,
    alpha_update_split: str | None = None,
) -> torch.Tensor:
    loss = reference.new_zeros(())
    for spec in _filter_selector_specs(
        specs,
        alpha_optim=alpha_optim,
        alpha_update_split=alpha_update_split,
    ):
        extra_loss = getattr(spec.module, "extra_loss", None)
        if callable(extra_loss):
            loss = loss + extra_loss()
    return loss


def _new_selector_diagnostic_accumulators(
    specs: Sequence[_SelectorTrainingSpec],
) -> dict[int, dict[str, object]]:
    return {
        id(spec.module): {
            "spec": spec,
            "count": 0,
            "probs_sum": None,
            "logits_sum": None,
            "entropy_sum": 0.0,
            "expected_cost_sum": None,
            "selected_counts": None,
            "last_mode": None,
            "last_gradient_mode": None,
            "last_temperature": None,
            "last_exploration_epsilon": None,
        }
        for spec in specs
    }


def _selector_last_gate_info(spec: _SelectorTrainingSpec):
    return getattr(spec.module, "last_gate_info", None)


def _mean_over_scope_last_dim(value: torch.Tensor) -> torch.Tensor:
    detached = value.detach().float().cpu()
    if detached.ndim == 0:
        return detached.reshape(1)
    return detached.reshape(-1, int(detached.shape[-1])).mean(dim=0)


def _record_selector_diagnostics(
    accumulators: dict[int, dict[str, object]],
    specs: Sequence[_SelectorTrainingSpec],
) -> None:
    for spec in specs:
        gate_info = _selector_last_gate_info(spec)
        if gate_info is None:
            continue
        accumulator = accumulators.get(id(spec.module))
        if accumulator is None:
            continue

        probs = _mean_over_scope_last_dim(gate_info.probs)
        logits = _mean_over_scope_last_dim(gate_info.logits)
        accumulator["count"] = int(accumulator["count"]) + 1
        accumulator["probs_sum"] = (
            probs
            if accumulator["probs_sum"] is None
            else accumulator["probs_sum"] + probs
        )
        accumulator["logits_sum"] = (
            logits
            if accumulator["logits_sum"] is None
            else accumulator["logits_sum"] + logits
        )
        accumulator["entropy_sum"] = float(accumulator["entropy_sum"]) + float(
            gate_info.entropy.detach().float().mean().cpu().item()
        )
        if gate_info.expected_cost is not None:
            expected_cost = float(
                gate_info.expected_cost.detach().float().mean().cpu().item()
            )
            accumulator["expected_cost_sum"] = (
                expected_cost
                if accumulator["expected_cost_sum"] is None
                else float(accumulator["expected_cost_sum"]) + expected_cost
            )
        if gate_info.selected_index is not None:
            selected = gate_info.selected_index.detach().reshape(-1).cpu().long()
            num_choices = int(getattr(spec.gate, "num_choices", probs.numel()))
            counts = torch.bincount(selected, minlength=num_choices)
            accumulator["selected_counts"] = (
                counts
                if accumulator["selected_counts"] is None
                else accumulator["selected_counts"] + counts
            )
        accumulator["last_mode"] = gate_info.mode
        accumulator["last_gradient_mode"] = gate_info.gradient_mode
        accumulator["last_temperature"] = float(gate_info.temperature)
        accumulator["last_exploration_epsilon"] = gate_info.exploration_epsilon


def _finalize_selector_diagnostics(
    accumulators: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for accumulator in accumulators.values():
        count = int(accumulator["count"])
        if count == 0:
            continue

        spec = accumulator["spec"]
        probs_mean = (accumulator["probs_sum"] / count).tolist()
        logits_mean = (accumulator["logits_sum"] / count).tolist()
        summary: dict[str, object] = {
            "name": spec.name,
            "alpha_optim": spec.alpha_optim,
            "alpha_update_split": spec.alpha_update_split,
            "num_forwards": count,
            "mode": accumulator["last_mode"],
            "gradient_mode": accumulator["last_gradient_mode"],
            "temperature": accumulator["last_temperature"],
            "exploration_epsilon": accumulator["last_exploration_epsilon"],
            "probs_mean": [float(value) for value in probs_mean],
            "logits_mean": [float(value) for value in logits_mean],
            "entropy_mean": float(accumulator["entropy_sum"]) / count,
        }
        if accumulator["expected_cost_sum"] is not None:
            summary["expected_cost_mean"] = (
                float(accumulator["expected_cost_sum"]) / count
            )
        if accumulator["selected_counts"] is not None:
            selected_counts = accumulator["selected_counts"].tolist()
            total = max(sum(int(value) for value in selected_counts), 1)
            summary["selected_counts"] = [int(value) for value in selected_counts]
            summary["selected_fractions"] = [
                float(value) / float(total) for value in selected_counts
            ]
        summaries.append(summary)
    return summaries


def _format_selector_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _min_accepted_batch_size(batch_size: int, last_batch_min_ratio: float) -> int:
    return max(1, int(math.ceil(int(batch_size) * float(last_batch_min_ratio))))


def _batch_sample_count(batch) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if isinstance(batch, (list, tuple)):
        for item in reversed(batch):
            if isinstance(item, torch.Tensor):
                return int(item.shape[0])
    raise ValueError("Batch must contain at least one tensor with a sample dimension.")


def _batch_meets_min_size(batch, min_batch_size: int) -> bool:
    return _batch_sample_count(batch) >= int(min_batch_size)


def _count_eligible_tensor_batches(loader: DataLoader, min_batch_size: int) -> int:
    dataset_size = len(loader.dataset)
    batch_size = loader.batch_size
    if batch_size is None:
        return sum(1 for batch in loader if _batch_meets_min_size(batch, min_batch_size))
    full_batches, remainder = divmod(int(dataset_size), int(batch_size))
    return full_batches + (1 if remainder >= int(min_batch_size) else 0)


def _optimizer_parameters(optimizer) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def _slice_tensor_batch(batch, start: int, end: int):
    return tuple(
        item[start:end] if isinstance(item, torch.Tensor) else item for item in batch
    )


def _scale_optimizer_gradients(optimizer, factor: float) -> None:
    with torch.no_grad():
        for parameter in _optimizer_parameters(optimizer):
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


@contextmanager
def _temporarily_requires_grad(
    params: Sequence[nn.Parameter],
    requires_grad: bool,
):
    states = [(param, param.requires_grad) for param in params]
    for param, _ in states:
        param.requires_grad_(requires_grad)
    try:
        yield
    finally:
        for param, old_requires_grad in states:
            param.requires_grad_(old_requires_grad)


@contextmanager
def _selector_alpha_update_scope(
    model: nn.Module,
    alpha_params: Sequence[nn.Parameter],
    gate_modules: Sequence[nn.Module],
):
    module_states = [(module, module.training) for module in model.modules()]
    param_states = [(param, param.requires_grad) for param in model.parameters()]
    alpha_param_ids = {id(param) for param in alpha_params}

    model.eval()
    for gate in gate_modules:
        gate.train()
    for param in model.parameters():
        param.requires_grad_(id(param) in alpha_param_ids)

    try:
        yield
    finally:
        for param, requires_grad in param_states:
            param.requires_grad_(requires_grad)
        for module, training in module_states:
            module.train(training)


def safe_roc_auc(
    y_true: np.ndarray, proba: np.ndarray, n_classes: int
) -> float | None:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    try:
        if n_classes <= 2:
            score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return float(roc_auc_score(y_true, score))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except ValueError:
        return None


def fmt_metric(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{ndigits}f}"


def print_torch_parameter_summary(model: nn.Module, header: str = "Model") -> None:
    print(f"[{header}] Parameter summary", flush=True)
    total_params = 0
    total_trainable = 0
    for name, param in model.named_parameters():
        n_params = int(param.numel())
        total_params += n_params
        if param.requires_grad:
            total_trainable += n_params
        print(
            f"[{header}] {name}: shape={tuple(param.shape)} params={n_params} "
            f"trainable={param.requires_grad}",
            flush=True,
        )
    print(
        f"[{header}] total_params={total_params} "
        f"trainable_params={total_trainable} "
        f"non_trainable_params={total_params - total_trainable}",
        flush=True,
    )


def torch_parameter_hashes(
    model: nn.Module,
    *,
    precision: float = 1e-5,
) -> tuple[dict[str, str], str, str]:
    """Return tolerance-aware parameter hashes and value/name model hashes."""

    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be a finite positive number.")

    parameter_hashes = {}
    value_model_hasher = hashlib.blake2b(digest_size=16)
    named_model_hasher = hashlib.blake2b(digest_size=16)
    for name, param in model.named_parameters():
        param_hasher = hashlib.blake2b(digest_size=16)
        metadata = f"{tuple(param.shape)}|{param.dtype}|{precision:.17g}"
        param_hasher.update(metadata.encode("utf-8"))
        param_hasher.update(_quantized_parameter_bytes(param, precision))
        digest = param_hasher.hexdigest()
        parameter_hashes[name] = digest
        value_model_hasher.update(bytes.fromhex(digest))
        named_model_hasher.update(name.encode("utf-8"))
        named_model_hasher.update(b"\0")
        named_model_hasher.update(bytes.fromhex(digest))

    return parameter_hashes, value_model_hasher.hexdigest(), named_model_hasher.hexdigest()


def print_torch_parameter_hashes(
    model: nn.Module,
    header: str = "Model",
    *,
    precision: float = 1e-5,
) -> None:
    """Print reproducible per-parameter and global model fingerprints."""

    parameter_hashes, value_model_hash, named_model_hash = torch_parameter_hashes(
        model,
        precision=precision,
    )
    print(
        f"[{header}] Parameter hashes algorithm=blake2b-128 "
        f"precision={precision:.1e}",
        flush=True,
    )
    for name, digest in parameter_hashes.items():
        print(f"[{header}] {name}: weight_hash={digest}", flush=True)
    print(f"[{header}] weight_model_hash={value_model_hash}", flush=True)
    print(f"[{header}] named_model_hash={named_model_hash}", flush=True)


def print_torch_custom_model_summary(
    model: nn.Module,
    header: str = "Model",
) -> None:
    """Print optional model-specific summary details."""
    custom_summary = getattr(model, "print_custom_summary", None)
    if callable(custom_summary):
        custom_summary(header=header)


def _quantized_parameter_bytes(param: torch.Tensor, precision: float) -> bytes:
    values = param.detach().cpu()
    if values.is_complex():
        values = torch.view_as_real(values)
    if values.is_floating_point():
        values = torch.round(values.to(torch.float64) / precision).to(torch.int64)
    return np.ascontiguousarray(values.numpy()).tobytes()


def resolve_coherence_utils():
    repo_root = Path(__file__).resolve().parents[2]
    configured_root = os.environ.get("WCT_COHERENT_MULTIPLEX_ROOT")
    coherent_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else repo_root / "Coherent_Multiplex"
    )
    coherence_module = coherent_root / "utils" / "coherence_utils.py"
    if not coherence_module.is_file():
        raise ImportError(
            "Coherent_Multiplex root does not contain utils/coherence_utils.py: "
            f"{coherent_root}"
        )
    coherent_root_str = str(coherent_root)
    if coherent_root_str not in sys.path:
        sys.path.insert(0, coherent_root_str)
    try:
        from utils import coherence_utils  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import Coherent_Multiplex wavelet helpers. "
            "Ensure Coherent_Multiplex is present and dependencies are installed."
        ) from exc
    if configured_root:
        resolved_module = Path(coherence_utils.__file__).resolve()
        try:
            resolved_module.relative_to(coherent_root)
        except ValueError as exc:
            raise ImportError(
                "Loaded coherence helpers from an unexpected source: "
                f"{resolved_module}; expected a module below {coherent_root}"
            ) from exc
    return coherence_utils.transform, coherence_utils.coherence


def phase_rule_deadzone_sign(
    delta: torch.Tensor, theta_dead_rad: float
) -> torch.Tensor:
    return (delta > float(theta_dead_rad)).to(delta.dtype)


def resolve_phase_rule(
    phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor],
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    if callable(phase_rule):
        return phase_rule
    if phase_rule == "deadzone_sign":
        return phase_rule_deadzone_sign
    raise ValueError(f"Unsupported phase_rule: {phase_rule}")


def ordered_pair_indices(n_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def upper_pair_indices(n_channels: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]


def _extract_groups_from_metadata(
    metadata, group_column: str, n_samples: int
) -> np.ndarray | None:
    if metadata is None:
        return None
    if hasattr(metadata, "columns") and group_column in metadata.columns:
        groups = np.asarray(metadata[group_column].values)
    elif isinstance(metadata, dict) and group_column in metadata:
        groups = np.asarray(metadata[group_column])
    else:
        return None
    if groups.shape[0] != n_samples:
        raise ValueError(
            f"metadata['{group_column}'] has length {groups.shape[0]} "
            f"but expected {n_samples}."
        )
    return groups


def resolve_train_val_indices(
    n_samples: int,
    y_idx: np.ndarray,
    seed: int,
    validation_split,
    validation_group_column: str | None,
    validation_groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)

    if validation_split is None:
        return indices, np.array([], dtype=np.int64), None

    if isinstance(validation_split, (float, int)):
        frac = float(validation_split)
        if frac <= 0.0:
            return indices, np.array([], dtype=np.int64), None
        if frac >= 1.0:
            raise ValueError("validation_split as float must be in (0, 1).")

        if validation_group_column is None:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=frac, random_state=seed
            )
            train_idx, val_idx = next(splitter.split(indices, y_idx))
            return np.sort(train_idx.astype(np.int64)), np.sort(
                val_idx.astype(np.int64)
            ), None

        if validation_groups is None:
            raise ValueError(
                f"validation_group_column='{validation_group_column}' requires "
                "validation_groups or metadata with that column."
            )
        unique_groups = np.unique(validation_groups)
        if unique_groups.size < 2:
            raise ValueError("Grouped validation requires at least 2 groups.")
        n_val_groups = max(1, min(unique_groups.size - 1, round(frac * unique_groups.size)))
        chosen_groups = np.asarray(rng.permutation(unique_groups)[:n_val_groups])
        val_mask = np.isin(validation_groups, chosen_groups)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Validation split produced an empty partition.")
        return train_idx, val_idx, chosen_groups

    if validation_group_column is None:
        raise ValueError(
            "Explicit validation group selection requires validation_group_column."
        )
    if validation_groups is None:
        raise ValueError(
            f"validation_group_column='{validation_group_column}' requires groups."
        )

    requested_values = list(validation_split)
    mask = np.isin(validation_groups, requested_values)
    if not np.any(mask):
        mask = np.isin(
            validation_groups.astype(str),
            np.array([str(v) for v in requested_values], dtype=object),
        )
    val_idx = np.where(mask)[0]
    train_idx = np.where(~mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Validation group selection produced an empty partition.")
    return train_idx, val_idx, np.unique(validation_groups[val_idx])


def validation_groups_from_metadata(
    metadata,
    group_column: str | None,
    validation_groups: np.ndarray | None,
    n_samples: int,
) -> np.ndarray | None:
    if group_column is None:
        return None
    if validation_groups is not None:
        groups = np.asarray(validation_groups)
    else:
        groups = _extract_groups_from_metadata(metadata, group_column, n_samples)
    if groups is not None and groups.shape[0] != n_samples:
        raise ValueError(f"validation_groups length {groups.shape[0]} != {n_samples}.")
    return groups


def to_float_tensors(features) -> tuple[torch.Tensor, ...]:
    if isinstance(features, tuple):
        values = features
    else:
        values = (features,)
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value.float())
        else:
            tensors.append(torch.from_numpy(np.asarray(value)).float())
    return tuple(tensors)

def make_gaussian_weight2d(
    kernel_size=(9, 5),   # (T kernel, S kernel)
    sigma=(None, None),    # (T sigma, S sigma)
    pad_h=None,
    pad_w=None,
    *,
    device=None,
    dtype=None,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Returns:
        weight: [1, 1, H, W], ready for F.conv2d
        pad:    tuple for F.pad: (left, right, top, bottom)
    """
    # if pad_h is None then use "same" padding - so pad_h_total = kh - 1; same for pad_w
    # otherwise use the provided pad_h and pad_w

    kh, kw = kernel_size
    sh, sw = sigma
    if sh is None:
        sh = (kh - 1) / 2
    if sw is None:
        sw = (kw - 1) / 2

    h = torch.arange(kh, device=device, dtype=dtype) - (kh - 1) / 2
    w = torch.arange(kw, device=device, dtype=dtype) - (kw - 1) / 2

    hh, ww_grid = torch.meshgrid(h, w, indexing="ij")

    kernel = torch.exp(-0.5 * ((hh / sh) ** 2 + (ww_grid / sw) ** 2))
    kernel = kernel / kernel.sum()

    # conv2d weight shape
    weight = kernel[None, None, :, :]  # [1, 1, KH, KW]

    # F.pad order is: (left, right, top, bottom)
    pad_h_total = kh - 1 if pad_h is None else pad_h
    pad_w_total = kw - 1 if pad_w is None else pad_w

    pad_h_top = pad_h_total // 2
    pad_h_bottom = pad_h_total - pad_h_top

    pad_w_left = pad_w_total // 2
    pad_w_right = pad_w_total - pad_w_left

    pad = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)

    return weight, pad


def prepare_cwt_tf(
    coeffs: np.ndarray,
    nfreqs: int,
    n_time: int,
) -> np.ndarray:
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 2:
        raise ValueError(f"CWT coeffs must be 2D, got shape {coeffs.shape}.")
    coeffs_tf = coeffs.T if coeffs.shape[0] == nfreqs else coeffs
    if coeffs_tf.shape[0] != n_time:
        coeffs_tf = resample(coeffs_tf, n_time, axis=0)
    if coeffs_tf.shape[1] != nfreqs:
        raise ValueError(
            f"Unexpected CWT shape. Expected F={nfreqs}, got {coeffs_tf.shape}."
        )
    return np.nan_to_num(coeffs_tf, nan=0.0, posinf=0.0, neginf=0.0)


def compute_cwt_real_imag_tensors(
    X: np.ndarray,
    *,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: int | None,
    transform_fn,
    verbose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples, n_channels, n_time_orig = X.shape
    n_time = n_time_orig if cwt_resample_n_time is None else int(cwt_resample_n_time)
    if n_time <= 0:
        raise ValueError("cwt_resample_n_time must be a positive integer or None.")

    w_real = np.zeros((n_samples, n_channels, n_time, nfreqs), dtype=np.float32)
    w_imag = np.zeros((n_samples, n_channels, n_time, nfreqs), dtype=np.float32)

    _, freqs = transform_fn(
        X[0, 0, :],
        sampling_rate,
        highest,
        lowest,
        nfreqs=nfreqs,
    )

    freqs = torch.from_numpy(freqs).float().expand(n_samples, nfreqs)

    with tqdm(
        total=n_samples * n_channels,
        desc="CWT",
        disable=verbose < 1,
        leave=False,
    ) as pbar:
        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                coeffs, _ = transform_fn(
                    X[sample_idx, ch_idx, :],
                    sampling_rate,
                    highest,
                    lowest,
                    nfreqs=nfreqs,
                )
                coeffs_tf = prepare_cwt_tf(coeffs, nfreqs=nfreqs, n_time=n_time)
                w_real[sample_idx, ch_idx] = np.real(coeffs_tf).astype(np.float32)
                w_imag[sample_idx, ch_idx] = np.imag(coeffs_tf).astype(np.float32)
                pbar.update(1)

    raw_x = resample(X, n_time, axis=2) if n_time != n_time_orig else X
    raw_x = np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (
        torch.from_numpy(raw_x).float(),
        torch.from_numpy(w_real).float(),
        torch.from_numpy(w_imag).float(),
        freqs,
    )


def compute_paired_cwt_noise_bank(
    *,
    bank_size: int,
    segment_length: int,
    sampling_rate: int,
    highest: float,
    lowest: float,
    nfreqs: int,
    cwt_resample_n_time: int | None,
    transform_fn,
    seed: int,
    verbose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build matched raw/CWT noise entries from the same white-noise segments."""

    if bank_size <= 0:
        raise ValueError("noise_bank_size must be > 0.")
    if segment_length <= 0:
        raise ValueError("Noise segment length must be > 0.")

    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((bank_size, 1, segment_length)).astype(np.float32)
    raw_noise, cwt_real_noise, cwt_imag_noise, _ = compute_cwt_real_imag_tensors(
        noise,
        sampling_rate=sampling_rate,
        highest=highest,
        lowest=lowest,
        nfreqs=nfreqs,
        cwt_resample_n_time=cwt_resample_n_time,
        transform_fn=transform_fn,
        verbose=verbose,
    )
    return (
        raw_noise[:, 0, :].contiguous(),
        cwt_real_noise[:, 0, :, :].contiguous(),
        cwt_imag_noise[:, 0, :, :].contiguous(),
    )


def augment_paired_cwt_batch(
    batch_inputs: tuple[torch.Tensor, ...],
    *,
    noise_bank: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    channel_std: torch.Tensor,
    apply_prob: float,
    strength: float,
) -> tuple[torch.Tensor, ...]:
    """Apply paired raw/CWT noise to a CWT-GNN input triple."""

    if len(batch_inputs) != 3:
        return batch_inputs
    if apply_prob <= 0.0 or strength <= 0.0:
        return batch_inputs

    raw_x, w_real, w_imag = batch_inputs
    raw_noise_bank, real_noise_bank, imag_noise_bank = noise_bank
    batch_size, n_channels, _ = raw_x.shape
    bank_size = int(raw_noise_bank.shape[0])
    if bank_size <= 0:
        return batch_inputs

    device = raw_x.device
    bank_tensors = (raw_noise_bank, real_noise_bank, imag_noise_bank, channel_std)
    if any(tensor.device != device for tensor in bank_tensors):
        raise ValueError(
            "Noise augmentation tensors must already be on the batch device."
        )

    prob_mask = torch.rand(batch_size, n_channels, device=device) < float(apply_prob)
    if not bool(prob_mask.any().item()):
        return batch_inputs

    noise_idx = torch.randint(bank_size, (batch_size, n_channels), device=device)
    scale = (
        float(strength)
        * channel_std.view(1, n_channels)
        * prob_mask.to(dtype=raw_x.dtype)
    )

    raw_noise = raw_noise_bank[noise_idx]
    real_noise = real_noise_bank[noise_idx]
    imag_noise = imag_noise_bank[noise_idx]

    raw_aug = raw_x + scale.unsqueeze(-1) * raw_noise
    real_aug = (
        w_real
        + scale.to(dtype=w_real.dtype).unsqueeze(-1).unsqueeze(-1) * real_noise
    )
    imag_aug = (
        w_imag
        + scale.to(dtype=w_imag.dtype).unsqueeze(-1).unsqueeze(-1) * imag_noise
    )
    return raw_aug, real_aug, imag_aug


class TorchEEGClassifier(ClassifierMixin, BaseEstimator):
    """Reusable sklearn-compatible PyTorch classifier lifecycle."""

    def _init_torch_classifier(
        self,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        validation_split: float | Sequence | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int | None = 42,
        use_class_weights: bool = False,
        last_batch_min_ratio: float = 0.0,
        selector_alpha_val_update_rate: float = 1.0,
        optimizer_step_batch_size: int | None = None,
        optimizer_step_batch_mode: str = "credit",
        optimizer_step_remainder_policy: str = "flush",
        verbose: int = 0,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.validation_split = validation_split
        self.validation_group_column = validation_group_column
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.seed = seed
        self.use_class_weights = use_class_weights
        self.last_batch_min_ratio = last_batch_min_ratio
        self.selector_alpha_val_update_rate = selector_alpha_val_update_rate
        self.optimizer_step_batch_size = optimizer_step_batch_size
        self.optimizer_step_batch_mode = optimizer_step_batch_mode
        self.optimizer_step_remainder_policy = optimizer_step_remainder_policy
        self.verbose = verbose
        self._validate_batch_control_params()

        self.model_ = None
        self.classes_ = None
        self.class_to_idx_ = None
        self.device_ = None
        self._reset_histories()

    def _reset_histories(self) -> None:
        self.train_loss_history_ = []
        self.train_accuracy_history_ = []
        self.train_roc_auc_history_ = []
        self.val_loss_history_ = []
        self.val_accuracy_history_ = []
        self.val_roc_auc_history_ = []
        self.selector_train_history_ = []
        self.selector_val_history_ = []
        self.selector_alpha_val_history_ = []
        self.edge_density_history_ = []
        self.optimizer_step_count_history_ = []
        self.best_epoch_ = None
        self.best_val_loss_ = None

    def _vprint(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message, flush=True)

    def _validate_batch_control_params(self) -> None:
        last_batch_min_ratio = float(self.last_batch_min_ratio)
        if (
            not np.isfinite(last_batch_min_ratio)
            or last_batch_min_ratio < 0.0
            or last_batch_min_ratio > 1.0
        ):
            raise ValueError("last_batch_min_ratio must be in [0.0, 1.0].")

        selector_alpha_val_update_rate = float(self.selector_alpha_val_update_rate)
        if (
            not np.isfinite(selector_alpha_val_update_rate)
            or selector_alpha_val_update_rate < 0.0
        ):
            raise ValueError("selector_alpha_val_update_rate must be >= 0.0.")

        optimizer_step_batch_size = self.optimizer_step_batch_size
        if optimizer_step_batch_size is not None and (
            isinstance(optimizer_step_batch_size, bool)
            or not isinstance(optimizer_step_batch_size, (int, np.integer))
            or int(optimizer_step_batch_size) <= 0
        ):
            raise ValueError("optimizer_step_batch_size must be a positive integer or None.")
        if self.optimizer_step_batch_mode not in {"credit", "split"}:
            raise ValueError("optimizer_step_batch_mode must be 'credit' or 'split'.")
        if self.optimizer_step_remainder_policy not in {"flush", "drop", "carry"}:
            raise ValueError(
                "optimizer_step_remainder_policy must be 'flush', 'drop', or 'carry'."
            )
        if (
            self.optimizer_step_batch_mode == "credit"
            and optimizer_step_batch_size is not None
            and int(optimizer_step_batch_size) < int(self.batch_size)
        ):
            raise ValueError(
                "optimizer_step_batch_size must be at least batch_size in credit mode."
            )

        self.last_batch_min_ratio = last_batch_min_ratio
        self.selector_alpha_val_update_rate = selector_alpha_val_update_rate

    def _effective_optimizer_step_batch_size(self) -> int:
        if self.optimizer_step_batch_size is None:
            return int(self.batch_size)
        return int(self.optimizer_step_batch_size)

    def _log_selector_diagnostics(
        self,
        *,
        split: str,
        epoch: int,
        summaries: Sequence[dict[str, object]],
    ) -> None:
        if self.verbose < 2:
            return
        for summary in summaries:
            probs = summary.get("probs_mean", [])
            logits = summary.get("logits_mean", [])
            argmax = int(np.argmax(probs)) if probs else None
            parts = [
                f"[Selector][{split}][Epoch {epoch + 1}/{self.epochs}]",
                str(summary["name"]),
                f"mode={summary.get('mode')}",
                f"grad={summary.get('gradient_mode')}",
                f"alpha={summary.get('alpha_optim')}/{summary.get('alpha_update_split')}",
                f"temp={float(summary.get('temperature')):.4g}",
                f"entropy={float(summary.get('entropy_mean')):.4f}",
                f"argmax={argmax}",
                f"probs={_format_selector_values(probs)}",
                f"logits={_format_selector_values(logits)}",
            ]
            if summary.get("exploration_epsilon") is not None:
                parts.append(f"epsilon={summary.get('exploration_epsilon')}")
            if summary.get("expected_cost_mean") is not None:
                parts.append(
                    f"expected_cost={float(summary['expected_cost_mean']):.4f}"
                )
            if summary.get("selected_counts") is not None:
                parts.append(f"selected_counts={summary['selected_counts']}")
                parts.append(
                    "selected_fractions="
                    f"{_format_selector_values(summary['selected_fractions'])}"
                )
            self._vprint(2, " ".join(parts))

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        raise NotImplementedError

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _augment_train_batch_inputs(
        self, batch_inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return batch_inputs

    def _prepare_training_state_on_device(self) -> None:
        return None

    def _model_forward(self, batch_inputs: tuple[torch.Tensor, ...]):
        if self.model_ is not None and self.model_.training:
            batch_inputs = self._augment_train_batch_inputs(batch_inputs)
        output = self.model_(*batch_inputs)
        if isinstance(output, tuple):
            logits = output[0]
            aux = float(output[1]) if len(output) > 1 else None
        else:
            logits = output
            aux = None
        return logits, aux

    def _criterion(self, y_idx: np.ndarray) -> nn.Module:
        if not self.use_class_weights:
            return nn.CrossEntropyLoss()
        class_counts = np.bincount(y_idx, minlength=len(self.classes_))
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / (class_counts / class_counts.sum())
        weights = weights / weights.sum() * len(class_counts)
        weights_t = torch.from_numpy(weights.astype(np.float32)).to(self.device_)
        self._vprint(1, f"[Train] class weights: {weights_t.cpu().numpy()}")
        return nn.CrossEntropyLoss(weight=weights_t)

    def fit(self, X, y, validation_groups: np.ndarray | None = None, metadata=None):
        X = validate_eeg_X(X)
        self._validate_batch_control_params()
        set_seed(self.seed)
        self.device_ = resolve_torch_device(self.device)

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[cls] for cls in y], dtype=np.int64)
        n_classes = len(self.classes_)

        groups = validation_groups_from_metadata(
            metadata,
            self.validation_group_column,
            validation_groups,
            X.shape[0],
        )
        train_idx, val_idx, chosen_groups = resolve_train_val_indices(
            X.shape[0],
            y_idx,
            int(self.seed or 0),
            self.validation_split,
            self.validation_group_column,
            groups,
        )
        if val_idx.size == 0:
            self._vprint(1, "[Train] validation disabled.")
        elif chosen_groups is None:
            self._vprint(
                1,
                f"[Train] validation split: {val_idx.size}/{X.shape[0]} samples.",
            )
        else:
            self._vprint(
                1,
                f"[Train] validation groups ({self.validation_group_column}): "
                f"{chosen_groups.tolist()} -> {val_idx.size}/{X.shape[0]} samples.",
            )

        features = self._prepare_features(X, fit=True, train_idx=train_idx)
        tensors = to_float_tensors(features)
        y_tensor = torch.from_numpy(y_idx).long()

        train_loader = DataLoader(
            TensorDataset(*(t[train_idx] for t in tensors), y_tensor[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = None
        if val_idx.size > 0:
            val_loader = DataLoader(
                TensorDataset(*(t[val_idx] for t in tensors), y_tensor[val_idx]),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        self.model_ = self._build_model_from_features(features, n_classes, device=self.device_).to(
            self.device_
        )
        self._prepare_training_state_on_device()
        if self.verbose >= 2:
            model_label = getattr(self, "model_label", self.__class__.__name__)
            print_torch_parameter_summary(
                self.model_, header=model_label
            )
            print_torch_parameter_hashes(self.model_, header=model_label)
            print_torch_custom_model_summary(self.model_, header=model_label)

        optimizer, alpha_optimizer, selector_specs = self._build_training_optimizers()
        use_val_alpha_updates = (
            alpha_optimizer is not None
            and float(self.selector_alpha_val_update_rate) > 0.0
        )
        min_batch_size = _min_accepted_batch_size(
            int(self.batch_size),
            float(self.last_batch_min_ratio),
        )
        if _count_eligible_tensor_batches(train_loader, min_batch_size) == 0:
            raise ValueError(
                "last_batch_min_ratio leaves no eligible training batches. "
                "Reduce last_batch_min_ratio or batch_size."
            )
        if use_val_alpha_updates and val_loader is None:
            raise ValueError(
                "Validation-split selector alpha updates require a non-empty "
                "validation loader. Enable validation_split or use "
                "message_mlp_selector_mode='separate_train'."
            )
        if (
            use_val_alpha_updates
            and val_loader is not None
            and _count_eligible_tensor_batches(val_loader, min_batch_size) == 0
        ):
            raise ValueError(
                "Validation-split selector alpha updates require at least one "
                "eligible validation batch. Reduce last_batch_min_ratio or "
                "batch_size, or set selector_alpha_val_update_rate=0.0."
            )
        criterion = self._criterion(y_idx[train_idx])
        self._reset_histories()
        self._train_loop(
            train_loader,
            val_loader,
            optimizer,
            criterion,
            n_classes,
            selector_specs=selector_specs,
            alpha_optimizer=alpha_optimizer,
        )
        return self

    def _build_training_optimizers(self):
        if self.model_ is None:
            raise ValueError("Model has not been built yet.")

        selector_specs = _collect_selector_training_specs(self.model_)
        separate_train_params = _selector_params(
            selector_specs,
            alpha_optim="separate",
            alpha_update_split="train",
        )
        separate_val_params = _selector_params(
            selector_specs,
            alpha_optim="separate",
            alpha_update_split="val",
        )
        separate_param_ids = {
            id(param) for param in (*separate_train_params, *separate_val_params)
        }
        regular_params = [
            param
            for param in self.model_.parameters()
            if param.requires_grad and id(param) not in separate_param_ids
        ]

        param_groups = []
        if regular_params:
            param_groups.append(
                {"params": regular_params, "weight_decay": self.weight_decay}
            )
        if separate_train_params:
            param_groups.append(
                {"params": list(separate_train_params), "weight_decay": 0.0}
            )
        if not param_groups:
            raise ValueError("No trainable parameters were found.")

        optimizer = optim.AdamW(param_groups, lr=self.learning_rate)
        alpha_optimizer = None
        if separate_val_params:
            alpha_optimizer = optim.AdamW(
                [{"params": list(separate_val_params), "weight_decay": 0.0}],
                lr=self.learning_rate,
            )
        return optimizer, alpha_optimizer, selector_specs

    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer,
        criterion,
        n_classes: int,
        *,
        selector_specs: Sequence[_SelectorTrainingSpec] | None = None,
        alpha_optimizer=None,
    ) -> None:
        selector_specs = [] if selector_specs is None else list(selector_specs)
        optimizer_step_batch_size = self._effective_optimizer_step_batch_size()
        optimizer_step_batch_mode = self.optimizer_step_batch_mode
        remainder_policy = self.optimizer_step_remainder_policy
        use_val_alpha_updates = (
            alpha_optimizer is not None
            and float(self.selector_alpha_val_update_rate) > 0.0
        )
        if use_val_alpha_updates and val_loader is None:
            raise ValueError(
                "Validation-split selector alpha updates require a validation loader."
            )
        min_batch_size = _min_accepted_batch_size(
            int(self.batch_size),
            float(self.last_batch_min_ratio),
        )
        eligible_train_batches = _count_eligible_tensor_batches(
            train_loader, min_batch_size
        )
        train_selector_specs = _filter_selector_specs(
            selector_specs,
            alpha_update_split="train",
        )
        val_alpha_specs = _filter_selector_specs(
            selector_specs,
            alpha_optim="separate",
            alpha_update_split="val",
        )
        val_alpha_params = _selector_params(
            val_alpha_specs,
            alpha_optim="separate",
            alpha_update_split="val",
        )
        val_alpha_gates = _selector_gate_modules(
            val_alpha_specs,
            alpha_optim="separate",
            alpha_update_split="val",
        )
        alpha_val_iter = iter(val_loader) if use_val_alpha_updates else None
        train_batch_suffix = ""
        if eligible_train_batches < len(train_loader):
            train_batch_suffix = (
                f" ({len(train_loader) - eligible_train_batches} loader batch(s) "
                f"skipped by last_batch_min_ratio)"
            )
        self._vprint(
            1,
            f"[Train] start epochs={self.epochs} "
            f"batches/epoch={eligible_train_batches}{train_batch_suffix} "
            f"batch_size={self.batch_size} "
            f"optimizer_step_batch_size={optimizer_step_batch_size} "
            f"optimizer_step_batch_mode={optimizer_step_batch_mode} "
            f"device={self.device_}",
        )
        prev_loss = None
        prev_acc = None
        prev_auc = None
        best_state = None
        best_epoch = -1
        best_val_loss = float("inf")
        no_improve_epochs = 0
        alpha_val_update_credit = 0.0
        pending_gradient_samples = 0
        optimizer_sample_credit = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            self.model_.train()
            loss_sum = 0.0
            aux_sum = 0.0
            aux_count = 0
            n_batches = 0
            n_correct = 0
            n_seen = 0
            epoch_targets = []
            epoch_probas = []
            train_selector_accumulators = _new_selector_diagnostic_accumulators(
                selector_specs
            )
            alpha_val_selector_accumulators = _new_selector_diagnostic_accumulators(
                val_alpha_specs
            )
            optimizer_steps = 0

            def take_optimizer_step() -> None:
                nonlocal alpha_val_iter
                nonlocal alpha_val_update_credit
                nonlocal optimizer_steps
                nonlocal pending_gradient_samples

                if pending_gradient_samples <= 0:
                    return
                _scale_optimizer_gradients(
                    optimizer,
                    1.0 / float(pending_gradient_samples),
                )
                if self.grad_clip_norm is not None and float(self.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        _optimizer_parameters(optimizer),
                        max_norm=float(self.grad_clip_norm),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_gradient_samples = 0
                optimizer_steps += 1

                if use_val_alpha_updates:
                    if val_loader is None or alpha_val_iter is None:
                        raise RuntimeError("alpha validation loader is not initialized.")
                    alpha_val_update_credit += float(
                        self.selector_alpha_val_update_rate
                    )
                    while alpha_val_update_credit + 1e-12 >= 1.0:
                        alpha_val_iter = self._update_selector_alpha_from_validation(
                            val_loader=val_loader,
                            val_iter=alpha_val_iter,
                            alpha_optimizer=alpha_optimizer,
                            criterion=criterion,
                            selector_specs=val_alpha_specs,
                            alpha_params=val_alpha_params,
                            gate_modules=val_alpha_gates,
                            diagnostic_accumulators=alpha_val_selector_accumulators,
                            min_batch_size=min_batch_size,
                        )
                        alpha_val_update_credit -= 1.0
                        if abs(alpha_val_update_credit) < 1e-12:
                            alpha_val_update_credit = 0.0

            for batch in train_loader:
                if not _batch_meets_min_size(batch, min_batch_size):
                    continue
                batch = tuple(x.to(self.device_) for x in batch)
                batch_size = _batch_sample_count(batch)

                step_start = time.perf_counter()
                weighted_loss = 0.0
                weighted_aux = 0.0
                aux_samples = 0
                logits_parts = []
                batch_y_parts = []
                start = 0
                while start < batch_size:
                    if optimizer_step_batch_mode == "split":
                        slice_size = min(
                            optimizer_step_batch_size - pending_gradient_samples,
                            batch_size - start,
                        )
                    else:
                        slice_size = batch_size
                    batch_slice = _slice_tensor_batch(batch, start, start + slice_size)
                    *batch_inputs, batch_y = batch_slice
                    with _temporarily_requires_grad(val_alpha_params, False):
                        logits, aux_value = self._model_forward(tuple(batch_inputs))
                        _record_selector_diagnostics(
                            train_selector_accumulators,
                            selector_specs,
                        )
                        loss = criterion(logits, batch_y)
                        # Train batches regularize only selectors updated on train.
                        loss = loss + _selector_extra_loss(train_selector_specs, loss)
                    (loss * slice_size).backward()
                    pending_gradient_samples += slice_size
                    weighted_loss += float(loss.item()) * slice_size
                    logits_parts.append(logits.detach())
                    batch_y_parts.append(batch_y.detach())
                    if aux_value is not None:
                        weighted_aux += aux_value * slice_size
                        aux_samples += slice_size

                    if optimizer_step_batch_mode == "split":
                        if pending_gradient_samples == optimizer_step_batch_size:
                            take_optimizer_step()
                    start += slice_size
                    if optimizer_step_batch_mode == "credit":
                        break

                if optimizer_step_batch_mode == "credit":
                    optimizer_sample_credit += batch_size
                    if optimizer_sample_credit >= optimizer_step_batch_size:
                        # Preserve excess credit for later batches; never replay one
                        # batch's gradients to manufacture multiple optimizer steps.
                        take_optimizer_step()
                        optimizer_sample_credit -= optimizer_step_batch_size

                technical_loss = weighted_loss / batch_size
                logits = torch.cat(logits_parts, dim=0)
                batch_y = torch.cat(batch_y_parts, dim=0)
                loss_sum += technical_loss
                n_batches += 1
                if aux_samples:
                    aux_sum += weighted_aux / aux_samples
                    aux_count += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_proba_np = torch.softmax(logits.detach(), dim=1).cpu().numpy()
                epoch_targets.append(batch_y_np)
                epoch_probas.append(batch_proba_np)

                if self.verbose >= 3:
                    running_loss = loss_sum / n_batches
                    running_acc = n_correct / max(1, n_seen)
                    running_auc = safe_roc_auc(
                        np.concatenate(epoch_targets, axis=0),
                        np.concatenate(epoch_probas, axis=0),
                        n_classes,
                    )
                    rate = batch_size / max(time.perf_counter() - step_start, 1e-6)
                    self._vprint(
                        2,
                        f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                        f"step={n_batches}/{eligible_train_batches} "
                        f"loss={technical_loss:.6f} "
                        f"running_loss={running_loss:.6f} "
                        f"running_acc={running_acc:.4f} "
                        f"running_roc_auc={fmt_metric(running_auc)} "
                        f"rate={rate:.2f} samples/s",
                    )

            if remainder_policy == "flush":
                take_optimizer_step()
                optimizer_sample_credit = 0
            elif remainder_policy == "drop":
                optimizer.zero_grad(set_to_none=True)
                pending_gradient_samples = 0
                optimizer_sample_credit = 0
            elif epoch == self.epochs - 1:
                # Carry is only meaningful when another epoch can consume it.
                optimizer.zero_grad(set_to_none=True)
                pending_gradient_samples = 0
                optimizer_sample_credit = 0

            if n_batches == 0:
                raise RuntimeError(
                    "No eligible training batches were processed. Reduce "
                    "last_batch_min_ratio or batch_size."
                )
            avg_loss = loss_sum / max(1, n_batches)
            avg_acc = n_correct / max(1, n_seen)
            avg_auc = safe_roc_auc(
                np.concatenate(epoch_targets, axis=0),
                np.concatenate(epoch_probas, axis=0),
                n_classes,
            )
            avg_aux = aux_sum / aux_count if aux_count else None
            if avg_aux is not None:
                self.edge_density_history_.append(avg_aux)
            self.train_loss_history_.append(avg_loss)
            self.train_accuracy_history_.append(avg_acc)
            self.train_roc_auc_history_.append(avg_auc)
            self.optimizer_step_count_history_.append(optimizer_steps)
            train_selector_summary = _finalize_selector_diagnostics(
                train_selector_accumulators
            )
            if selector_specs:
                self.selector_train_history_.append(train_selector_summary)
                self._log_selector_diagnostics(
                    split="train",
                    epoch=epoch,
                    summaries=train_selector_summary,
                )
            alpha_val_selector_summary = _finalize_selector_diagnostics(
                alpha_val_selector_accumulators
            )
            if use_val_alpha_updates:
                self.selector_alpha_val_history_.append(alpha_val_selector_summary)
                self._log_selector_diagnostics(
                    split="alpha_val",
                    epoch=epoch,
                    summaries=alpha_val_selector_summary,
                )

            loss_delta = 0.0 if prev_loss is None else (prev_loss - avg_loss)
            acc_delta = 0.0 if prev_acc is None else (avg_acc - prev_acc)
            auc_delta = None if prev_auc is None or avg_auc is None else avg_auc - prev_auc
            prev_loss = avg_loss
            prev_acc = avg_acc
            prev_auc = avg_auc

            val_suffix = ""
            if val_loader is not None:
                val_loss, val_acc, val_auc, val_selector_summary = self._evaluate_loader(
                    val_loader,
                    criterion,
                    n_classes,
                    selector_specs=selector_specs,
                    return_selector_summary=True,
                )
                self.val_loss_history_.append(val_loss)
                self.val_accuracy_history_.append(val_acc)
                self.val_roc_auc_history_.append(val_auc)
                if selector_specs:
                    self.selector_val_history_.append(val_selector_summary)
                    self._log_selector_diagnostics(
                        split="val",
                        epoch=epoch,
                        summaries=val_selector_summary,
                    )
                val_suffix = (
                    f" val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                    f"val_roc_auc={fmt_metric(val_auc)}"
                )
                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state = deepcopy(self.model_.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            epoch_time = time.perf_counter() - epoch_start
            aux_suffix = "" if avg_aux is None else f" edge_density={avg_aux:.6f}"
            self._vprint(
                1,
                f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                f"loss={avg_loss:.6f} (improve {loss_delta:+.6f}) "
                f"acc={avg_acc:.4f} (delta {acc_delta:+.4f}) "
                f"roc_auc={fmt_metric(avg_auc)} "
                f"(delta {fmt_metric(auc_delta) if auc_delta is not None else 'n/a'})"
                f" optimizer_steps={optimizer_steps}{aux_suffix} "
                f"epoch_time={epoch_time:.2f}s{val_suffix}",
            )

            if (
                val_loader is not None
                and self.early_stopping_patience is not None
                and self.early_stopping_patience >= 0
                and no_improve_epochs >= self.early_stopping_patience
            ):
                self._vprint(
                    1,
                    f"[Train] early stopping at epoch {epoch + 1}; "
                    f"best epoch={best_epoch} best_val_loss={best_val_loss:.6f}",
                )
                break

        # An early-stopped carry policy also has no subsequent epoch to consume it.
        if pending_gradient_samples > 0:
            optimizer.zero_grad(set_to_none=True)

        if val_loader is not None and best_state is not None:
            self.model_.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
            self.best_val_loss_ = best_val_loss
            self._vprint(
                1,
                f"[Train] restored best model from epoch {best_epoch} "
                f"(val_loss={best_val_loss:.6f})",
            )

    def _update_selector_alpha_from_validation(
        self,
        *,
        val_loader: DataLoader,
        val_iter,
        alpha_optimizer,
        criterion,
        selector_specs: Sequence[_SelectorTrainingSpec],
        alpha_params: Sequence[nn.Parameter],
        gate_modules: Sequence[nn.Module],
        diagnostic_accumulators: dict[int, dict[str, object]] | None = None,
        min_batch_size: int = 1,
    ):
        max_attempts = len(val_loader)
        if max_attempts <= 0:
            raise RuntimeError("alpha validation loader is empty.")
        for _ in range(max_attempts):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            if _batch_meets_min_size(batch, min_batch_size):
                break
        else:
            raise RuntimeError(
                "No validation-alpha batch meets last_batch_min_ratio."
            )

        *batch_inputs, batch_y = batch
        batch_inputs = tuple(x.to(self.device_) for x in batch_inputs)
        batch_y = batch_y.to(self.device_)

        alpha_optimizer.zero_grad(set_to_none=True)
        with _selector_alpha_update_scope(self.model_, alpha_params, gate_modules):
            logits, _ = self._model_forward(batch_inputs)
            if diagnostic_accumulators is not None:
                _record_selector_diagnostics(diagnostic_accumulators, selector_specs)
            loss = criterion(logits, batch_y)
            # Validation-alpha batches regularize only validation-updated selectors.
            loss = loss + _selector_extra_loss(selector_specs, loss)
            loss.backward()
        alpha_optimizer.step()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_param_ids = {id(param) for param in alpha_params}
        for param in self.model_.parameters():
            if id(param) not in alpha_param_ids:
                param.grad = None

        return val_iter

    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        n_classes: int,
        *,
        selector_specs: Sequence[_SelectorTrainingSpec] | None = None,
        return_selector_summary: bool = False,
    ) -> tuple[float, float, float | None] | tuple[
        float,
        float,
        float | None,
        list[dict[str, object]],
    ]:
        selector_specs = [] if selector_specs is None else list(selector_specs)
        selector_accumulators = _new_selector_diagnostic_accumulators(selector_specs)
        self.model_.eval()
        loss_sum = 0.0
        n_batches = 0
        n_correct = 0
        n_seen = 0
        y_all = []
        p_all = []
        with torch.no_grad():
            for batch in loader:
                *batch_inputs, batch_y = batch
                batch_inputs = tuple(x.to(self.device_) for x in batch_inputs)
                batch_y = batch_y.to(self.device_)
                logits, _ = self._model_forward(batch_inputs)
                _record_selector_diagnostics(selector_accumulators, selector_specs)
                loss = criterion(logits, batch_y)
                # Validation reports the full regularized objective.
                loss = loss + _selector_extra_loss(selector_specs, loss)
                loss_sum += float(loss.item())
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                y_all.append(batch_y.detach().cpu().numpy())
                p_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

        result = (
            loss_sum / max(1, n_batches),
            n_correct / max(1, n_seen),
            safe_roc_auc(np.concatenate(y_all), np.concatenate(p_all), n_classes),
        )
        if return_selector_summary:
            return (*result, _finalize_selector_diagnostics(selector_accumulators))
        return result

    def _predict_logits(self, X) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise ValueError("Model has not been fitted yet.")
        X = validate_eeg_X(X)
        tensors = to_float_tensors(self._prepare_features(X, fit=False))
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        logits_list = []
        self.model_.eval()
        with torch.no_grad():
            for batch_inputs in loader:
                batch_inputs = tuple(x.to(self.device_) for x in batch_inputs)
                logits, _ = self._model_forward(batch_inputs)
                logits_list.append(logits.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    def predict(self, X) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.classes_[np.argmax(self._predict_logits(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self._predict_logits(X)), dim=1).numpy()

    def score(self, X, y) -> float:
        return float(accuracy_score(y, self.predict(X)))
