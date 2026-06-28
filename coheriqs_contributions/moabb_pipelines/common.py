"""Shared utilities for local MOABB-compatible classifiers."""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
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
    precision: float = 1e-6,
) -> tuple[dict[str, str], str]:
    """Return tolerance-aware hashes for model parameters and the full model."""

    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be a finite positive number.")

    parameter_hashes = {}
    model_hasher = hashlib.blake2b(digest_size=16)
    for name, param in model.named_parameters():
        param_hasher = hashlib.blake2b(digest_size=16)
        metadata = f"{name}|{tuple(param.shape)}|{param.dtype}|{precision:.17g}"
        param_hasher.update(metadata.encode("utf-8"))
        param_hasher.update(_quantized_parameter_bytes(param, precision))
        digest = param_hasher.hexdigest()
        parameter_hashes[name] = digest
        model_hasher.update(name.encode("utf-8"))
        model_hasher.update(b"\0")
        model_hasher.update(bytes.fromhex(digest))

    return parameter_hashes, model_hasher.hexdigest()


def print_torch_parameter_hashes(
    model: nn.Module,
    header: str = "Model",
    *,
    precision: float = 1e-6,
) -> None:
    """Print reproducible per-parameter and global model fingerprints."""

    parameter_hashes, model_hash = torch_parameter_hashes(
        model,
        precision=precision,
    )
    print(
        f"[{header}] Parameter hashes algorithm=blake2b-128 "
        f"precision={precision:.1e}",
        flush=True,
    )
    for name, digest in parameter_hashes.items():
        print(f"[{header}] {name}: hash={digest}", flush=True)
    print(f"[{header}] model_hash={model_hash}", flush=True)


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
    coherent_root = repo_root / "Coherent_Multiplex"
    if coherent_root.exists():
        coherent_root_str = str(coherent_root)
        if coherent_root_str not in sys.path:
            sys.path.insert(0, coherent_root_str)
    try:
        from utils.coherence_utils import coherence, transform  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import Coherent_Multiplex wavelet helpers. "
            "Ensure Coherent_Multiplex is present and dependencies are installed."
        ) from exc
    return transform, coherence


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
        self.verbose = verbose

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
        self.edge_density_history_ = []
        self.best_epoch_ = None
        self.best_val_loss_ = None

    def _vprint(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message, flush=True)

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

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = self._criterion(y_idx[train_idx])
        self._reset_histories()
        self._train_loop(train_loader, val_loader, optimizer, criterion, n_classes)
        return self

    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer,
        criterion,
        n_classes: int,
    ) -> None:
        self._vprint(
            1,
            f"[Train] start epochs={self.epochs} batches/epoch={len(train_loader)} "
            f"batch_size={self.batch_size} device={self.device_}",
        )
        prev_loss = None
        prev_acc = None
        prev_auc = None
        best_state = None
        best_epoch = -1
        best_val_loss = float("inf")
        no_improve_epochs = 0

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

            for step_idx, batch in enumerate(train_loader, start=1):
                *batch_inputs, batch_y = batch
                batch_inputs = tuple(x.to(self.device_) for x in batch_inputs)
                batch_y = batch_y.to(self.device_)

                step_start = time.perf_counter()
                optimizer.zero_grad()
                logits, aux_value = self._model_forward(batch_inputs)
                loss = criterion(logits, batch_y)
                loss.backward()
                if self.grad_clip_norm is not None and float(self.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(), max_norm=float(self.grad_clip_norm)
                    )
                optimizer.step()

                loss_sum += float(loss.item())
                n_batches += 1
                if aux_value is not None:
                    aux_sum += aux_value
                    aux_count += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_proba_np = torch.softmax(logits.detach(), dim=1).cpu().numpy()
                epoch_targets.append(batch_y_np)
                epoch_probas.append(batch_proba_np)

                if self.verbose >= 2:
                    running_loss = loss_sum / n_batches
                    running_acc = n_correct / max(1, n_seen)
                    running_auc = safe_roc_auc(
                        np.concatenate(epoch_targets, axis=0),
                        np.concatenate(epoch_probas, axis=0),
                        n_classes,
                    )
                    rate = batch_y.numel() / max(time.perf_counter() - step_start, 1e-6)
                    self._vprint(
                        2,
                        f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                        f"step={step_idx}/{len(train_loader)} "
                        f"loss={float(loss.item()):.6f} "
                        f"running_loss={running_loss:.6f} "
                        f"running_acc={running_acc:.4f} "
                        f"running_roc_auc={fmt_metric(running_auc)} "
                        f"rate={rate:.2f} samples/s",
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

            loss_delta = 0.0 if prev_loss is None else (prev_loss - avg_loss)
            acc_delta = 0.0 if prev_acc is None else (avg_acc - prev_acc)
            auc_delta = None if prev_auc is None or avg_auc is None else avg_auc - prev_auc
            prev_loss = avg_loss
            prev_acc = avg_acc
            prev_auc = avg_auc

            val_suffix = ""
            if val_loader is not None:
                val_loss, val_acc, val_auc = self._evaluate_loader(
                    val_loader, criterion, n_classes
                )
                self.val_loss_history_.append(val_loss)
                self.val_accuracy_history_.append(val_acc)
                self.val_roc_auc_history_.append(val_auc)
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
                f"{aux_suffix} epoch_time={epoch_time:.2f}s{val_suffix}",
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

        if val_loader is not None and best_state is not None:
            self.model_.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
            self.best_val_loss_ = best_val_loss
            self._vprint(
                1,
                f"[Train] restored best model from epoch {best_epoch} "
                f"(val_loss={best_val_loss:.6f})",
            )

    def _evaluate_loader(
        self, loader: DataLoader, criterion: nn.Module, n_classes: int
    ) -> tuple[float, float, float | None]:
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
                loss = criterion(logits, batch_y)
                loss_sum += float(loss.item())
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                y_all.append(batch_y.detach().cpu().numpy())
                p_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

        return (
            loss_sum / max(1, n_batches),
            n_correct / max(1, n_seen),
            safe_roc_auc(np.concatenate(y_all), np.concatenate(p_all), n_classes),
        )

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
