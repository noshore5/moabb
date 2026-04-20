"""Level-0 XWT phase-conditioned GNN classifier.

This module contains:
- a torch core model that performs message passing over phase-gated XWT edges
- an sklearn-compatible wrapper for MOABB pipeline integration
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

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


log = logging.getLogger(__name__)


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float | None:
    """Compute ROC-AUC safely; return None when undefined for current labels."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    try:
        if n_classes <= 2:
            if proba.ndim == 2:
                score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                score = proba
            return float(roc_auc_score(y_true, score))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except ValueError:
        return None


def _fmt_metric(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{ndigits}f}"


def _fit_global_zscore_stats(X: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(X))
    std = float(np.std(X))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def _apply_global_zscore(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (X - mean) / (std + 1e-8)


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
            f"metadata['{group_column}'] has length {groups.shape[0]} but expected {n_samples}."
        )
    return groups


def _resolve_train_val_indices(
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
            train_idx = np.sort(train_idx.astype(np.int64))
            val_idx = np.sort(val_idx.astype(np.int64))
            if train_idx.size == 0 or val_idx.size == 0:
                raise ValueError(
                    "Validation split produced empty train or validation set."
                )
            return train_idx, val_idx, None

        if validation_groups is None:
            raise ValueError(
                f"validation_group_column='{validation_group_column}' requires "
                "validation_groups (or metadata with that column) in fit()."
            )
        if validation_groups.shape[0] != n_samples:
            raise ValueError(
                f"validation_groups length {validation_groups.shape[0]} != {n_samples}."
            )
        unique_groups = np.unique(validation_groups)
        if unique_groups.size < 2:
            raise ValueError(
                "Grouped validation split requires at least 2 unique group values."
            )
        n_val_groups = int(round(frac * unique_groups.size))
        n_val_groups = max(1, min(unique_groups.size - 1, n_val_groups))
        chosen_groups = np.asarray(rng.permutation(unique_groups)[:n_val_groups])
        val_mask = np.isin(validation_groups, chosen_groups)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError(
                "Grouped validation split produced empty train or validation set."
            )
        return train_idx, val_idx, chosen_groups

    if validation_group_column is None:
        raise ValueError(
            "validation_split as list/set/tuple requires validation_group_column to be set."
        )
    if validation_groups is None:
        raise ValueError(
            f"validation_group_column='{validation_group_column}' requires "
            "validation_groups (or metadata with that column) in fit()."
        )
    if validation_groups.shape[0] != n_samples:
        raise ValueError(
            f"validation_groups length {validation_groups.shape[0]} != {n_samples}."
        )

    requested_values = list(validation_split)
    mask = np.isin(validation_groups, requested_values)
    if not np.any(mask):
        as_str = np.array([str(v) for v in requested_values], dtype=object)
        mask = np.isin(validation_groups.astype(str), as_str)
    val_idx = np.where(mask)[0]
    train_idx = np.where(~mask)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError(
            "Explicit validation group selection produced empty train or validation set."
        )
    chosen_groups = np.unique(validation_groups[val_idx]) if val_idx.size > 0 else np.array([], dtype=object)
    return train_idx, val_idx, chosen_groups


def _evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    n_classes: int,
) -> tuple[float, float, float | None]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    n_correct = 0
    n_seen = 0
    y_all = []
    p_all = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                batch_x, batch_wr, batch_wi, batch_y = batch
            else:
                raise ValueError("Validation loader must yield 4 tensors.")
            batch_x = batch_x.to(device)
            batch_wr = batch_wr.to(device)
            batch_wi = batch_wi.to(device)
            batch_y = batch_y.to(device)

            logits, _ = model(batch_x, batch_wr, batch_wi)
            loss = criterion(logits, batch_y)
            loss_sum += float(loss.item())
            n_batches += 1

            preds = torch.argmax(logits, dim=1)
            n_correct += int((preds == batch_y).sum().item())
            n_seen += int(batch_y.numel())

            y_all.append(batch_y.detach().cpu().numpy())
            p_all.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    avg_loss = loss_sum / max(1, n_batches)
    avg_acc = n_correct / max(1, n_seen)
    avg_auc = _safe_roc_auc(
        np.concatenate(y_all, axis=0),
        np.concatenate(p_all, axis=0),
        n_classes,
    )
    return avg_loss, avg_acc, avg_auc


def _print_torch_parameter_summary(model: nn.Module, header: str = "Model") -> None:
    """Print a parameter summary with shapes and counts."""
    print(f"[{header}] Parameter summary", flush=True)
    total_params = 0
    total_trainable = 0
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        n_params = int(param.numel())
        total_params += n_params
        if param.requires_grad:
            total_trainable += n_params
        print(
            f"[{header}] {name}: shape={shape} params={n_params} "
            f"trainable={param.requires_grad}",
            flush=True,
        )
    print(
        f"[{header}] total_params={total_params} "
        f"trainable_params={total_trainable} "
        f"non_trainable_params={total_params - total_trainable}",
        flush=True,
    )


def _phase_rule_deadzone_sign(
    delta: torch.Tensor, theta_dead_rad: float
) -> torch.Tensor:
    """Return edge gate for ordered pairs using a circular-aware dead zone.

    For ordered edge (i -> j), gate is active only when delta(i, j) > theta.
    The reverse direction is handled by the reverse ordered pair.
    """
    theta = float(theta_dead_rad)
    return (delta > theta).to(delta.dtype)


def _resolve_phase_rule(
    phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor],
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    if callable(phase_rule):
        return phase_rule
    if phase_rule == "deadzone_sign":
        return _phase_rule_deadzone_sign
    raise ValueError(f"Unsupported phase_rule: {phase_rule}")


def _ordered_pair_indices(n_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_transform_callable():
    """Load Coherent_Multiplex wavelet transform helper."""
    repo_root = Path(__file__).resolve().parents[2]
    coherent_root = repo_root / "Coherent_Multiplex"
    if coherent_root.exists():
        coherent_root_str = str(coherent_root)
        if coherent_root_str not in sys.path:
            sys.path.insert(0, coherent_root_str)

    try:
        from utils.coherence_utils import transform  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import Coherent_Multiplex wavelet transform. "
            "Ensure Coherent_Multiplex is present and dependencies are installed."
        ) from exc
    return transform


class XWTPhaseGNNCore(nn.Module):
    """Torch core for level-0 phase-gated XWT message passing."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        theta_dead_deg: float = 45.0,
        time_stride: int = 1,
        state_mode: str = "per_node",
        phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor] = "deadzone_sign",
        use_mag: bool = True,
        use_ang: bool = True,
        use_raw: bool = True,
        use_state_src: bool = True,
        use_state_dst: bool = True,
        readout_mode: str = "trial",
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        if state_mode not in {"per_node", "per_node_per_freq"}:
            raise ValueError("state_mode must be one of {'per_node', 'per_node_per_freq'}")
        if readout_mode != "trial":
            raise ValueError("Only readout_mode='trial' is implemented for level 0")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.theta_dead_rad = math.radians(theta_dead_deg)
        self.time_stride = time_stride
        self.state_mode = state_mode
        self.phase_rule_fn = _resolve_phase_rule(phase_rule)
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_state_src = use_state_src
        self.use_state_dst = use_state_dst
        self.readout_mode = readout_mode

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        payload_dim = 0
        if self.use_mag:
            payload_dim += 1
        if self.use_ang:
            payload_dim += 1
        if self.use_raw:
            payload_dim += 2
        if self.use_state_src:
            payload_dim += hidden_dim
        if self.use_state_dst:
            payload_dim += hidden_dim
        if payload_dim == 0:
            raise ValueError("At least one payload component must be enabled.")

        self.message_mlp = nn.Sequential(
            nn.Linear(payload_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, hidden_dim),
        )
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def _aggregate_per_node(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, H] messages to [B, C, H] by destination."""
        batch_size, num_edges, hidden_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels, hidden_dim, device=device, dtype=msg.dtype
        )
        batch_offsets = (
            torch.arange(batch_size, device=device).unsqueeze(1) * self.n_channels
        )
        dst = (self.dst_idx.unsqueeze(0) + batch_offsets).reshape(-1)
        agg.index_add_(0, dst, msg.reshape(batch_size * num_edges, hidden_dim))
        return agg.view(batch_size, self.n_channels, hidden_dim)

    def _aggregate_per_node_per_freq(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, F, H] messages to [B, C, F, H] by destination."""
        batch_size, num_edges, nfreqs, hidden_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels * nfreqs,
            hidden_dim,
            device=device,
            dtype=msg.dtype,
        )

        base_b = (
            torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            * self.n_channels
            * nfreqs
        )
        base_f = torch.arange(nfreqs, device=device).view(1, 1, nfreqs)
        dst = self.dst_idx.view(1, num_edges, 1) * nfreqs + base_f + base_b
        agg.index_add_(0, dst.reshape(-1), msg.reshape(batch_size * num_edges * nfreqs, hidden_dim))
        return agg.view(batch_size, self.n_channels, nfreqs, hidden_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Forward pass.

        Parameters
        ----------
        raw_x : tensor, shape (B, C, T)
        w_real : tensor, shape (B, C, T, F)
        w_imag : tensor, shape (B, C, T, F)
        """
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}."
            )

        device = raw_x.device
        num_edges = self.src_idx.numel()

        if self.state_mode == "per_node":
            state = torch.zeros(
                batch_size, self.n_channels, self.hidden_dim, device=device
            )
        else:
            state = torch.zeros(
                batch_size,
                self.n_channels,
                self.nfreqs,
                self.hidden_dim,
                device=device,
            )

        gate_sum = 0.0
        gate_count = 0.0

        for t in range(0, n_time, self.time_stride):
            src_r = w_real[:, self.src_idx, t, :]
            src_i = w_imag[:, self.src_idx, t, :]
            dst_r = w_real[:, self.dst_idx, t, :]
            dst_i = w_imag[:, self.dst_idx, t, :]

            # (a + ib) * conj(c + id) = (ac + bd) + i(bc - ad)
            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i

            mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            ang = torch.atan2(xwt_imag, xwt_real)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))
            gate = self.phase_rule_fn(delta, self.theta_dead_rad)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)

            gate_sum += float(gate.sum().item())
            gate_count += float(gate.numel())

            mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)

            features = []
            if self.use_mag:
                features.append(mag.unsqueeze(-1))
            if self.use_ang:
                features.append(ang.unsqueeze(-1))
            if self.use_raw:
                raw_t = raw_x[:, :, t]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                src_raw = src_raw.expand(batch_size, num_edges, self.nfreqs, 1)
                dst_raw = dst_raw.expand(batch_size, num_edges, self.nfreqs, 1)
                features.append(src_raw)
                features.append(dst_raw)

            if self.state_mode == "per_node":
                if self.use_state_src:
                    src_state = state[:, self.src_idx, :].unsqueeze(2)
                    src_state = src_state.expand(
                        batch_size, num_edges, self.nfreqs, self.hidden_dim
                    )
                    features.append(src_state)
                if self.use_state_dst:
                    dst_state = state[:, self.dst_idx, :].unsqueeze(2)
                    dst_state = dst_state.expand(
                        batch_size, num_edges, self.nfreqs, self.hidden_dim
                    )
                    features.append(dst_state)
            else:
                if self.use_state_src:
                    features.append(state[:, self.src_idx, :, :])
                if self.use_state_dst:
                    features.append(state[:, self.dst_idx, :, :])

            payload = torch.cat(features, dim=-1)
            msg = self.message_mlp(payload)
            msg = msg * gate.unsqueeze(-1)

            if self.state_mode == "per_node":
                msg_sum_f = msg.sum(dim=2)
                agg = self._aggregate_per_node(msg_sum_f)
                state_flat = state.reshape(batch_size * self.n_channels, self.hidden_dim)
                agg_flat = agg.reshape(batch_size * self.n_channels, self.hidden_dim)
                state = self.state_cell(agg_flat, state_flat).view(
                    batch_size, self.n_channels, self.hidden_dim
                )
            else:
                agg = self._aggregate_per_node_per_freq(msg)
                state_flat = state.reshape(
                    batch_size * self.n_channels * self.nfreqs, self.hidden_dim
                )
                agg_flat = agg.reshape(
                    batch_size * self.n_channels * self.nfreqs, self.hidden_dim
                )
                state = self.state_cell(agg_flat, state_flat).view(
                    batch_size, self.n_channels, self.nfreqs, self.hidden_dim
                )

        if self.state_mode == "per_node":
            pooled = state.mean(dim=1)
        else:
            pooled = state.mean(dim=(1, 2))

        logits = self.classifier(pooled)
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return logits, edge_density


class XWTPhaseGNNClassifier(ClassifierMixin, BaseEstimator):
    """sklearn/MOABB wrapper around the level-0 XWT phase GNN core."""

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 48,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        coi_mode: str = "ignore",
        state_mode: str = "per_node",
        phase_rule: str | Callable[[torch.Tensor, float], torch.Tensor] = "deadzone_sign",
        use_mag: bool = True,
        use_ang: bool = True,
        use_raw: bool = True,
        use_state_src: bool = True,
        use_state_dst: bool = True,
        hidden_dim: int = 64,
        message_dim: int = 64,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        readout_mode: str = "trial",
        verbose: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.cwt_resample_n_time = cwt_resample_n_time
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
        self.coi_mode = coi_mode
        self.state_mode = state_mode
        self.phase_rule = phase_rule
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_state_src = use_state_src
        self.use_state_dst = use_state_dst
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.normalize_input = normalize_input
        self.validation_split = validation_split
        self.validation_group_column = validation_group_column
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.seed = seed
        self.readout_mode = readout_mode
        self.verbose = verbose

        self.model_: XWTPhaseGNNCore | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_idx_: dict | None = None
        self.device_: torch.device | None = None
        self.transform_ = None
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        self.edge_density_history_: list[float] = []
        self.train_loss_history_: list[float] = []
        self.train_accuracy_history_: list[float] = []
        self.train_roc_auc_history_: list[float | None] = []
        self.val_loss_history_: list[float] = []
        self.val_accuracy_history_: list[float] = []
        self.val_roc_auc_history_: list[float | None] = []
        self.best_epoch_: int | None = None
        self.best_val_loss_: float | None = None
    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _vprint(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message, flush=True)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must have shape (n_samples, n_channels, n_timepoints)")
        return np.asarray(X, dtype=np.float32)

    def _compute_cwt_tensors(
        self, X: np.ndarray, fit_normalizer: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            if fit_normalizer:
                self.X_mean_, self.X_std_ = _fit_global_zscore_stats(X)
            if self.X_mean_ is None or self.X_std_ is None:
                raise ValueError("Input normalization stats are not initialized. Call fit first.")
            X_proc = _apply_global_zscore(X, self.X_mean_, self.X_std_)
        else:
            X_proc = X

        n_samples, n_channels, n_time_orig = X_proc.shape
        if self.cwt_resample_n_time is None:
            n_time = n_time_orig
        else:
            n_time = int(self.cwt_resample_n_time)
            if n_time <= 0:
                raise ValueError("cwt_resample_n_time must be a positive integer or None.")

        w_real = np.zeros((n_samples, n_channels, n_time, self.nfreqs), dtype=np.float32)
        w_imag = np.zeros((n_samples, n_channels, n_time, self.nfreqs), dtype=np.float32)

        if self.transform_ is None:
            self.transform_ = _resolve_transform_callable()

        total = n_samples * n_channels
        with tqdm(
            total=total,
            desc="CWT",
            disable=self.verbose < 1,
            leave=False,
        ) as pbar:
            for sample_idx in range(n_samples):
                for ch_idx in range(n_channels):
                    signal = X_proc[sample_idx, ch_idx, :]
                    coeffs, _ = self.transform_(
                        signal,
                        self.sampling_rate,
                        self.highest,
                        self.lowest,
                        nfreqs=self.nfreqs,
                    )
                    coeffs = np.asarray(coeffs)
                    if coeffs.ndim != 2:
                        raise ValueError(
                            f"CWT coeffs must be 2D, got shape {coeffs.shape}."
                        )

                    if coeffs.shape[0] == self.nfreqs:
                        coeffs_tf = coeffs.T
                    else:
                        coeffs_tf = coeffs

                    if coeffs_tf.shape[0] != n_time and self.cwt_resample_n_time is not None:
                        coeffs_tf = resample(coeffs_tf, n_time, axis=0)

                    if coeffs_tf.shape[0] != n_time or coeffs_tf.shape[1] != self.nfreqs:
                        raise ValueError(
                            "Unexpected CWT shape after transform. "
                            f"Expected (T={n_time}, F={self.nfreqs}), got {coeffs_tf.shape}."
                        )

                    coeffs_tf = np.nan_to_num(
                        coeffs_tf, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    w_real[sample_idx, ch_idx] = np.real(coeffs_tf).astype(np.float32)
                    w_imag[sample_idx, ch_idx] = np.imag(coeffs_tf).astype(np.float32)
                    pbar.update(1)

        if self.cwt_resample_n_time is not None and n_time != n_time_orig:
            raw_x = resample(X_proc, n_time, axis=2)
        else:
            raw_x = X_proc

        x_tensor = torch.from_numpy(
            np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        )
        w_real_tensor = torch.from_numpy(w_real)
        w_imag_tensor = torch.from_numpy(w_imag)
        return x_tensor.float(), w_real_tensor.float(), w_imag_tensor.float()

    def _build_model(self, n_channels: int, n_classes: int) -> XWTPhaseGNNCore:
        if self.coi_mode != "ignore":
            raise ValueError("Only coi_mode='ignore' is supported at level 0.")

        model = XWTPhaseGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            theta_dead_deg=self.theta_dead_deg,
            time_stride=self.time_stride,
            state_mode=self.state_mode,
            phase_rule=self.phase_rule,
            use_mag=self.use_mag,
            use_ang=self.use_ang,
            use_raw=self.use_raw,
            use_state_src=self.use_state_src,
            use_state_dst=self.use_state_dst,
            readout_mode=self.readout_mode,
        )
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_groups: np.ndarray | None = None,
        metadata=None,
    ):
        X = self._validate_X(X)
        _set_seed(self.seed)

        self.device_ = self._resolve_device()
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[cls] for cls in y], dtype=np.int64)
        n_classes = len(self.classes_)

        groups_for_split = None
        if self.validation_group_column is not None:
            if validation_groups is not None:
                groups_for_split = np.asarray(validation_groups)
            else:
                groups_for_split = _extract_groups_from_metadata(
                    metadata, self.validation_group_column, X.shape[0]
                )

        train_idx, val_idx, chosen_groups = _resolve_train_val_indices(
            n_samples=X.shape[0],
            y_idx=y_idx,
            seed=self.seed,
            validation_split=self.validation_split,
            validation_group_column=self.validation_group_column,
            validation_groups=groups_for_split,
        )
        if val_idx.size == 0:
            self._vprint(1, "[Train] validation disabled (no held-out samples).")
        else:
            if chosen_groups is None:
                self._vprint(
                    1,
                    f"[Train] validation split: {val_idx.size}/{X.shape[0]} samples held out.",
                )
            else:
                self._vprint(
                    1,
                    f"[Train] validation groups ({self.validation_group_column}): "
                    f"{chosen_groups.tolist()} -> {val_idx.size}/{X.shape[0]} samples.",
                )

        if self.normalize_input:
            self.X_mean_, self.X_std_ = _fit_global_zscore_stats(X[train_idx])
        x_tensor, w_real_tensor, w_imag_tensor = self._compute_cwt_tensors(
            X, fit_normalizer=False
        )
        y_tensor = torch.from_numpy(y_idx).long()

        train_dataset = TensorDataset(
            x_tensor[train_idx],
            w_real_tensor[train_idx],
            w_imag_tensor[train_idx],
            y_tensor[train_idx],
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = None
        if val_idx.size > 0:
            val_dataset = TensorDataset(
                x_tensor[val_idx],
                w_real_tensor[val_idx],
                w_imag_tensor[val_idx],
                y_tensor[val_idx],
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        n_channels = X.shape[1]
        self.model_ = self._build_model(n_channels=n_channels, n_classes=n_classes).to(self.device_)
        _print_torch_parameter_summary(self.model_, header="XWT-V1")

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.edge_density_history_ = []
        self.train_loss_history_ = []
        self.train_accuracy_history_ = []
        self.train_roc_auc_history_ = []
        self.val_loss_history_ = []
        self.val_accuracy_history_ = []
        self.val_roc_auc_history_ = []
        self.best_epoch_ = None
        self.best_val_loss_ = None

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
            edge_density_sum = 0.0
            n_batches = 0
            n_correct = 0
            n_seen = 0
            epoch_targets = []
            epoch_probas = []

            for step_idx, (batch_x, batch_wr, batch_wi, batch_y) in enumerate(train_loader, start=1):
                step_start = time.perf_counter()
                batch_x = batch_x.to(self.device_)
                batch_wr = batch_wr.to(self.device_)
                batch_wi = batch_wi.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits, edge_density = self.model_(batch_x, batch_wr, batch_wi)
                loss = criterion(logits, batch_y)
                loss.backward()
                if self.grad_clip_norm is not None and float(self.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(), max_norm=float(self.grad_clip_norm)
                    )
                optimizer.step()

                loss_sum += float(loss.item())
                edge_density_sum += float(edge_density)
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_proba_np = torch.softmax(logits.detach(), dim=1).cpu().numpy()
                epoch_targets.append(batch_y_np)
                epoch_probas.append(batch_proba_np)

                if self.verbose >= 2:
                    step_elapsed = time.perf_counter() - step_start
                    step_acc = float((preds == batch_y).float().mean().item())
                    step_rate = batch_y.numel() / max(step_elapsed, 1e-6)
                    running_loss = loss_sum / n_batches
                    running_acc = n_correct / max(1, n_seen)
                    step_auc = _safe_roc_auc(batch_y_np, batch_proba_np, n_classes)
                    running_auc = _safe_roc_auc(
                        np.concatenate(epoch_targets, axis=0),
                        np.concatenate(epoch_probas, axis=0),
                        n_classes,
                    )
                    self._vprint(
                        2,
                        f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                        f"step={step_idx}/{len(train_loader)} loss={float(loss.item()):.6f} "
                        f"acc={step_acc:.4f} roc_auc={_fmt_metric(step_auc)} "
                        f"edge_density={float(edge_density):.6f} "
                        f"running_loss={running_loss:.6f} running_acc={running_acc:.4f} "
                        f"running_roc_auc={_fmt_metric(running_auc)} "
                        f"rate={step_rate:.2f} samples/s",
                    )

            avg_loss = loss_sum / max(1, n_batches)
            avg_acc = n_correct / max(1, n_seen)
            avg_auc = _safe_roc_auc(
                np.concatenate(epoch_targets, axis=0),
                np.concatenate(epoch_probas, axis=0),
                n_classes,
            )
            avg_edge_density = edge_density_sum / max(1, n_batches)
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_rate = n_seen / max(epoch_elapsed, 1e-6)
            self.edge_density_history_.append(avg_edge_density)
            self.train_loss_history_.append(avg_loss)
            self.train_accuracy_history_.append(avg_acc)
            self.train_roc_auc_history_.append(avg_auc)

            loss_delta = 0.0 if prev_loss is None else (prev_loss - avg_loss)
            acc_delta = 0.0 if prev_acc is None else (avg_acc - prev_acc)
            auc_delta = None if (prev_auc is None or avg_auc is None) else (avg_auc - prev_auc)
            prev_loss = avg_loss
            prev_acc = avg_acc
            prev_auc = avg_auc

            val_suffix = ""
            if val_loader is not None:
                val_loss, val_acc, val_auc = _evaluate_epoch(
                    self.model_,
                    val_loader,
                    self.device_,
                    criterion,
                    n_classes,
                )
                self.val_loss_history_.append(val_loss)
                self.val_accuracy_history_.append(val_acc)
                self.val_roc_auc_history_.append(val_auc)
                val_suffix = (
                    f" val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                    f"val_roc_auc={_fmt_metric(val_auc)}"
                )

                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state = deepcopy(self.model_.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            self._vprint(
                1,
                f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                f"loss={avg_loss:.6f} (improve {loss_delta:+.6f}) "
                f"acc={avg_acc:.4f} (delta {acc_delta:+.4f}) "
                f"roc_auc={_fmt_metric(avg_auc)} "
                f"(delta {_fmt_metric(auc_delta) if auc_delta is not None else 'n/a'}) "
                f"edge_density={avg_edge_density:.6f} "
                f"epoch_time={epoch_elapsed:.2f}s rate={epoch_rate:.2f} samples/s"
                f"{val_suffix}",
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

        return self

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._validate_X(X)
        x_tensor, w_real_tensor, w_imag_tensor = self._compute_cwt_tensors(X)
        dataset = TensorDataset(x_tensor, w_real_tensor, w_imag_tensor)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
        )

        logits_list = []
        self.model_.eval()
        with torch.no_grad():
            for batch_x, batch_wr, batch_wi in loader:
                batch_x = batch_x.to(self.device_)
                batch_wr = batch_wr.to(self.device_)
                batch_wi = batch_wi.to(self.device_)
                logits, _ = self.model_(batch_x, batch_wr, batch_wi)
                logits_list.append(logits.cpu().numpy())

        return np.concatenate(logits_list, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        logits = self._predict_logits(X)
        y_idx = np.argmax(logits, axis=1)
        return self.classes_[y_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        logits_t = torch.from_numpy(logits)
        proba = torch.softmax(logits_t, dim=1).numpy()
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))


class XWTPhaseGNNV2Core(nn.Module):
    """V2 core with channel-local temporal encoder and frequency-indexed state."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        message_dim: int = 3,
        hidden_state_dim: int = 32,
        encoder_dim: int = 16,
        use_encoder_batch_norm: bool = True,
        encoder_dropout: float | None = 0.5,
        use_local_residual: bool = True,
        use_prev_state_mean: bool = True,
        gru_input_dropout: float | None = 0.0,
        readout_dropout: float | None = 0.0,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        use_raw_in_message: bool = True,
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        if encoder_dropout is not None:
            if float(encoder_dropout) < 0.0 or float(encoder_dropout) >= 1.0:
                raise ValueError("encoder_dropout must be in [0.0, 1.0), or None.")
        if gru_input_dropout is not None:
            if float(gru_input_dropout) < 0.0 or float(gru_input_dropout) >= 1.0:
                raise ValueError("gru_input_dropout must be in [0.0, 1.0), or None.")
        if readout_dropout is not None:
            if float(readout_dropout) < 0.0 or float(readout_dropout) >= 1.0:
                raise ValueError("readout_dropout must be in [0.0, 1.0), or None.")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.message_dim = message_dim
        self.hidden_state_dim = hidden_state_dim
        self.encoder_dim = encoder_dim
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.encoder_dropout = (
            None if encoder_dropout is None else float(encoder_dropout)
        )
        self.use_local_residual = use_local_residual
        self.use_prev_state_mean = use_prev_state_mean
        self.gru_input_dropout = (
            None if gru_input_dropout is None else float(gru_input_dropout)
        )
        self.readout_dropout = (
            None if readout_dropout is None else float(readout_dropout)
        )
        self.time_stride = time_stride
        self.theta_dead_rad = math.radians(theta_dead_deg)
        self.use_raw_in_message = use_raw_in_message
        self.phase_rule_fn = _phase_rule_deadzone_sign

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        # Channel-local temporal encoder: no channel mixing, no temporal pooling.
        encoder_layers: list[nn.Module] = [nn.Conv1d(1, encoder_dim, kernel_size=5, padding=2)]
        if self.use_encoder_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encoder_dim))
        encoder_layers.append(nn.ReLU())
        if self.encoder_dropout is not None and self.encoder_dropout > 0.0:
            encoder_layers.append(nn.Dropout(p=self.encoder_dropout))

        encoder_layers.append(nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, padding=2))
        if self.use_encoder_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(encoder_dim))
        encoder_layers.append(nn.ReLU())
        if self.encoder_dropout is not None and self.encoder_dropout > 0.0:
            encoder_layers.append(nn.Dropout(p=self.encoder_dropout))

        self.channel_encoder = nn.Sequential(*encoder_layers)

        message_in_dim = 1 + 2 * message_dim + (2 if use_raw_in_message else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(message_in_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

        # Optional local residual path from encoded local signal.
        # Project directly to per-frequency message slots [F * M].
        self.local_enc_proj = (
            nn.Linear(encoder_dim, nfreqs * message_dim)
            if use_local_residual
            else None
        )

        # Bridge hidden node state <-> per-frequency message state.
        self.state_to_freq_proj = nn.Linear(hidden_state_dim, nfreqs * message_dim)
        self.gru_input_proj = nn.Linear(nfreqs * message_dim, hidden_state_dim)
        self.gru_input_dropout_layer = (
            nn.Dropout(self.gru_input_dropout)
            if self.gru_input_dropout is not None and self.gru_input_dropout > 0.0
            else None
        )
        self.readout_dropout_layer = (
            nn.Dropout(self.readout_dropout)
            if self.readout_dropout is not None and self.readout_dropout > 0.0
            else None
        )

        # Shared across nodes.
        self.state_cell = nn.GRUCell(hidden_state_dim, hidden_state_dim)
        readout_dim = 2 * hidden_state_dim if self.use_prev_state_mean else hidden_state_dim
        self.classifier = nn.Linear(readout_dim, n_classes)

    def _aggregate_per_node_per_freq(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, F, M] to [B, C, F, M] by destination node."""
        batch_size, num_edges, nfreqs, k_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels * nfreqs,
            k_dim,
            device=device,
            dtype=msg.dtype,
        )

        base_b = (
            torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            * self.n_channels
            * nfreqs
        )
        base_f = torch.arange(nfreqs, device=device).view(1, 1, nfreqs)
        dst = self.dst_idx.view(1, num_edges, 1) * nfreqs + base_f + base_b
        agg.index_add_(0, dst.reshape(-1), msg.reshape(batch_size * num_edges * nfreqs, k_dim))
        return agg.view(batch_size, self.n_channels, nfreqs, k_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Forward pass.

        Parameters
        ----------
        raw_x : tensor, shape (B, C, T)
        w_real : tensor, shape (B, C, T, F)
        w_imag : tensor, shape (B, C, T, F)
        """
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}."
            )

        # Channel-local temporal encoding.
        enc = raw_x.reshape(batch_size * n_channels, 1, n_time)
        enc = self.channel_encoder(enc)  # [B*C, E, T]
        enc = enc.reshape(batch_size, n_channels, self.encoder_dim, n_time).permute(0, 1, 3, 2)
        # enc shape: [B, C, T, E]

        state = torch.zeros(
            batch_size, self.n_channels, self.hidden_state_dim, device=raw_x.device
        )

        gate_sum = 0.0
        gate_count = 0.0
        prev_state_sum = torch.zeros(batch_size, self.hidden_state_dim, device=raw_x.device)
        last_state_pooled = torch.zeros(
            batch_size, self.hidden_state_dim, device=raw_x.device
        )
        step_count = 0

        for t in range(0, n_time, self.time_stride):
            src_r = w_real[:, self.src_idx, t, :]
            src_i = w_imag[:, self.src_idx, t, :]
            dst_r = w_real[:, self.dst_idx, t, :]
            dst_i = w_imag[:, self.dst_idx, t, :]

            # (a + ib) * conj(c + id) = (ac + bd) + i(bc - ad)
            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i

            xwt_mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            xwt_mag = torch.nan_to_num(xwt_mag, nan=0.0, posinf=0.0, neginf=0.0)
            xwt_mag_log = torch.log1p(xwt_mag)

            ang = torch.atan2(xwt_imag, xwt_real)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))
            gate = self.phase_rule_fn(delta, self.theta_dead_rad)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)

            gate_sum += float(gate.sum().item())
            gate_count += float(gate.numel())

            freq_state = self.state_to_freq_proj(state).reshape(
                batch_size, self.n_channels, self.nfreqs, self.message_dim
            )
            src_state = freq_state[:, self.src_idx, :, :]
            dst_state = freq_state[:, self.dst_idx, :, :]
            feats = [xwt_mag_log.unsqueeze(-1), src_state, dst_state]

            if self.use_raw_in_message:
                raw_t = raw_x[:, :, t]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                src_raw = src_raw.expand(batch_size, self.src_idx.numel(), self.nfreqs, 1)
                dst_raw = dst_raw.expand(batch_size, self.dst_idx.numel(), self.nfreqs, 1)
                feats.extend([src_raw, dst_raw])

            message_in = torch.cat(feats, dim=-1)
            msg_raw = self.message_mlp(message_in)
            msg = msg_raw * gate.unsqueeze(-1)  # gate after message MLP

            agg_msg = self._aggregate_per_node_per_freq(msg)  # [B, C, F, K]

            if self.use_local_residual:
                # Local residual path.
                enc_t = enc[:, :, t, :]  # [B, C, E]
                if self.local_enc_proj is None:
                    raise RuntimeError("local_enc_proj is not initialized.")
                local_enc_term = self.local_enc_proj(enc_t).reshape(
                    batch_size, self.n_channels, self.nfreqs, self.message_dim
                )  # [B, C, F, M]
                update_in_freq = agg_msg + local_enc_term
            else:
                update_in_freq = agg_msg
            update_in = self.gru_input_proj(
                update_in_freq.reshape(batch_size, self.n_channels, self.nfreqs * self.message_dim)
            )
            if self.gru_input_dropout_layer is not None:
                update_in = self.gru_input_dropout_layer(update_in)

            state_flat = state.reshape(batch_size * self.n_channels, self.hidden_state_dim)
            update_flat = update_in.reshape(batch_size * self.n_channels, self.hidden_state_dim)
            state = self.state_cell(update_flat, state_flat).view(
                batch_size, self.n_channels, self.hidden_state_dim
            )

            pooled_nodes = state.mean(dim=1)  # [B, H]
            if self.use_prev_state_mean and step_count > 0:
                prev_state_sum += last_state_pooled
            last_state_pooled = pooled_nodes
            step_count += 1

        readout = last_state_pooled.reshape(batch_size, self.hidden_state_dim)
        if self.use_prev_state_mean:
            if step_count > 1:
                prev_state_mean = prev_state_sum / float(step_count - 1)
            else:
                prev_state_mean = torch.zeros_like(prev_state_sum)
            prev_flat = prev_state_mean.reshape(batch_size, self.hidden_state_dim)
            readout = torch.cat([readout, prev_flat], dim=1)
        if self.readout_dropout_layer is not None:
            readout = self.readout_dropout_layer(readout)
        logits = self.classifier(readout)
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return logits, edge_density


class XWTPhaseGNNV2Classifier(ClassifierMixin, BaseEstimator):
    """V2 sklearn/MOABB wrapper with channel-local encoder and freq-indexed state."""
    _estimator_type = "classifier"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 32,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
        coi_mode: str = "ignore",
        message_dim: int = 3,
        hidden_state_dim: int = 32,
        encoder_dim: int = 16,
        use_encoder_batch_norm: bool = True,
        encoder_dropout: float | None = 0.5,
        use_local_residual: bool = True,
        use_prev_state_mean: bool = True,
        gru_input_dropout: float | None = 0.0,
        readout_dropout: float | None = 0.0,
        use_raw_in_message: bool = True,
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        readout_mode: str = "trial",
        verbose: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.cwt_resample_n_time = cwt_resample_n_time
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
        self.coi_mode = coi_mode
        self.message_dim = message_dim
        self.hidden_state_dim = hidden_state_dim
        self.encoder_dim = encoder_dim
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.encoder_dropout = encoder_dropout
        self.use_local_residual = use_local_residual
        self.use_prev_state_mean = use_prev_state_mean
        self.gru_input_dropout = gru_input_dropout
        self.readout_dropout = readout_dropout
        self.use_raw_in_message = use_raw_in_message
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.normalize_input = normalize_input
        self.validation_split = validation_split
        self.validation_group_column = validation_group_column
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.seed = seed
        self.readout_mode = readout_mode
        self.verbose = verbose

        self.model_: XWTPhaseGNNV2Core | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_idx_: dict | None = None
        self.device_: torch.device | None = None
        self.transform_ = None
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        self.edge_density_history_: list[float] = []
        self.train_loss_history_: list[float] = []
        self.train_accuracy_history_: list[float] = []
        self.train_roc_auc_history_: list[float | None] = []
        self.val_loss_history_: list[float] = []
        self.val_accuracy_history_: list[float] = []
        self.val_roc_auc_history_: list[float | None] = []
        self.best_epoch_: int | None = None
        self.best_val_loss_: float | None = None

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _vprint(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(message, flush=True)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must have shape (n_samples, n_channels, n_timepoints)")
        return np.asarray(X, dtype=np.float32)

    def _compute_cwt_tensors(
        self, X: np.ndarray, fit_normalizer: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalize_input:
            if fit_normalizer:
                self.X_mean_, self.X_std_ = _fit_global_zscore_stats(X)
            if self.X_mean_ is None or self.X_std_ is None:
                raise ValueError("Input normalization stats are not initialized. Call fit first.")
            X_proc = _apply_global_zscore(X, self.X_mean_, self.X_std_)
        else:
            X_proc = X

        n_samples, n_channels, n_time_orig = X_proc.shape
        if self.cwt_resample_n_time is None:
            n_time = n_time_orig
        else:
            n_time = int(self.cwt_resample_n_time)
            if n_time <= 0:
                raise ValueError("cwt_resample_n_time must be a positive integer or None.")

        w_real = np.zeros((n_samples, n_channels, n_time, self.nfreqs), dtype=np.float32)
        w_imag = np.zeros((n_samples, n_channels, n_time, self.nfreqs), dtype=np.float32)

        if self.transform_ is None:
            self.transform_ = _resolve_transform_callable()

        total = n_samples * n_channels
        with tqdm(total=total, desc="CWT", disable=self.verbose < 1, leave=False) as pbar:
            for sample_idx in range(n_samples):
                for ch_idx in range(n_channels):
                    signal = X_proc[sample_idx, ch_idx, :]
                    coeffs, _ = self.transform_(
                        signal,
                        self.sampling_rate,
                        self.highest,
                        self.lowest,
                        nfreqs=self.nfreqs,
                    )
                    coeffs = np.asarray(coeffs)
                    if coeffs.ndim != 2:
                        raise ValueError(
                            f"CWT coeffs must be 2D, got shape {coeffs.shape}."
                        )

                    if coeffs.shape[0] == self.nfreqs:
                        coeffs_tf = coeffs.T
                    else:
                        coeffs_tf = coeffs

                    if coeffs_tf.shape[0] != n_time and self.cwt_resample_n_time is not None:
                        coeffs_tf = resample(coeffs_tf, n_time, axis=0)

                    if coeffs_tf.shape[0] != n_time or coeffs_tf.shape[1] != self.nfreqs:
                        raise ValueError(
                            "Unexpected CWT shape after transform. "
                            f"Expected (T={n_time}, F={self.nfreqs}), got {coeffs_tf.shape}."
                        )

                    coeffs_tf = np.nan_to_num(coeffs_tf, nan=0.0, posinf=0.0, neginf=0.0)
                    w_real[sample_idx, ch_idx] = np.real(coeffs_tf).astype(np.float32)
                    w_imag[sample_idx, ch_idx] = np.imag(coeffs_tf).astype(np.float32)
                    pbar.update(1)

        if self.cwt_resample_n_time is not None and n_time != n_time_orig:
            raw_x = resample(X_proc, n_time, axis=2)
        else:
            raw_x = X_proc

        x_tensor = torch.from_numpy(
            np.nan_to_num(raw_x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        )
        w_real_tensor = torch.from_numpy(w_real)
        w_imag_tensor = torch.from_numpy(w_imag)
        return x_tensor.float(), w_real_tensor.float(), w_imag_tensor.float()

    def _build_model(self, n_channels: int, n_classes: int) -> XWTPhaseGNNV2Core:
        if self.coi_mode != "ignore":
            raise ValueError("Only coi_mode='ignore' is supported at level 0.")
        if self.readout_mode != "trial":
            raise ValueError("Only readout_mode='trial' is supported for V2.")

        return XWTPhaseGNNV2Core(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            message_dim=self.message_dim,
            hidden_state_dim=self.hidden_state_dim,
            encoder_dim=self.encoder_dim,
            use_encoder_batch_norm=self.use_encoder_batch_norm,
            encoder_dropout=self.encoder_dropout,
            use_local_residual=self.use_local_residual,
            use_prev_state_mean=self.use_prev_state_mean,
            gru_input_dropout=self.gru_input_dropout,
            readout_dropout=self.readout_dropout,
            time_stride=self.time_stride,
            theta_dead_deg=self.theta_dead_deg,
            use_raw_in_message=self.use_raw_in_message,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_groups: np.ndarray | None = None,
        metadata=None,
    ):
        X = self._validate_X(X)
        _set_seed(self.seed)

        self.device_ = self._resolve_device()

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[cls] for cls in y], dtype=np.int64)
        n_classes = len(self.classes_)

        groups_for_split = None
        if self.validation_group_column is not None:
            if validation_groups is not None:
                groups_for_split = np.asarray(validation_groups)
            else:
                groups_for_split = _extract_groups_from_metadata(
                    metadata, self.validation_group_column, X.shape[0]
                )

        train_idx, val_idx, chosen_groups = _resolve_train_val_indices(
            n_samples=X.shape[0],
            y_idx=y_idx,
            seed=self.seed,
            validation_split=self.validation_split,
            validation_group_column=self.validation_group_column,
            validation_groups=groups_for_split,
        )
        if val_idx.size == 0:
            self._vprint(1, "[Train-V2] validation disabled (no held-out samples).")
        else:
            if chosen_groups is None:
                self._vprint(
                    1,
                    f"[Train-V2] validation split: {val_idx.size}/{X.shape[0]} samples held out.",
                )
            else:
                self._vprint(
                    1,
                    f"[Train-V2] validation groups ({self.validation_group_column}): "
                    f"{chosen_groups.tolist()} -> {val_idx.size}/{X.shape[0]} samples.",
                )

        if self.normalize_input:
            self.X_mean_, self.X_std_ = _fit_global_zscore_stats(X[train_idx])
        x_tensor, w_real_tensor, w_imag_tensor = self._compute_cwt_tensors(
            X, fit_normalizer=False
        )
        y_tensor = torch.from_numpy(y_idx).long()

        train_dataset = TensorDataset(
            x_tensor[train_idx],
            w_real_tensor[train_idx],
            w_imag_tensor[train_idx],
            y_tensor[train_idx],
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = None
        if val_idx.size > 0:
            val_dataset = TensorDataset(
                x_tensor[val_idx],
                w_real_tensor[val_idx],
                w_imag_tensor[val_idx],
                y_tensor[val_idx],
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        n_channels = X.shape[1]
        self.model_ = self._build_model(n_channels=n_channels, n_classes=n_classes).to(self.device_)
        _print_torch_parameter_summary(self.model_, header="XWT-V2")

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        self.edge_density_history_ = []
        self.train_loss_history_ = []
        self.train_accuracy_history_ = []
        self.train_roc_auc_history_ = []
        self.val_loss_history_ = []
        self.val_accuracy_history_ = []
        self.val_roc_auc_history_ = []
        self.best_epoch_ = None
        self.best_val_loss_ = None

        self._vprint(
            1,
            f"[Train-V2] start epochs={self.epochs} batches/epoch={len(train_loader)} "
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
            edge_density_sum = 0.0
            n_batches = 0
            n_correct = 0
            n_seen = 0
            epoch_targets = []
            epoch_probas = []

            for step_idx, (batch_x, batch_wr, batch_wi, batch_y) in enumerate(train_loader, start=1):
                step_start = time.perf_counter()
                batch_x = batch_x.to(self.device_)
                batch_wr = batch_wr.to(self.device_)
                batch_wi = batch_wi.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits, edge_density = self.model_(batch_x, batch_wr, batch_wi)
                loss = criterion(logits, batch_y)
                loss.backward()
                if self.grad_clip_norm is not None and float(self.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(), max_norm=float(self.grad_clip_norm)
                    )
                optimizer.step()

                loss_sum += float(loss.item())
                edge_density_sum += float(edge_density)
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_proba_np = torch.softmax(logits.detach(), dim=1).cpu().numpy()
                epoch_targets.append(batch_y_np)
                epoch_probas.append(batch_proba_np)

                if self.verbose >= 2:
                    step_elapsed = time.perf_counter() - step_start
                    step_acc = float((preds == batch_y).float().mean().item())
                    step_rate = batch_y.numel() / max(step_elapsed, 1e-6)
                    running_loss = loss_sum / n_batches
                    running_acc = n_correct / max(1, n_seen)
                    step_auc = _safe_roc_auc(batch_y_np, batch_proba_np, n_classes)
                    running_auc = _safe_roc_auc(
                        np.concatenate(epoch_targets, axis=0),
                        np.concatenate(epoch_probas, axis=0),
                        n_classes,
                    )
                    self._vprint(
                        2,
                        f"[Train-V2][Epoch {epoch + 1}/{self.epochs}] "
                        f"step={step_idx}/{len(train_loader)} loss={float(loss.item()):.6f} "
                        f"acc={step_acc:.4f} roc_auc={_fmt_metric(step_auc)} "
                        f"edge_density={float(edge_density):.6f} "
                        f"running_loss={running_loss:.6f} running_acc={running_acc:.4f} "
                        f"running_roc_auc={_fmt_metric(running_auc)} "
                        f"rate={step_rate:.2f} samples/s",
                    )

            avg_loss = loss_sum / max(1, n_batches)
            avg_acc = n_correct / max(1, n_seen)
            avg_auc = _safe_roc_auc(
                np.concatenate(epoch_targets, axis=0),
                np.concatenate(epoch_probas, axis=0),
                n_classes,
            )
            avg_edge_density = edge_density_sum / max(1, n_batches)
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_rate = n_seen / max(epoch_elapsed, 1e-6)
            self.edge_density_history_.append(avg_edge_density)
            self.train_loss_history_.append(avg_loss)
            self.train_accuracy_history_.append(avg_acc)
            self.train_roc_auc_history_.append(avg_auc)

            loss_delta = 0.0 if prev_loss is None else (prev_loss - avg_loss)
            acc_delta = 0.0 if prev_acc is None else (avg_acc - prev_acc)
            auc_delta = None if (prev_auc is None or avg_auc is None) else (avg_auc - prev_auc)
            prev_loss = avg_loss
            prev_acc = avg_acc
            prev_auc = avg_auc

            val_suffix = ""
            if val_loader is not None:
                val_loss, val_acc, val_auc = _evaluate_epoch(
                    self.model_,
                    val_loader,
                    self.device_,
                    criterion,
                    n_classes,
                )
                self.val_loss_history_.append(val_loss)
                self.val_accuracy_history_.append(val_acc)
                self.val_roc_auc_history_.append(val_auc)
                val_suffix = (
                    f" val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
                    f"val_roc_auc={_fmt_metric(val_auc)}"
                )

                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state = deepcopy(self.model_.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            self._vprint(
                1,
                f"[Train-V2][Epoch {epoch + 1}/{self.epochs}] "
                f"loss={avg_loss:.6f} (improve {loss_delta:+.6f}) "
                f"acc={avg_acc:.4f} (delta {acc_delta:+.4f}) "
                f"roc_auc={_fmt_metric(avg_auc)} "
                f"(delta {_fmt_metric(auc_delta) if auc_delta is not None else 'n/a'}) "
                f"edge_density={avg_edge_density:.6f} "
                f"epoch_time={epoch_elapsed:.2f}s rate={epoch_rate:.2f} samples/s"
                f"{val_suffix}",
            )

            if (
                val_loader is not None
                and self.early_stopping_patience is not None
                and self.early_stopping_patience >= 0
                and no_improve_epochs >= self.early_stopping_patience
            ):
                self._vprint(
                    1,
                    f"[Train-V2] early stopping at epoch {epoch + 1}; "
                    f"best epoch={best_epoch} best_val_loss={best_val_loss:.6f}",
                )
                break

        if val_loader is not None and best_state is not None:
            self.model_.load_state_dict(best_state)
            self.best_epoch_ = best_epoch
            self.best_val_loss_ = best_val_loss
            self._vprint(
                1,
                f"[Train-V2] restored best model from epoch {best_epoch} "
                f"(val_loss={best_val_loss:.6f})",
            )

        return self

    def _predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._validate_X(X)
        x_tensor, w_real_tensor, w_imag_tensor = self._compute_cwt_tensors(X)
        dataset = TensorDataset(x_tensor, w_real_tensor, w_imag_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        logits_list = []
        self.model_.eval()
        with torch.no_grad():
            for batch_x, batch_wr, batch_wi in loader:
                batch_x = batch_x.to(self.device_)
                batch_wr = batch_wr.to(self.device_)
                batch_wi = batch_wi.to(self.device_)
                logits, _ = self.model_(batch_x, batch_wr, batch_wi)
                logits_list.append(logits.cpu().numpy())

        return np.concatenate(logits_list, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")
        logits = self._predict_logits(X)
        y_idx = np.argmax(logits, axis=1)
        return self.classes_[y_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(X)
        logits_t = torch.from_numpy(logits)
        proba = torch.softmax(logits_t, dim=1).numpy()
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(X)))
