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
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


log = logging.getLogger(__name__)


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


class XWTPhaseGNNClassifier(BaseEstimator, ClassifierMixin):
    """sklearn/MOABB wrapper around the level-0 XWT phase GNN core."""

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 48,
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
        device: str = "auto",
        seed: int = 42,
        readout_mode: str = "trial",
        verbose: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
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
        self.device = device
        self.seed = seed
        self.readout_mode = readout_mode
        self.verbose = verbose

        self.model_: XWTPhaseGNNCore | None = None
        self.classes_: np.ndarray | None = None
        self.class_to_idx_: dict | None = None
        self.device_: torch.device | None = None
        self.transform_ = None
        self.edge_density_history_: list[float] = []
        self.train_loss_history_: list[float] = []
        self.train_accuracy_history_: list[float] = []

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

    def _compute_cwt_tensors(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples, n_channels, n_time = X.shape
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
                    signal = X[sample_idx, ch_idx, :]
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

        x_tensor = torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._validate_X(X)
        _set_seed(self.seed)

        self.device_ = self._resolve_device()
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[cls] for cls in y], dtype=np.int64)

        x_tensor, w_real_tensor, w_imag_tensor = self._compute_cwt_tensors(X)
        y_tensor = torch.from_numpy(y_idx).long()

        dataset = TensorDataset(x_tensor, w_real_tensor, w_imag_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        n_channels = X.shape[1]
        n_classes = len(self.classes_)
        self.model_ = self._build_model(n_channels=n_channels, n_classes=n_classes).to(self.device_)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.edge_density_history_ = []
        self.train_loss_history_ = []
        self.train_accuracy_history_ = []

        self._vprint(
            1,
            f"[Train] start epochs={self.epochs} batches/epoch={len(loader)} "
            f"batch_size={self.batch_size} device={self.device_}",
        )
        prev_loss = None
        prev_acc = None

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            self.model_.train()
            loss_sum = 0.0
            edge_density_sum = 0.0
            n_batches = 0
            n_correct = 0
            n_seen = 0

            for step_idx, (batch_x, batch_wr, batch_wi, batch_y) in enumerate(loader, start=1):
                step_start = time.perf_counter()
                batch_x = batch_x.to(self.device_)
                batch_wr = batch_wr.to(self.device_)
                batch_wi = batch_wi.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits, edge_density = self.model_(batch_x, batch_wr, batch_wi)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())
                edge_density_sum += float(edge_density)
                n_batches += 1
                preds = torch.argmax(logits, dim=1)
                n_correct += int((preds == batch_y).sum().item())
                n_seen += int(batch_y.numel())

                if self.verbose >= 2:
                    step_elapsed = time.perf_counter() - step_start
                    step_acc = float((preds == batch_y).float().mean().item())
                    step_rate = batch_y.numel() / max(step_elapsed, 1e-6)
                    running_loss = loss_sum / n_batches
                    running_acc = n_correct / max(1, n_seen)
                    self._vprint(
                        2,
                        f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                        f"step={step_idx}/{len(loader)} loss={float(loss.item()):.6f} "
                        f"acc={step_acc:.4f} edge_density={float(edge_density):.6f} "
                        f"running_loss={running_loss:.6f} running_acc={running_acc:.4f} "
                        f"rate={step_rate:.2f} samples/s",
                    )

            avg_loss = loss_sum / max(1, n_batches)
            avg_acc = n_correct / max(1, n_seen)
            avg_edge_density = edge_density_sum / max(1, n_batches)
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_rate = n_seen / max(epoch_elapsed, 1e-6)
            self.edge_density_history_.append(avg_edge_density)
            self.train_loss_history_.append(avg_loss)
            self.train_accuracy_history_.append(avg_acc)

            loss_delta = 0.0 if prev_loss is None else (prev_loss - avg_loss)
            acc_delta = 0.0 if prev_acc is None else (avg_acc - prev_acc)
            prev_loss = avg_loss
            prev_acc = avg_acc

            self._vprint(
                1,
                f"[Train][Epoch {epoch + 1}/{self.epochs}] "
                f"loss={avg_loss:.6f} (improve {loss_delta:+.6f}) "
                f"acc={avg_acc:.4f} (delta {acc_delta:+.4f}) "
                f"edge_density={avg_edge_density:.6f} "
                f"epoch_time={epoch_elapsed:.2f}s rate={epoch_rate:.2f} samples/s",
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
