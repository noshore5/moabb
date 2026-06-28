"""Level-0 XWT phase-conditioned GNN classifiers."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

try:
    from coheriqs_contributions.moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        augment_paired_cwt_batch,
        compute_paired_cwt_noise_bank,
        compute_cwt_real_imag_tensors,
        fit_global_zscore_stats,
        ordered_pair_indices as _ordered_pair_indices,
        phase_rule_deadzone_sign as _phase_rule_deadzone_sign,
        resolve_coherence_utils,
        resolve_phase_rule as _resolve_phase_rule,
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        augment_paired_cwt_batch,
        compute_paired_cwt_noise_bank,
        compute_cwt_real_imag_tensors,
        fit_global_zscore_stats,
        ordered_pair_indices as _ordered_pair_indices,
        phase_rule_deadzone_sign as _phase_rule_deadzone_sign,
        resolve_coherence_utils,
        resolve_phase_rule as _resolve_phase_rule,
    )


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
        **kwargs,
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        if state_mode not in {"per_node", "per_node_per_freq"}:
            raise ValueError("state_mode must be one of {'per_node', 'per_node_per_freq'}")

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
            batch_size * self.n_channels,
            hidden_dim,
            device=device,
            dtype=msg.dtype,
        )
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * self.n_channels
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
        agg.index_add_(
            0,
            dst.reshape(-1),
            msg.reshape(batch_size * num_edges * nfreqs, hidden_dim),
        )
        return agg.view(batch_size, self.n_channels, nfreqs, hidden_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        device = raw_x.device
        num_edges = self.src_idx.numel()
        if self.state_mode == "per_node":
            state = torch.zeros(batch_size, self.n_channels, self.hidden_dim, device=device)
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
                features.append(src_raw.expand(batch_size, num_edges, self.nfreqs, 1))
                features.append(dst_raw.expand(batch_size, num_edges, self.nfreqs, 1))

            if self.state_mode == "per_node":
                if self.use_state_src:
                    src_state = state[:, self.src_idx, :].unsqueeze(2)
                    features.append(
                        src_state.expand(batch_size, num_edges, self.nfreqs, self.hidden_dim)
                    )
                if self.use_state_dst:
                    dst_state = state[:, self.dst_idx, :].unsqueeze(2)
                    features.append(
                        dst_state.expand(batch_size, num_edges, self.nfreqs, self.hidden_dim)
                    )
            else:
                if self.use_state_src:
                    features.append(state[:, self.src_idx, :, :])
                if self.use_state_dst:
                    features.append(state[:, self.dst_idx, :, :])

            msg = self.message_mlp(torch.cat(features, dim=-1))
            msg = msg * gate.unsqueeze(-1)

            if self.state_mode == "per_node":
                agg = self._aggregate_per_node(msg.sum(dim=2))
                state = self.state_cell(
                    agg.reshape(batch_size * self.n_channels, self.hidden_dim),
                    state.reshape(batch_size * self.n_channels, self.hidden_dim),
                ).view(batch_size, self.n_channels, self.hidden_dim)
            else:
                agg = self._aggregate_per_node_per_freq(msg)
                state = self.state_cell(
                    agg.reshape(batch_size * self.n_channels * self.nfreqs, self.hidden_dim),
                    state.reshape(batch_size * self.n_channels * self.nfreqs, self.hidden_dim),
                ).view(batch_size, self.n_channels, self.nfreqs, self.hidden_dim)

        pooled = state.mean(dim=1) if self.state_mode == "per_node" else state.mean(dim=(1, 2))
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return self.classifier(pooled), edge_density


class _BaseCWTGNNClassifier(TorchEEGClassifier):
    """Shared sklearn wrapper logic for XWT/WCT CWT-tensor GNNs."""

    _estimator_type = "classifier"
    model_label = "CWT-GNN"

    def _init_cwt_gnn_classifier(
        self,
        *,
        sampling_rate: int,
        lowest: float,
        highest: float,
        nfreqs: int,
        cwt_resample_n_time: int | None,
        normalize_input: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        grad_clip_norm: float | None,
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.cwt_resample_n_time = cwt_resample_n_time
        self.normalize_input = normalize_input
        self.noise_augmentation_enabled = noise_augmentation_enabled
        self.noise_apply_prob = noise_apply_prob
        self.noise_strength = noise_strength
        self.noise_bank_size = noise_bank_size
        self.noise_bank_seed = noise_bank_seed
        self.transform_ = None
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
        self.noise_bank_: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.noise_channel_std_: torch.Tensor | None = None
        self.noise_bank_device_: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.noise_channel_std_device_: torch.Tensor | None = None
        self._validate_noise_augmentation_params()
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            use_class_weights=False,
            verbose=verbose,
        )

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        if fit:
            self._validate_noise_augmentation_params()
        if self.normalize_input:
            if fit:
                ref = X if train_idx is None else X[train_idx]
                self.X_mean_, self.X_std_ = fit_global_zscore_stats(ref)
            if self.X_mean_ is None or self.X_std_ is None:
                raise ValueError("Input normalization stats are not initialized.")
            X = apply_global_zscore(X, self.X_mean_, self.X_std_)

        if self.transform_ is None:
            self.transform_, _ = resolve_coherence_utils()
        features = compute_cwt_real_imag_tensors(
            X,
            sampling_rate=self.sampling_rate,
            highest=self.highest,
            lowest=self.lowest,
            nfreqs=self.nfreqs,
            cwt_resample_n_time=self.cwt_resample_n_time,
            transform_fn=self.transform_,
            verbose=self.verbose,
        )
        if fit:
            self._fit_noise_augmentation_state(features, X, train_idx)
        return features

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        raw_x = features[0] if isinstance(features, tuple) else features
        return self._build_model(n_channels=int(raw_x.shape[1]), n_classes=n_classes, **kwargs)

    def _validate_noise_augmentation_params(self) -> None:
        if not 0.0 <= float(self.noise_apply_prob) <= 1.0:
            raise ValueError("noise_apply_prob must be in [0.0, 1.0].")
        if float(self.noise_strength) < 0.0:
            raise ValueError("noise_strength must be >= 0.0.")
        if int(self.noise_bank_size) <= 0:
            raise ValueError("noise_bank_size must be > 0.")

    def _uses_noise_augmentation(self) -> bool:
        return (
            bool(self.noise_augmentation_enabled)
            and float(self.noise_apply_prob) > 0.0
            and float(self.noise_strength) > 0.0
        )

    def _fit_noise_augmentation_state(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        X: np.ndarray,
        train_idx,
    ) -> None:
        self.noise_bank_ = None
        self.noise_channel_std_ = None
        self.noise_bank_device_ = None
        self.noise_channel_std_device_ = None
        if not self._uses_noise_augmentation():
            return

        raw_x = features[0]
        if train_idx is None:
            ref = raw_x
        else:
            ref = raw_x[torch.as_tensor(train_idx, dtype=torch.long)]
        channel_std = torch.std(ref, dim=(0, 2), unbiased=False)
        channel_std = torch.nan_to_num(channel_std, nan=0.0, posinf=0.0, neginf=0.0)
        self.noise_channel_std_ = channel_std.float().contiguous()
        bank_seed = (
            int(self.noise_bank_seed)
            if self.noise_bank_seed is not None
            else int(self.seed or 0) + 10_003
        )
        self.noise_bank_ = compute_paired_cwt_noise_bank(
            bank_size=int(self.noise_bank_size),
            segment_length=int(X.shape[2]),
            sampling_rate=self.sampling_rate,
            highest=self.highest,
            lowest=self.lowest,
            nfreqs=self.nfreqs,
            cwt_resample_n_time=self.cwt_resample_n_time,
            transform_fn=self.transform_,
            seed=bank_seed,
            verbose=self.verbose,
        )

    def _prepare_training_state_on_device(self) -> None:
        self.noise_bank_device_ = None
        self.noise_channel_std_device_ = None
        if not self._uses_noise_augmentation():
            return
        if self.device_ is None:
            raise ValueError("Torch device is not initialized.")
        if self.noise_bank_ is None or self.noise_channel_std_ is None:
            raise ValueError("Noise augmentation state is not initialized.")
        self.noise_bank_device_ = tuple(
            tensor.to(device=self.device_, dtype=torch.float32)
            for tensor in self.noise_bank_
        )
        self.noise_channel_std_device_ = self.noise_channel_std_.to(
            device=self.device_,
            dtype=torch.float32,
        )

    def _augment_train_batch_inputs(
        self, batch_inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if not self._uses_noise_augmentation():
            return batch_inputs
        if self.noise_bank_device_ is None or self.noise_channel_std_device_ is None:
            raise ValueError("Noise augmentation device state is not initialized.")
        return augment_paired_cwt_batch(
            batch_inputs,
            noise_bank=self.noise_bank_device_,
            channel_std=self.noise_channel_std_device_,
            apply_prob=float(self.noise_apply_prob),
            strength=float(self.noise_strength),
        )


class XWTPhaseGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around the level-0 XWT phase GNN core."""

    model_label = "XWT-V1"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 48,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
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
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
        self.state_mode = state_mode
        self.phase_rule = phase_rule
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_state_src = use_state_src
        self.use_state_dst = use_state_dst
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self._init_cwt_gnn_classifier(
            sampling_rate=sampling_rate,
            lowest=lowest,
            highest=highest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            normalize_input=normalize_input,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            noise_augmentation_enabled=noise_augmentation_enabled,
            noise_apply_prob=noise_apply_prob,
            noise_strength=noise_strength,
            noise_bank_size=noise_bank_size,
            noise_bank_seed=noise_bank_seed,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            verbose=verbose,
        )

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> XWTPhaseGNNCore:
        return XWTPhaseGNNCore(
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
            **kwargs,
        )


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
        **kwargs,
    ) -> None:
        super().__init__()
        if time_stride <= 0:
            raise ValueError("time_stride must be >= 1")
        for name, value in {
            "encoder_dropout": encoder_dropout,
            "gru_input_dropout": gru_input_dropout,
            "readout_dropout": readout_dropout,
        }.items():
            if value is not None and (float(value) < 0.0 or float(value) >= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0), or None.")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.message_dim = message_dim
        self.hidden_state_dim = hidden_state_dim
        self.encoder_dim = encoder_dim
        self.use_encoder_batch_norm = use_encoder_batch_norm
        self.encoder_dropout = None if encoder_dropout is None else float(encoder_dropout)
        self.use_local_residual = use_local_residual
        self.use_prev_state_mean = use_prev_state_mean
        self.gru_input_dropout = None if gru_input_dropout is None else float(gru_input_dropout)
        self.readout_dropout = None if readout_dropout is None else float(readout_dropout)
        self.time_stride = time_stride
        self.theta_dead_rad = math.radians(theta_dead_deg)
        self.use_raw_in_message = use_raw_in_message
        self.phase_rule_fn = _phase_rule_deadzone_sign

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

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
        self.local_enc_proj = (
            nn.Linear(encoder_dim, nfreqs * message_dim) if use_local_residual else None
        )
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
        self.state_cell = nn.GRUCell(hidden_state_dim, hidden_state_dim)
        readout_dim = 2 * hidden_state_dim if self.use_prev_state_mean else hidden_state_dim
        self.classifier = nn.Linear(readout_dim, n_classes)

    def _aggregate_per_node_per_freq(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, F, M] to [B, C, F, M] by destination node."""
        batch_size, num_edges, nfreqs, message_dim = msg.shape
        device = msg.device
        agg = torch.zeros(
            batch_size * self.n_channels * nfreqs,
            message_dim,
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
        agg.index_add_(
            0,
            dst.reshape(-1),
            msg.reshape(batch_size * num_edges * nfreqs, message_dim),
        )
        return agg.view(batch_size, self.n_channels, nfreqs, message_dim)

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        enc = raw_x.reshape(batch_size * n_channels, 1, n_time)
        enc = self.channel_encoder(enc)
        enc = enc.reshape(batch_size, n_channels, self.encoder_dim, n_time).permute(0, 1, 3, 2)

        state = torch.zeros(
            batch_size,
            self.n_channels,
            self.hidden_state_dim,
            device=raw_x.device,
        )
        gate_sum = 0.0
        gate_count = 0.0
        prev_state_sum = torch.zeros(batch_size, self.hidden_state_dim, device=raw_x.device)
        last_state_pooled = torch.zeros(batch_size, self.hidden_state_dim, device=raw_x.device)
        step_count = 0

        for t in range(0, n_time, self.time_stride):
            src_r = w_real[:, self.src_idx, t, :]
            src_i = w_imag[:, self.src_idx, t, :]
            dst_r = w_real[:, self.dst_idx, t, :]
            dst_i = w_imag[:, self.dst_idx, t, :]

            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i
            xwt_mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            xwt_mag_log = torch.log1p(torch.nan_to_num(xwt_mag, nan=0.0, posinf=0.0, neginf=0.0))

            ang = torch.atan2(xwt_imag, xwt_real)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))
            gate = self.phase_rule_fn(delta, self.theta_dead_rad)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
            gate_sum += float(gate.sum().item())
            gate_count += float(gate.numel())

            freq_state = self.state_to_freq_proj(state).reshape(
                batch_size,
                self.n_channels,
                self.nfreqs,
                self.message_dim,
            )
            src_state = freq_state[:, self.src_idx, :, :]
            dst_state = freq_state[:, self.dst_idx, :, :]
            feats = [xwt_mag_log.unsqueeze(-1), src_state, dst_state]
            if self.use_raw_in_message:
                raw_t = raw_x[:, :, t]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                feats.extend(
                    [
                        src_raw.expand(batch_size, self.src_idx.numel(), self.nfreqs, 1),
                        dst_raw.expand(batch_size, self.dst_idx.numel(), self.nfreqs, 1),
                    ]
                )

            msg = self.message_mlp(torch.cat(feats, dim=-1))
            msg = msg * gate.unsqueeze(-1)
            agg_msg = self._aggregate_per_node_per_freq(msg)

            if self.use_local_residual:
                if self.local_enc_proj is None:
                    raise RuntimeError("local_enc_proj is not initialized.")
                local_enc_term = self.local_enc_proj(enc[:, :, t, :]).reshape(
                    batch_size,
                    self.n_channels,
                    self.nfreqs,
                    self.message_dim,
                )
                update_in_freq = agg_msg + local_enc_term
            else:
                update_in_freq = agg_msg

            update_in = self.gru_input_proj(
                update_in_freq.reshape(batch_size, self.n_channels, self.nfreqs * self.message_dim)
            )
            if self.gru_input_dropout_layer is not None:
                update_in = self.gru_input_dropout_layer(update_in)

            state = self.state_cell(
                update_in.reshape(batch_size * self.n_channels, self.hidden_state_dim),
                state.reshape(batch_size * self.n_channels, self.hidden_state_dim),
            ).view(batch_size, self.n_channels, self.hidden_state_dim)

            pooled_nodes = state.mean(dim=1)
            if self.use_prev_state_mean and step_count > 0:
                prev_state_sum += last_state_pooled
            last_state_pooled = pooled_nodes
            step_count += 1

        readout = last_state_pooled.reshape(batch_size, self.hidden_state_dim)
        if self.use_prev_state_mean:
            prev_state_mean = (
                prev_state_sum / float(step_count - 1)
                if step_count > 1
                else torch.zeros_like(prev_state_sum)
            )
            readout = torch.cat([readout, prev_state_mean], dim=1)
        if self.readout_dropout_layer is not None:
            readout = self.readout_dropout_layer(readout)

        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return self.classifier(readout), edge_density


class XWTPhaseGNNV2Classifier(_BaseCWTGNNClassifier):
    """V2 sklearn/MOABB wrapper with channel-local encoder and freq-indexed state."""

    model_label = "XWT-V2"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 32,
        cwt_resample_n_time: int | None = None,
        time_stride: int = 1,
        theta_dead_deg: float = 45.0,
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
        noise_augmentation_enabled: bool = False,
        noise_apply_prob: float = 0.0,
        noise_strength: float = 0.0,
        noise_bank_size: int = 128,
        noise_bank_seed: int | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.time_stride = time_stride
        self.theta_dead_deg = theta_dead_deg
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
        self._init_cwt_gnn_classifier(
            sampling_rate=sampling_rate,
            lowest=lowest,
            highest=highest,
            nfreqs=nfreqs,
            cwt_resample_n_time=cwt_resample_n_time,
            normalize_input=normalize_input,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            noise_augmentation_enabled=noise_augmentation_enabled,
            noise_apply_prob=noise_apply_prob,
            noise_strength=noise_strength,
            noise_bank_size=noise_bank_size,
            noise_bank_seed=noise_bank_seed,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            verbose=verbose,
        )

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> XWTPhaseGNNV2Core:
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
            **kwargs,
        )
