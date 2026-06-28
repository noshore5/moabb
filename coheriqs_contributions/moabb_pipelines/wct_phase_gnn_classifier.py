"""Level-0 WCT phase-conditioned GNN classifier.

This module contains:
- a torch core model that performs message passing over phase-gated WCT edges
- sklearn-compatible wrappers for MOABB pipeline integration (V1/V2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from coheriqs_contributions.constructions.coherence_analysis import visualize_complex_points_and_mean

try:
    from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _ordered_pair_indices,
    )
except ModuleNotFoundError:
    from moabb_pipelines.xwt_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _ordered_pair_indices,
    )


def _compute_wct_window_features(
    src_r: torch.Tensor,
    src_i: torch.Tensor,
    dst_r: torch.Tensor,
    dst_i: torch.Tensor,
    freqs: torch.Tensor,
    smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]] = None,
    padding_mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute window-level WCT features for directed source/destination pairs."""
    # (a + ib) * conj(c + id) = (ac + bd) + i(bc - ad)
    xwt_real = src_r * dst_r + src_i * dst_i
    xwt_imag = src_i * dst_r - src_r * dst_i
    

    mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
    # ang = torch.atan2(xwt_imag, xwt_real)

    auto1 = src_r * src_r + src_i * src_i
    auto2 = dst_r * dst_r + dst_i * dst_i
    if smooth_kernel_and_pad is None:
        cross = torch.complex(xwt_real, xwt_imag)
        smooth_cross = torch.mean(cross, dim=2)
        smooth_auto1 = torch.mean(auto1, dim=2)
        smooth_auto2 = torch.mean(auto2, dim=2)
        coh = (smooth_cross.abs() ** 2) / (smooth_auto1 * smooth_auto2 + 1e-12)
    else:
        smooth_kernel, pad = smooth_kernel_and_pad
        B, E, T, F = xwt_real.shape

        inv_scale = freqs.view(B, 1, 1, F)
        xwt_real = xwt_real * inv_scale
        xwt_imag = xwt_imag * inv_scale
        auto1 = auto1 * inv_scale
        auto2 = auto2 * inv_scale

        # Stack all maps so we smooth everything in one conv2d call.
        # Shape: [B, E, 4, T, F]
        maps = torch.stack([xwt_real, xwt_imag, auto1, auto2], dim=2)
        maps = maps.view(B * E * 4, 1, T, F)

        maps = torch.nn.functional.pad(maps, pad, mode=padding_mode)
        smoothed = torch.nn.functional.conv2d(maps, smooth_kernel)

        out_T, out_F = smoothed.shape[-2:]

        smoothed = smoothed.view(B, E, 4, out_T, out_F)

        smooth_cross = torch.complex(smoothed[:,:,0], smoothed[:,:,1])
        smooth_auto1 = smoothed[:,:,2]
        smooth_auto2 = smoothed[:,:,3]

        coh = (smooth_cross.abs() ** 2) / (smooth_auto1 * smooth_auto2 + 1e-12)

        coh, coh_max_idx = torch.max(coh, dim=2)

        idx = coh_max_idx.unsqueeze(2)  # [B, E, 1, F]

        smooth_cross = torch.gather(smooth_cross, dim=2, index=idx).squeeze(2)

    coh = coh.clamp(min=0.0, max=1.0)

    mean_mag = torch.mean(mag, dim=2)

    # mean_phase = torch.mean(ang, dim=2)
    mean_phase = torch.angle(smooth_cross)

    return mean_mag, mean_phase, coh


def _original_next_state(self, state, gate_mask, mag, ang, raw_t):
    """
    Original batch loop version.

    Expected shapes:
      gate_mask: [B, E, F]
      mag:       [B, E, F]
      ang:       [B, E, F]
      raw_t:     [B, C]
      state:
        if self.state_mode == "per_node":
            [B, C, H]
        else:
            [B, C, F, H]
    """
    device = state.device
    batch_size = state.shape[0]

    next_state = []

    for b in range(batch_size):
        active_idx = torch.nonzero(gate_mask[b], as_tuple=False)

        if active_idx.numel() == 0:
            if self.state_mode == "per_node":
                agg_b = torch.zeros(
                    self.n_channels,
                    self.hidden_dim,
                    device=device,
                    dtype=state.dtype,
                )
                state_b = self.state_cell(agg_b, state[b])
            else:
                agg_b = torch.zeros(
                    self.n_channels * self.nfreqs,
                    self.hidden_dim,
                    device=device,
                    dtype=state.dtype,
                )
                state_b = state[b].reshape(
                    self.n_channels * self.nfreqs,
                    self.hidden_dim,
                )
                state_b = self.state_cell(agg_b, state_b).view(
                    self.n_channels,
                    self.nfreqs,
                    self.hidden_dim,
                )

            next_state.append(state_b)
            continue

        edge_idx = active_idx[:, 0]
        freq_idx = active_idx[:, 1]

        features = []

        if self.use_mag:
            features.append(mag[b, edge_idx, freq_idx].unsqueeze(-1))

        if self.use_ang:
            features.append(ang[b, edge_idx, freq_idx].unsqueeze(-1))

        if self.use_raw:
            src_raw = raw_t[b, self.src_idx[edge_idx]].unsqueeze(-1)
            dst_raw = raw_t[b, self.dst_idx[edge_idx]].unsqueeze(-1)
            features.append(src_raw)
            features.append(dst_raw)

        if self.state_mode == "per_node":
            if self.use_state_src:
                src_state = state[b, self.src_idx[edge_idx], :]
                features.append(src_state)

            if self.use_state_dst:
                dst_state = state[b, self.dst_idx[edge_idx], :]
                features.append(dst_state)

        else:
            if self.use_state_src:
                src_state = state[b, self.src_idx[edge_idx], freq_idx, :]
                features.append(src_state)

            if self.use_state_dst:
                dst_state = state[b, self.dst_idx[edge_idx], freq_idx, :]
                features.append(dst_state)

        payload = torch.cat(features, dim=-1)
        msg = self.message_mlp(payload)

        if self.state_mode == "per_node":
            agg_b = torch.zeros(
                self.n_channels,
                self.hidden_dim,
                device=device,
                dtype=msg.dtype,
            )
            agg_b.index_add_(0, self.dst_idx[edge_idx], msg)

            state_b = self.state_cell(agg_b, state[b])

        else:
            agg_b = torch.zeros(
                self.n_channels * self.nfreqs,
                self.hidden_dim,
                device=device,
                dtype=msg.dtype,
            )

            flat_idx = self.dst_idx[edge_idx] * self.nfreqs + freq_idx
            agg_b.index_add_(0, flat_idx, msg)

            state_b = state[b].reshape(
                self.n_channels * self.nfreqs,
                self.hidden_dim,
            )
            state_b = self.state_cell(agg_b, state_b).view(
                self.n_channels,
                self.nfreqs,
                self.hidden_dim,
            )

        next_state.append(state_b)

    return torch.stack(next_state, dim=0)

def _forward_dense_phase_update(
    self,
    raw_x: torch.Tensor,
    mag: torch.Tensor,
    ang: torch.Tensor,
    gate_mask: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Dense batched version of the previous per-batch loop.

    Parameters
    ----------
    raw_x : tensor, shape [B, C]
        Raw signal at one time step.

    mag : tensor, shape [B, E, F]
        Magnitude features.

    ang : tensor, shape [B, E, F]
        Angle features.

    gate_mask : tensor, shape [B, E, F]
        Boolean or 0/1 mask saying which edge-frequency pairs are active.

    state :
        If self.state_mode == "per_node":
            shape [B, C, H]
        Else:
            shape [B, C, F, H]

    Returns
    -------
    next_state :
        Same shape as state.
    """
    batch_size = state.shape[0]
    device = state.device
    dtype = state.dtype
    num_edges = self.src_idx.numel()

    gate = gate_mask.to(dtype=dtype)

    features = []

    if self.use_mag:
        # [B, E, F] -> [B, E, F, 1]
        features.append(mag.unsqueeze(-1))

    if self.use_ang:
        # [B, E, F] -> [B, E, F, 1]
        features.append(ang.unsqueeze(-1))

    if self.use_raw:
        # raw_x: [B, C]
        src_raw = raw_x[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
        dst_raw = raw_x[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)

        # [B, E, 1, 1] -> [B, E, F, 1]
        src_raw = src_raw.expand(batch_size, num_edges, self.nfreqs, 1)
        dst_raw = dst_raw.expand(batch_size, num_edges, self.nfreqs, 1)

        features.append(src_raw)
        features.append(dst_raw)

    if self.state_mode == "per_node":
        if self.use_state_src:
            # state[:, self.src_idx, :] gives [B, E, H]
            # unsqueeze -> [B, E, 1, H]
            # expand -> [B, E, F, H]
            src_state = state[:, self.src_idx, :].unsqueeze(2)
            src_state = src_state.expand(
                batch_size,
                num_edges,
                self.nfreqs,
                self.hidden_dim,
            )
            features.append(src_state)

        if self.use_state_dst:
            dst_state = state[:, self.dst_idx, :].unsqueeze(2)
            dst_state = dst_state.expand(
                batch_size,
                num_edges,
                self.nfreqs,
                self.hidden_dim,
            )
            features.append(dst_state)

    else:
        if self.use_state_src:
            # [B, E, F, H]
            src_state = state[:, self.src_idx, :, :]
            features.append(src_state)

        if self.use_state_dst:
            # [B, E, F, H]
            dst_state = state[:, self.dst_idx, :, :]
            features.append(dst_state)

    payload = torch.cat(features, dim=-1)

    # [B, E, F, input_dim] -> [B, E, F, H]
    msg = self.message_mlp(payload)

    # Zero out inactive edge-frequency pairs.
    #
    # This matches the previous sparse version if gate_mask is 0/1.
    msg = msg * gate.unsqueeze(-1)

    if self.state_mode == "per_node":
        # Sum over frequency first:
        # [B, E, F, H] -> [B, E, H]
        msg_sum_f = msg.sum(dim=2)

        # Aggregate edges into destination nodes:
        # [B, E, H] -> [B, C, H]
        agg = self._aggregate_per_node(msg_sum_f)

        # GRUCell over all batch-node pairs:
        # [B, C, H] -> [B*C, H]
        state_flat = state.reshape(batch_size * self.n_channels, self.hidden_dim)
        agg_flat = agg.reshape(batch_size * self.n_channels, self.hidden_dim)

        next_state = self.state_cell(agg_flat, state_flat).view(
            batch_size,
            self.n_channels,
            self.hidden_dim,
        )

    else:
        # Aggregate edge-frequency messages into destination node-frequency slots:
        # [B, E, F, H] -> [B, C, F, H]
        agg = self._aggregate_per_node_per_freq(msg)

        # GRUCell over all batch-node-frequency pairs:
        # [B, C, F, H] -> [B*C*F, H]
        state_flat = state.reshape(
            batch_size * self.n_channels * self.nfreqs,
            self.hidden_dim,
        )
        agg_flat = agg.reshape(
            batch_size * self.n_channels * self.nfreqs,
            self.hidden_dim,
        )

        next_state = self.state_cell(agg_flat, state_flat).view(
            batch_size,
            self.n_channels,
            self.nfreqs,
            self.hidden_dim,
        )

    return next_state, agg

@torch.no_grad()
def _compare_original_and_batched(self, state, gate_mask, mag, ang, raw_t):
    """
    Runs both implementations and compares the produced next states.
    """
    next_original = _original_next_state(
        self=self,
        state=state,
        gate_mask=gate_mask,
        mag=mag,
        ang=ang,
        raw_t=raw_t,
    )

    next_batched, _ = _forward_dense_phase_update(
        self=self,
        state=state,
        gate_mask=gate_mask,
        mag=mag,
        ang=ang,
        raw_x=raw_t,
    )

    print("next_original.shape:", next_original.shape)
    print("next_batched.shape: ", next_batched.shape)

    diff = (next_original - next_batched).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    print("max rel diff:", (diff / next_original.abs().clamp_min(1e-8)).max().item())
    print("mean rel diff:", (diff / next_original.abs().clamp_min(1e-8)).mean().item())

    print(
        "allclose:",
        torch.allclose(
            next_original,
            next_batched,
            rtol=1e-5,
            atol=1e-6,
        ),
    )

    return next_original, next_batched

class WCTPhaseGNNCore(nn.Module):
    """Torch core for level-0 phase-gated WCT message passing (sparse edges)."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
        state_mode: str = "per_node",
        use_mag: bool = True,
        use_ang: bool = True,
        use_raw: bool = True,
        use_state_src: bool = True,
        use_state_dst: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if state_mode not in {"per_node", "per_node_per_freq"}:
            raise ValueError("state_mode must be one of {'per_node', 'per_node_per_freq'}")
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.coherence_threshold = float(coherence_threshold)
        self.phase_threshold_rad = math.radians(phase_threshold_deg)
        self.window_size = int(window_size)
        self.state_mode = state_mode
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
            batch_size,
            self.n_channels,
            hidden_dim,
            device=device,
            dtype=msg.dtype,
        )

        agg.index_add_(1, self.dst_idx, msg)

        return agg

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Forward pass.

        Parameters
        ----------
        raw_x : tensor, shape (B, C, T)
        w_real : tensor, shape (B, C, T, F)
        w_imag : tensor, shape (B, C, T, F)
        freqs : tensor, shape (F,)
        """
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}."
            )

        device = raw_x.device

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

        window_step = self.window_size
        for window_start in range(0, n_time - self.window_size + 1, window_step):
            window_end = window_start + self.window_size
            t_center = window_start + (self.window_size // 2)

            src_r = w_real[:, self.src_idx, window_start:window_end, :]
            src_i = w_imag[:, self.src_idx, window_start:window_end, :]
            dst_r = w_real[:, self.dst_idx, window_start:window_end, :]
            dst_i = w_imag[:, self.dst_idx, window_start:window_end, :]

            mean_mag, mean_phase, coh = _compute_wct_window_features(
                src_r,
                src_i,
                dst_r,
                dst_i,
            )
            gate_mask = (coh > self.coherence_threshold) & (
                mean_phase > self.phase_threshold_rad
            )

            gate_sum += float(gate_mask.sum().item())
            gate_count += float(gate_mask.numel())

            mean_mag = torch.nan_to_num(mean_mag, nan=0.0, posinf=0.0, neginf=0.0)
            mean_phase = torch.nan_to_num(mean_phase, nan=0.0, posinf=0.0, neginf=0.0)

            raw_t = raw_x[:, :, t_center]

            state, _ = _forward_dense_phase_update(
                self,
                raw_x=raw_t,
                mag=mean_mag,
                ang=mean_phase,
                gate_mask=gate_mask,
                state=state,
            )

        if self.state_mode == "per_node":
            pooled = state.mean(dim=1)
        else:
            pooled = state.mean(dim=(1, 2))

        logits = self.classifier(pooled)
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return logits, edge_density


class WCTPhaseGNNV2Core(nn.Module):
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
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
        use_raw_in_message: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if encoder_dropout is not None:
            if float(encoder_dropout) < 0.0 or float(encoder_dropout) >= 1.0:
                raise ValueError("encoder_dropout must be in [0.0, 1.0), or None.")
        if gru_input_dropout is not None:
            if float(gru_input_dropout) < 0.0 or float(gru_input_dropout) >= 1.0:
                raise ValueError("gru_input_dropout must be in [0.0, 1.0), or None.")
        if readout_dropout is not None:
            if float(readout_dropout) < 0.0 or float(readout_dropout) >= 1.0:
                raise ValueError("readout_dropout must be in [0.0, 1.0), or None.")
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")

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
        self.coherence_threshold = float(coherence_threshold)
        self.phase_threshold_rad = math.radians(phase_threshold_deg)
        self.window_size = int(window_size)
        self.use_raw_in_message = use_raw_in_message

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

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Forward pass.

        Parameters
        ----------
        raw_x : tensor, shape (B, C, T)
        w_real : tensor, shape (B, C, T, F)
        w_imag : tensor, shape (B, C, T, F)
        freqs : tensor, shape (F,)
        """
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}."
            )

        enc = raw_x.reshape(batch_size * n_channels, 1, n_time)
        enc = self.channel_encoder(enc)
        enc = enc.reshape(batch_size, n_channels, self.encoder_dim, n_time).permute(0, 1, 3, 2)

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

        window_step = self.window_size
        for window_start in range(0, n_time - self.window_size + 1, window_step):
            window_end = window_start + self.window_size
            t_center = window_start + (self.window_size // 2)

            src_r = w_real[:, self.src_idx, window_start:window_end, :]
            src_i = w_imag[:, self.src_idx, window_start:window_end, :]
            dst_r = w_real[:, self.dst_idx, window_start:window_end, :]
            dst_i = w_imag[:, self.dst_idx, window_start:window_end, :]

            xwt_real = src_r * dst_r + src_i * dst_i
            xwt_imag = src_i * dst_r - src_r * dst_i

            xwt_mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
            xwt_mag = torch.nan_to_num(xwt_mag, nan=0.0, posinf=0.0, neginf=0.0)
            xwt_mag_log = torch.log1p(xwt_mag)

            ang = torch.atan2(xwt_imag, xwt_real)
            ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
            delta = torch.atan2(torch.sin(ang), torch.cos(ang))

            cross = torch.complex(xwt_real, xwt_imag)
            auto1 = src_r * src_r + src_i * src_i
            auto2 = dst_r * dst_r + dst_i * dst_i
            mean_cross = torch.mean(cross, dim=2)
            mean_auto1 = torch.mean(auto1, dim=2)
            mean_auto2 = torch.mean(auto2, dim=2)
            coh = (mean_cross.abs() ** 2) / (mean_auto1 * mean_auto2 + 1e-12)
            mean_phase = torch.mean(delta, dim=2)

            gate_mask = (coh > self.coherence_threshold) & (
                mean_phase > self.phase_threshold_rad
            )

            gate_sum += float(gate_mask.sum().item())
            gate_count += float(gate_mask.numel())

            xwt_mag_log = torch.mean(xwt_mag_log, dim=2)

            freq_state = self.state_to_freq_proj(state).reshape(
                batch_size, self.n_channels, self.nfreqs, self.message_dim
            )
            raw_t = raw_x[:, :, t_center]
            next_state = []

            for b in range(batch_size):
                active_idx = torch.nonzero(gate_mask[b], as_tuple=False)
                if active_idx.numel() == 0:
                    agg_b = torch.zeros(
                        self.n_channels, self.nfreqs, self.message_dim, device=raw_x.device
                    )
                    if self.use_local_residual:
                        enc_t = enc[b, :, t_center, :]
                        if self.local_enc_proj is None:
                            raise RuntimeError("local_enc_proj is not initialized.")
                        local_enc_term = self.local_enc_proj(enc_t).reshape(
                            self.n_channels, self.nfreqs, self.message_dim
                        )
                        update_in_freq = agg_b + local_enc_term
                    else:
                        update_in_freq = agg_b
                    update_in = self.gru_input_proj(
                        update_in_freq.reshape(self.n_channels, self.nfreqs * self.message_dim)
                    )
                    if self.gru_input_dropout_layer is not None:
                        update_in = self.gru_input_dropout_layer(update_in)
                    state_b = self.state_cell(update_in, state[b])
                    next_state.append(state_b)
                    continue

                edge_idx = active_idx[:, 0]
                freq_idx = active_idx[:, 1]

                src_state = freq_state[b, self.src_idx[edge_idx], freq_idx, :]
                dst_state = freq_state[b, self.dst_idx[edge_idx], freq_idx, :]
                feats = [xwt_mag_log[b, edge_idx, freq_idx].unsqueeze(-1), src_state, dst_state]

                if self.use_raw_in_message:
                    src_raw = raw_t[b, self.src_idx[edge_idx]].unsqueeze(-1)
                    dst_raw = raw_t[b, self.dst_idx[edge_idx]].unsqueeze(-1)
                    feats.extend([src_raw, dst_raw])

                message_in = torch.cat(feats, dim=-1)
                msg = self.message_mlp(message_in)

                agg_b = torch.zeros(
                    self.n_channels * self.nfreqs,
                    self.message_dim,
                    device=raw_x.device,
                    dtype=msg.dtype,
                )
                flat_idx = self.dst_idx[edge_idx] * self.nfreqs + freq_idx
                agg_b.index_add_(0, flat_idx, msg)
                agg_b = agg_b.view(self.n_channels, self.nfreqs, self.message_dim)

                if self.use_local_residual:
                    enc_t = enc[b, :, t_center, :]
                    if self.local_enc_proj is None:
                        raise RuntimeError("local_enc_proj is not initialized.")
                    local_enc_term = self.local_enc_proj(enc_t).reshape(
                        self.n_channels, self.nfreqs, self.message_dim
                    )
                    update_in_freq = agg_b + local_enc_term
                else:
                    update_in_freq = agg_b

                update_in = self.gru_input_proj(
                    update_in_freq.reshape(self.n_channels, self.nfreqs * self.message_dim)
                )
                if self.gru_input_dropout_layer is not None:
                    update_in = self.gru_input_dropout_layer(update_in)
                state_b = self.state_cell(update_in, state[b])
                next_state.append(state_b)

            state = torch.stack(next_state, dim=0)

            pooled_nodes = state.mean(dim=1)
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


class WCTPhaseGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around the level-0 WCT phase GNN core."""

    model_label = "WCT-V1"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 48,
        cwt_resample_n_time: int | None = None,
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
        state_mode: str = "per_node",
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
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.window_size = window_size
        self.state_mode = state_mode
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
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            normalize_input=normalize_input,
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

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> WCTPhaseGNNCore:
        return WCTPhaseGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            coherence_threshold=self.coherence_threshold,
            phase_threshold_deg=self.phase_threshold_deg,
            window_size=self.window_size,
            state_mode=self.state_mode,
            use_mag=self.use_mag,
            use_ang=self.use_ang,
            use_raw=self.use_raw,
            use_state_src=self.use_state_src,
            use_state_dst=self.use_state_dst,
            **kwargs,
        )


class WCTPhaseGNNV2Classifier(_BaseCWTGNNClassifier):
    """V2 sklearn/MOABB wrapper with channel-local encoder and freq-indexed state."""

    _estimator_type = "classifier"
    model_label = "WCT-V2"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 32,
        cwt_resample_n_time: int | None = None,
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
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
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.window_size = window_size
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
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            normalize_input=normalize_input,
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

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> WCTPhaseGNNV2Core:
        return WCTPhaseGNNV2Core(
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
            coherence_threshold=self.coherence_threshold,
            phase_threshold_deg=self.phase_threshold_deg,
            window_size=self.window_size,
            use_raw_in_message=self.use_raw_in_message,
            **kwargs,
        )
