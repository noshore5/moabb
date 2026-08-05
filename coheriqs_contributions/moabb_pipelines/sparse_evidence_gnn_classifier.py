"""Sparse/event-based WCT evidence GNN classifier.

Instead of pooling coherence into fixed time windows (as WCTEvidenceGNN
does), this computes coherence + phase at full time resolution, thresholds
them, and CONSOLIDATES temporally-adjacent surviving samples (per channel
pair, per frequency bin) into region-level "events" -- one event per burst,
not one per sample. Each event carries (timestamp, frequency, magnitude,
sin/cos(angle)) plus a learned per-channel signal embedding for its source
and destination channel, and is routed into its destination node's evidence
via the same graph topology as WCTEvidenceGNN.

Validated in exploratory testing (subject 1, BNCI2014-001, cross-session):
mean test accuracy 0.750 vs WCTEvidenceGNN baseline 0.7135 (window_size=25)
and 0.753 (window_size=5) -- comparable to the best windowed result. Only
tested on subject 1 with default architecture params; not yet validated for
robustness across subjects the way the windowed pipeline's hyperparameters
were (see run_wct_gnn.py's PIPELINE_PARAM_GRIDS comments for that history).

This reuses WCTEvidenceGNNCore's buffers (src_idx/dst_idx from
ordered_pair_indices) and its (non-trainable) coherence/phase computation
methods (_full_edge_wct_maps, _smooth_wct_maps) via subclassing, so the
underlying wavelet math is identical to the windowed pipeline -- only what
happens after coherence is computed differs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from coheriqs_contributions.moabb_pipelines.common import make_gaussian_weight2d
    from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
        WCTEvidenceGNNCore,
    )
    from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import make_gaussian_weight2d
    from moabb_pipelines.wct_evidence_gnn_classifier import WCTEvidenceGNNCore
    from moabb_pipelines.xwt_phase_gnn_classifier import _BaseCWTGNNClassifier


class ChannelSignalEncoder(nn.Module):
    """Lightweight learned per-channel signature: gives each graph node an
    actual representation of its raw signal shape, not just the
    coherence/timing scalars that arrive on its edges. A much smaller
    version of WCTEvidenceGNNCore's feature_conv, not a copy of it."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4), nn.GELU(),
            nn.Conv1d(8, embed_dim, kernel_size=9, padding=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_time = raw_x.shape
        x = raw_x.reshape(batch_size * n_channels, 1, n_time)
        emb = self.net(x).squeeze(-1)
        return emb.reshape(batch_size, n_channels, -1)


class SparseEvidenceGNNCore(WCTEvidenceGNNCore):
    """Torch core: full-resolution coherence -> region-consolidated sparse
    events -> per-channel-conditioned message passing -> flatten -> classify.

    Subclasses WCTEvidenceGNNCore purely to reuse its edge-index buffers
    (src_idx/dst_idx) and its (parameter-free) coherence computation methods
    -- the inherited feature_conv/message_mlp/classifier submodules are
    constructed but never used by this class's forward(); only
    channel_encoder/sparse_message_mlp/sparse_classifier are.
    """

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        hidden_dim: int = 8,
        channel_embed_dim: int = 8,
        coherence_threshold: float = 0.5,
        phase_threshold_deg: float = 30.0,
        smooth_kernel_sigma: tuple[float | None, float | None] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (5, 3),
        model_init_seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            nfreqs=nfreqs,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            message_dim=hidden_dim,
            coherence_threshold=coherence_threshold,
            phase_threshold_deg=phase_threshold_deg,
            window_size=25,  # unused by this class's forward(); kept valid for super().__init__
            use_mag=False, use_ang=False, use_raw=False, use_freq=True, use_time=True,
            readout_mode="flatten", evidence_norm="active_slots",
            smooth_kernel_sigma=smooth_kernel_sigma,
            smooth_kernel_size=smooth_kernel_size,
            model_init_seed=model_init_seed,
        )
        self.channel_embed_dim = channel_embed_dim
        self.channel_encoder = ChannelSignalEncoder(channel_embed_dim)
        message_in = 5 + 2 * channel_embed_dim  # timestamp, freq, mag, sin, cos + src/dst embeds
        self.sparse_message_mlp = nn.Sequential(
            nn.Linear(message_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sparse_classifier = nn.Linear(n_channels * hidden_dim, n_classes)
        self._freq_lo = None  # set on first forward() call from observed freqs
        self._freq_hi = None

    def _build_sparse_events(self, w_real, w_imag, freqs_batched, smooth_kernel_and_pad):
        with torch.no_grad():
            _, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
                w_real, w_imag, freqs_batched, compute_mag=False
            )
            smooth_cross, coh, _ = self._smooth_wct_maps(
                xwt_real, xwt_imag, auto1, auto2, smooth_kernel_and_pad, stride=(1, 1)
            )
            phase = torch.angle(smooth_cross)
            gate = (coh > self.coherence_threshold) & (phase > self.phase_threshold_rad)

            B, E, T, F = gate.shape
            gate_r = gate.permute(0, 1, 3, 2).reshape(B * E * F, T)
            coh_r = coh.permute(0, 1, 3, 2).reshape(B * E * F, T)
            phase_r = phase.permute(0, 1, 3, 2).reshape(B * E * F, T)
            R = gate_r.shape[0]

            if gate_r.any():
                starts = torch.zeros_like(gate_r)
                starts[:, 0] = gate_r[:, 0]
                starts[:, 1:] = gate_r[:, 1:] & (~gate_r[:, :-1])
                run_id_local = starts.cumsum(dim=1)
                row_offset = torch.arange(R, device=gate.device).view(R, 1) * (T + 1)
                global_run_id = run_id_local + row_offset

                valid_pos = gate_r.nonzero(as_tuple=False)
                row_idx, time_idx = valid_pos.unbind(1)
                run_ids_at_valid = global_run_id[row_idx, time_idx]
                mag_at_valid = coh_r[row_idx, time_idx]
                angle_at_valid = phase_r[row_idx, time_idx]
                time_at_valid = time_idx.to(coh.dtype)

                unique_runs, inverse = torch.unique(run_ids_at_valid, return_inverse=True)
                n_runs = unique_runs.shape[0]

                def scatter_mean(values):
                    s = torch.zeros(n_runs, dtype=coh.dtype, device=gate.device).index_add_(
                        0, inverse, values
                    )
                    c = torch.zeros(n_runs, dtype=coh.dtype, device=gate.device).index_add_(
                        0, inverse, torch.ones_like(values)
                    )
                    return s / c

                mean_mag = scatter_mean(mag_at_valid)
                mean_angle = scatter_mean(angle_at_valid)
                mean_time = scatter_mean(time_at_valid)

                row_of_run = torch.zeros(n_runs, dtype=torch.long, device=gate.device)
                row_of_run[inverse] = row_idx

                b_of_run = row_of_run // (E * F)
                rem = row_of_run % (E * F)
                e_of_run = rem // F
                f_of_run = rem % F

                dst_node = self.dst_idx[e_of_run]
                src_node = self.src_idx[e_of_run]
                freq_vals_raw = freqs_batched[b_of_run, f_of_run]
                freq_vals = (freq_vals_raw - self._freq_lo) / max(
                    self._freq_hi - self._freq_lo, 1e-6
                )
                timestamp_vals = mean_time / float(max(T - 1, 1))
                b_idx = b_of_run

                event_features = torch.stack(
                    [timestamp_vals, freq_vals, mean_mag,
                     torch.sin(mean_angle), torch.cos(mean_angle)], dim=-1
                )
            else:
                n_runs = 0
                event_features = torch.zeros(0, 5, dtype=coh.dtype, device=gate.device)
                dst_node = torch.zeros(0, dtype=torch.long, device=gate.device)
                src_node = torch.zeros(0, dtype=torch.long, device=gate.device)
                b_idx = torch.zeros(0, dtype=torch.long, device=gate.device)

            counts = torch.bincount(b_idx, minlength=B)
            max_count = max(int(counts.max().item()) if n_runs > 0 else 1, 1)
            offsets = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=gate.device), counts.cumsum(0)[:-1]]
            )
            global_idx = torch.arange(n_runs, device=gate.device)
            pos_within_trial = global_idx - offsets[b_idx]

            events_padded = torch.zeros(B, max_count, 5, dtype=coh.dtype, device=gate.device)
            dst_padded = torch.zeros(B, max_count, dtype=torch.long, device=gate.device)
            src_padded = torch.zeros(B, max_count, dtype=torch.long, device=gate.device)
            valid_mask = torch.zeros(B, max_count, dtype=torch.bool, device=gate.device)
            events_padded[b_idx, pos_within_trial] = event_features
            dst_padded[b_idx, pos_within_trial] = dst_node
            src_padded[b_idx, pos_within_trial] = src_node
            valid_mask[b_idx, pos_within_trial] = True

        event_density = n_runs / max(B * E * F, 1)
        return events_padded, src_padded, dst_padded, valid_mask, event_density

    def forward(self, raw_x, w_real, w_imag, freqs):
        batch_size = raw_x.shape[0]
        freqs_batched = self._batched_freqs(freqs, batch_size)
        if self._freq_lo is None:
            self._freq_lo = float(freqs_batched.min().item())
            self._freq_hi = float(freqs_batched.max().item())

        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size, sigma=self.smooth_kernel_sigma,
            pad_h=0, device=raw_x.device, dtype=raw_x.dtype,
        )
        events_padded, src_padded, dst_padded, valid_mask, event_density = (
            self._build_sparse_events(w_real, w_imag, freqs_batched, smooth_kernel_and_pad)
        )

        channel_emb = self.channel_encoder(raw_x)
        max_count = events_padded.shape[1]
        batch_idx = torch.arange(batch_size, device=raw_x.device).unsqueeze(1).expand(-1, max_count)
        src_emb = channel_emb[batch_idx, src_padded]
        dst_emb = channel_emb[batch_idx, dst_padded]

        full_features = torch.cat([events_padded, src_emb, dst_emb], dim=-1)
        msg = self.sparse_message_mlp(full_features) * valid_mask.unsqueeze(-1).to(raw_x.dtype)

        evidence = torch.zeros(batch_size, self.n_channels, self.hidden_dim, dtype=raw_x.dtype, device=raw_x.device)
        evidence.scatter_add_(1, dst_padded.unsqueeze(-1).expand(-1, -1, self.hidden_dim), msg)
        active = torch.zeros(batch_size, self.n_channels, dtype=raw_x.dtype, device=raw_x.device)
        active.scatter_add_(1, dst_padded, valid_mask.to(raw_x.dtype))
        evidence = evidence / active.clamp_min(1.0).unsqueeze(-1)

        readout = evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
        logits = self.sparse_classifier(readout)
        return logits, event_density


class SparseEvidenceGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around SparseEvidenceGNNCore."""

    model_label = "Sparse-Evidence"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 16,
        cwt_resample_n_time: int | None = 200,
        coherence_threshold: float = 0.5,
        phase_threshold_deg: float = 30.0,
        hidden_dim: int = 8,
        channel_embed_dim: int = 8,
        epochs: int = 50,
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
        last_batch_min_ratio: float = 0.0,
        selector_alpha_val_update_rate: float = 1.0,
        optimizer_step_batch_size: int | None = None,
        optimizer_step_batch_mode: str = "credit",
        optimizer_step_remainder_policy: str = "flush",
        smooth_kernel_sigma: tuple[float | None, float | None] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (5, 3),
        channel_subset: list[int] | list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = channel_embed_dim
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.smooth_kernel_size = smooth_kernel_size
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
            last_batch_min_ratio=last_batch_min_ratio,
            selector_alpha_val_update_rate=selector_alpha_val_update_rate,
            optimizer_step_batch_size=optimizer_step_batch_size,
            optimizer_step_batch_mode=optimizer_step_batch_mode,
            optimizer_step_remainder_policy=optimizer_step_remainder_policy,
            channel_subset=channel_subset,
            verbose=verbose,
        )

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> SparseEvidenceGNNCore:
        raw_x = features[0] if isinstance(features, tuple) else features
        model = self._build_model(n_channels=int(raw_x.shape[1]), n_classes=n_classes, **kwargs)
        model.configure_summary_context(
            batch_size=int(self.batch_size),
            n_time=int(raw_x.shape[2]),
            dtype=raw_x.dtype,
            n_samples=int(raw_x.shape[0]),
        )
        return model

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> SparseEvidenceGNNCore:
        return SparseEvidenceGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            hidden_dim=self.hidden_dim,
            channel_embed_dim=self.channel_embed_dim,
            coherence_threshold=self.coherence_threshold,
            phase_threshold_deg=self.phase_threshold_deg,
            smooth_kernel_sigma=self.smooth_kernel_sigma,
            smooth_kernel_size=self.smooth_kernel_size,
            model_init_seed=self.seed,
            **kwargs,
        )
