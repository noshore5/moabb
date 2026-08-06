"""Sparse/event-based WCT evidence GNN classifier.

Instead of pooling coherence into fixed time windows (as WCTEvidenceGNN
does), this computes coherence + phase at full time resolution, thresholds
them, and CONSOLIDATES temporally-adjacent surviving samples (per channel
pair, per frequency bin) into region-level "events" -- one event per burst,
not one per sample. Each event carries (timestamp, frequency, magnitude,
sin/cos(angle)) plus a learned per-channel signal embedding for its source
and destination channel, and is routed into its destination node's evidence
via the same graph topology as WCTEvidenceGNN.

Validated in exploratory testing (BNCI2014-001, cross-session, canonical
4-subject run via run_canonical_setup.py): subj1=0.801 subj2=0.557 subj3=0.947
subj4=0.539, pipeline mean=0.711. subj2/subj4 sitting near chance (0.5) is a
property of those subjects, not this pipeline specifically -- EEGNet (100
epochs) gets 0.603 on subject 2 too. Earlier single-subject number (0.750 on
subject 1) is superseded by the above; not yet validated on subjects 5-9.
See ChannelSignalEncoder's docstring below for the receptive-field fix
(channel_encoder_dilation) that this accuracy depends on.

This reuses WCTEvidenceGNNCore's buffers (src_idx/dst_idx from
ordered_pair_indices) and its (non-trainable) coherence/phase computation
methods (_full_edge_wct_maps, _smooth_wct_maps) via subclassing, so the
underlying wavelet math is identical to the windowed pipeline -- only what
happens after coherence is computed differs.

Two fixes validated via debug_plots/edge0_*.png before being wired in here:
  - cwt_resample_n_time now defaults to None (native resolution). Resampling
    the complex CWT coefficients via scipy.signal.resample (the old default,
    200) was destroying real signal above ~n_time/(2*trial_secs) Hz -- a
    clean 30Hz test tone measured 0.81 magnitude natively vs 0.006 after
    resample to 200 samples on a ~4s trial.
  - SparseEvidenceGNNCore._build_sparse_events now ANDs a cone-of-influence
    mask into its gate (see _coi_valid_mask): fcwt.cwt() returns no COI, and
    without it, events could be built from time/freq cells where the wavelet
    ran off the edge of the trial.
  - _build_sparse_events (coherence/gate/COI/run-consolidation) is entirely
    non-trainable and deterministic given fixed CWT features, yet forward()
    used to call it on every (batch, epoch) -- profiling showed it was 94.8%
    of forward()'s time. SparseEvidenceGNNClassifier._prepare_features now
    calls it once per trial (see _precompute_sparse_events) and forward()
    only does the trainable part (channel_encoder + sparse_message_mlp +
    sparse_classifier) every step. Measured ~9x faster end to end.
The kernel_size=(5,3) smoothing is deliberately left at this value: at native
resolution (n_time~1001, 4.0ms/sample) it spans only ~20ms of time smoothing,
vs ~100ms if the kernel were widened to (25,3) to match the smoothing width
the old cwt_resample_n_time=200 pipeline had by accident. Re-tested (25,3)
against (5,3) after the channel_encoder_dilation fix above -- 0.8008 vs
0.7991 on subject 1, still noise-level -- so widening the kernel buys nothing
and (5,3) is kept for the time/frequency resolution this sparse-event
architecture is built to exploit.
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
    version of WCTEvidenceGNNCore's feature_conv, not a copy of it.

    `dilation` controls the receptive field in real time, independent of
    input length. Two stacked kernel_size=9 convs give a fixed 17-SAMPLE
    receptive field (RF = 1 + (9-1) + (9-1)) regardless of sampling rate --
    at native ~250Hz that's only ~68ms, shorter than a single mu-band cycle
    (8-12Hz, ~83-125ms), so at native resolution this encoder was
    architecturally blind to oscillatory envelope shape and could only see
    sub-cycle sample texture (empirically confirmed: resampling ONLY raw_x
    to 200 samples -- i.e. giving this encoder the same 17-sample window a
    ~5x larger real-time span -- recovered accuracy from ~0.76 to ~0.80
    while leaving coherence/COI untouched at native resolution). Dilation
    grows the *time* the kernel spans without growing kernel size (avoiding
    both extra parameters and a large-kernel's own smoothing effect), and
    without touching/resampling the input signal itself (so no real
    high-frequency content is discarded, unlike naively downsampling raw_x).
    dilation=5 with kernel_size=9 gives RF = 1 + 8*5 + 8*5 = 81 samples =
    ~324ms at 250Hz -- ~3.2 cycles of an 8-12Hz mu rhythm."""

    def __init__(self, embed_dim: int, dilation: int = 1):
        super().__init__()
        kernel_size = 9
        pad = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
            nn.Conv1d(8, embed_dim, kernel_size=kernel_size, padding=pad, dilation=dilation), nn.GELU(),
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

    # Cone-of-influence wavelet parameter. Must match coherence_utils.transform's
    # hardcoded `fcwt.Morlet(2.0)` -- fcwt.cpp's own edge-of-support formula is
    # getSupport(scale) = int(fb*scale*3.0), scale == sampling_rate/freq for this
    # wavelet. fcwt.cwt() returns no COI itself; this reproduces it from source.
    _COI_WAVELET_FB = 2.0

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
        sampling_rate: int = 250,
        coi_enabled: bool = True,
        channel_encoder_dilation: int = 1,
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
        self.channel_encoder = ChannelSignalEncoder(
            channel_embed_dim, dilation=channel_encoder_dilation
        )
        message_in = 5 + 2 * channel_embed_dim  # timestamp, freq, mag, sin, cos + src/dst embeds
        self.sparse_message_mlp = nn.Sequential(
            nn.Linear(message_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sparse_classifier = nn.Linear(n_channels * hidden_dim, n_classes)
        self._freq_lo = None  # set on first forward() call from observed freqs
        self._freq_hi = None
        self.sampling_rate = sampling_rate  # needed for the COI mask below
        # Diagnostic toggle: COI support scales as 1/freq, so at this
        # pipeline's trial length it disproportionately crops the low-freq
        # (mu-band, most discriminative for motor imagery) end of the
        # spectrum -- up to ~37% of the trial at 8Hz vs ~8% at 35Hz. Measured
        # to shift the surviving events' frequency mix away from mu-band
        # (mu share 20.6%->17.6% of events, COI off->on, at native res).
        # Kept as a real constructor arg (not a monkeypatch) so it's a
        # legitimate, re-runnable pipeline configuration, not just a
        # debugging hack.
        self.coi_enabled = coi_enabled

    def _coi_valid_mask(self, freqs_batched: torch.Tensor, n_time_in: int, T_out: int) -> torch.Tensor:
        """Cone-of-influence validity mask, aligned to coh/phase's time axis.

        NOT computed anywhere upstream in this pipeline (fcwt.cwt returns no
        COI array). Assumes native-resolution CWT coefficients -- i.e. the
        classifier's cwt_resample_n_time=None, so `n_time_in` (w_real's own
        time axis) already equals the original CWT's sample count and no
        extra rescale factor is needed. If cwt_resample_n_time is set to a
        non-None value upstream, this mask will be wrong (see the warning
        raised in SparseEvidenceGNNClassifier.__init__).
        """
        device, dtype = freqs_batched.device, freqs_batched.dtype
        scale = self.sampling_rate / freqs_batched  # [B, F], samples
        support = torch.floor(self._COI_WAVELET_FB * scale * 3.0)  # [B, F]
        time_offset = (self.smooth_kernel_size[0] - 1) // 2
        t_idx = torch.arange(T_out, device=device, dtype=dtype).view(1, T_out, 1) + time_offset
        support_b = support.unsqueeze(1)  # [B, 1, F]
        valid = (t_idx >= support_b) & (t_idx < (n_time_in - support_b))  # [B, T_out, F]
        return valid.unsqueeze(1)  # [B, 1, T_out, F] -- broadcasts over the edge dim

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
            if self.coi_enabled:
                coi_valid = self._coi_valid_mask(
                    freqs_batched, n_time_in=w_real.shape[2], T_out=coh.shape[2]
                )
                gate = gate & coi_valid

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

    def compute_events(self, w_real, w_imag, freqs):
        """Runs the full non-trainable pipeline (cross-spectrum -> smoothing
        -> gate -> COI -> run-consolidation) and returns padded per-trial
        events. This is what forward() used to do on every call; profiling
        showed it was 94.8% of forward()'s time despite being a deterministic
        function of these (fixed, precomputed) CWT features -- it's now
        called once per trial by SparseEvidenceGNNClassifier._prepare_features
        instead of once per (batch, epoch). Kept as its own method so it's
        still directly callable for debugging (e.g. the earlier
        debug_plots/edge0_*.png scripts call the pieces of this directly).
        """
        batch_size = w_real.shape[0]
        freqs_batched = self._batched_freqs(freqs, batch_size)
        if self._freq_lo is None:
            self._freq_lo = float(freqs_batched.min().item())
            self._freq_hi = float(freqs_batched.max().item())
        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size, sigma=self.smooth_kernel_sigma,
            pad_h=0, device=w_real.device, dtype=w_real.dtype,
        )
        events_padded, src_padded, dst_padded, valid_mask, _ = self._build_sparse_events(
            w_real, w_imag, freqs_batched, smooth_kernel_and_pad
        )
        return events_padded, src_padded, dst_padded, valid_mask

    def forward(self, raw_x, events_padded, src_padded, dst_padded, valid_mask):
        """Trainable-only forward pass over PRECOMPUTED sparse events (see
        compute_events()). `to_float_tensors` upstream casts everything to
        float for DataLoader/TensorDataset batching, so src_padded/dst_padded/
        valid_mask arrive as float and need casting back here."""
        src_padded = src_padded.long()
        dst_padded = dst_padded.long()
        valid_mask = valid_mask.bool()
        batch_size = raw_x.shape[0]

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
        # matches the old event_density = n_runs / max(B*E*F, 1) exactly:
        # valid_mask.sum() over a batch IS that batch's n_runs.
        event_density = float(valid_mask.sum().item()) / max(
            batch_size * self.src_idx.numel() * self.nfreqs, 1
        )
        return logits, event_density


class SparseEvidenceGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around SparseEvidenceGNNCore."""

    model_label = "Sparse-Evidence"
    # This model's forward() aux value is n_runs / (batch*edges*freqs) -- an
    # unbounded average burst-count-per-row, NOT a bounded [0,1] fraction
    # like WCTEvidenceGNN's edge_density. Override the shared log label so
    # it doesn't misleadingly read as a percentage (it can exceed 1.0).
    aux_metric_name = "bursts_per_row"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 16,
        # None = native resolution (no post-CWT resample). A non-None value
        # here previously destroyed real signal above ~n_time/(2*trial_secs)
        # Hz via scipy.signal.resample on the complex coefficients themselves
        # (verified with a clean 30Hz test tone: magnitude 0.81 natively ->
        # 0.006 after resample to 200). It would also make the COI mask in
        # SparseEvidenceGNNCore wrong (see the warning below).
        cwt_resample_n_time: int | None = None,
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
        coi_enabled: bool = True,
        # Diagnostic: independently resample ONLY raw_x (the signal fed to
        # ChannelSignalEncoder), leaving w_real/w_imag/coherence/events at
        # whatever resolution cwt_resample_n_time implies. Unlike
        # cwt_resample_n_time, this is safe to set alongside native-res
        # coherence -- compute_events() never touches raw_x, so this cannot
        # corrupt the COI mask or coherence estimate. Added specifically to
        # isolate whether the old pipeline's ~0.80 (vs ~0.76 now) traces to
        # ChannelSignalEncoder seeing a resampled (T=200) vs native (T~1001)
        # raw signal, independent of everything already ruled out in the
        # coherence/event pathway (density, COI, kernel, thresholds all
        # tested with no effect -- see run_wct_gnn.py's _make_sparse_evidence_gnn).
        raw_x_resample_n_time: int | None = None,
        # Real fix for the same finding raw_x_resample_n_time diagnosed:
        # ChannelSignalEncoder's two kernel_size=9 convs give a fixed
        # 17-sample receptive field, too short (~68ms at native 250Hz) to
        # span even one mu-band cycle (~83-125ms). Dilation grows that
        # window in real time without resampling/discarding any of the raw
        # signal (unlike raw_x_resample_n_time) and without growing the
        # kernel's parameter count. See ChannelSignalEncoder's docstring.
        channel_encoder_dilation: int = 1,
        channel_subset: list[int] | list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = channel_embed_dim
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.smooth_kernel_size = smooth_kernel_size
        self.coi_enabled = coi_enabled
        self.raw_x_resample_n_time = raw_x_resample_n_time
        self.channel_encoder_dilation = channel_encoder_dilation
        if cwt_resample_n_time is not None:
            import warnings
            warnings.warn(
                "SparseEvidenceGNNClassifier(cwt_resample_n_time=...) is set to a "
                "non-None value. SparseEvidenceGNNCore's COI mask assumes native "
                "resolution (cwt_resample_n_time=None) and will be computed "
                "incorrectly otherwise. Resampling the CWT coefficients also "
                "destroys real high-frequency signal (see class docstring / "
                "SparseEvidenceGNNCore._coi_valid_mask).",
                stacklevel=2,
            )
        if noise_augmentation_enabled:
            raise ValueError(
                "SparseEvidenceGNNClassifier does not support "
                "noise_augmentation_enabled=True together with its event-caching "
                "optimization: _build_sparse_events now runs once per trial during "
                "_prepare_features rather than once per training batch (see "
                "_precompute_sparse_events), so live per-batch CWT-domain noise "
                "injection (_augment_train_batch_inputs / augment_paired_cwt_batch) "
                "would have no effect on the cached events. Disable noise "
                "augmentation to use this classifier."
            )
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

    def _prepare_features(self, X, *, fit: bool, train_idx=None):
        raw_x, w_real, w_imag, freqs = super()._prepare_features(X, fit=fit, train_idx=train_idx)
        # Sparse events are computed from w_real/w_imag/freqs alone (see
        # compute_events -- raw_x never enters that path), so resampling
        # raw_x here has zero effect on coherence/COI/events. This lets us
        # test ChannelSignalEncoder's sensitivity to raw-signal resolution
        # in isolation from the (already-ruled-out) coherence pathway.
        if self.raw_x_resample_n_time is not None and int(
            self.raw_x_resample_n_time
        ) != int(raw_x.shape[2]):
            import numpy as np
            from scipy.signal import resample

            raw_np = resample(raw_x.numpy(), int(self.raw_x_resample_n_time), axis=2)
            raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
            raw_x = torch.from_numpy(raw_np).float()
        events_padded, src_padded, dst_padded, valid_mask = self._precompute_sparse_events(
            raw_x, w_real, w_imag, freqs
        )
        return raw_x, events_padded, src_padded, dst_padded, valid_mask

    def _precompute_sparse_events(self, raw_x, w_real, w_imag, freqs):
        """Runs SparseEvidenceGNNCore.compute_events (non-trainable: cross-
        spectrum, smoothing, gate, COI, run-consolidation) ONCE per trial,
        chunked to bound memory, instead of once per (batch, epoch) inside
        forward(). Profiling: this stage is 94.8% of a forward() call's time
        despite depending only on fixed CWT features + fixed hyperparameters,
        never on trainable weights -- so its output is identical on every
        epoch for a given trial. Padding (event count) is computed once here
        across the whole dataset, not per training mini-batch as before.

        Uses a throwaway SparseEvidenceGNNCore purely for its non-trainable
        buffers/thresholds (src_idx/dst_idx, coherence_threshold, etc.) --
        its trainable submodules are constructed but never used here. Its
        random init is RNG-isolated (torch.random.fork_rng) so it has no
        effect on the real model built later in _build_model_from_features.
        """
        n_channels = int(raw_x.shape[1])
        n_samples = int(raw_x.shape[0])
        with torch.random.fork_rng(devices=[]):
            helper = self._build_model(n_channels=n_channels, n_classes=2)
        helper.eval()
        helper._freq_lo = float(freqs.min().item())
        helper._freq_hi = float(freqs.max().item())

        # Chunk by trials independently of self.batch_size (which governs
        # training, not this one-time precompute). _smooth_wct_maps's im2col
        # buffers scale ~linearly with trials-per-chunk * edges * 4; even
        # with the separable-conv fix, a full batch_size=16 chunk at this
        # pipeline's shapes (9 channels -> 72 edges, T~1001, nfreqs=16)
        # measured ~8GB peak RSS for large kernels like (25,3) -- too close
        # to this machine's ~17GB RAM. Capping at 4 trials/chunk keeps peak
        # RSS in the ~2GB range regardless of self.batch_size.
        chunk = max(1, min(int(self.batch_size), 4))
        all_events, all_src, all_dst, all_valid = [], [], [], []
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            with torch.no_grad():
                ev, sp, dp, vm = helper.compute_events(
                    w_real[start:end], w_imag[start:end], freqs[start:end]
                )
            all_events.append(ev)
            all_src.append(sp)
            all_dst.append(dp)
            all_valid.append(vm)

        max_count = max(t.shape[1] for t in all_events)

        def pad(t, fill=0):
            if t.shape[1] == max_count:
                return t
            pad_shape = list(t.shape)
            pad_shape[1] = max_count - t.shape[1]
            filler = torch.full(pad_shape, fill, dtype=t.dtype, device=t.device)
            return torch.cat([t, filler], dim=1)

        events_padded = torch.cat([pad(t) for t in all_events], dim=0)
        src_padded = torch.cat([pad(t) for t in all_src], dim=0)
        dst_padded = torch.cat([pad(t) for t in all_dst], dim=0)
        valid_mask = torch.cat([pad(t, fill=False) for t in all_valid], dim=0)
        return events_padded, src_padded, dst_padded, valid_mask

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
            sampling_rate=self.sampling_rate,
            coi_enabled=self.coi_enabled,
            channel_encoder_dilation=self.channel_encoder_dilation,
            **kwargs,
        )
