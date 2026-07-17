"""Non-recurrent WCT evidence GNN classifier."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _compute_wct_window_features,
        _ordered_pair_indices,
    )
    from coheriqs_contributions.moabb_pipelines.torch_coherence_utils import (
        compute_coherence_fcwt,
    )
except ModuleNotFoundError:
    from moabb_pipelines.wct_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _compute_wct_window_features,
        _ordered_pair_indices,
    )
    from torch_coherence_utils import compute_coherence_fcwt


class WCTEvidenceGNNCore(nn.Module):
    """Torch core for windowed WCT message evidence accumulation."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        hidden_dim: int = 8,
        message_dim: int = 8,
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
        use_mag: bool = True,
        use_ang: bool = False,
        use_raw: bool = True,
        readout_mode: str = "mean",
        evidence_norm: str = "all_slots",
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")
        if readout_mode not in {"mean", "flatten"}:
            raise ValueError("readout_mode must be one of {'mean', 'flatten'}")
        if evidence_norm not in {"all_slots", "windows", "active_slots", "none"}:
            raise ValueError(
                "evidence_norm must be one of "
                "{'all_slots', 'windows', 'active_slots', 'none'}"
            )

        self.n_channels = n_channels
        self.nfreqs = nfreqs
        self.sampling_rate = sampling_rate
        self.lowest = lowest
        self.highest = highest
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.coherence_threshold = float(coherence_threshold)
        self.phase_threshold_rad = math.radians(phase_threshold_deg)
        self.window_size = int(window_size)
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.readout_mode = readout_mode
        self.evidence_norm = evidence_norm

        src_idx, dst_idx = _ordered_pair_indices(n_channels)
        self.register_buffer("src_idx", src_idx, persistent=False)
        self.register_buffer("dst_idx", dst_idx, persistent=False)

        payload_dim = 0
        if self.use_mag:
            payload_dim += 2  # binary indicator + normalized frequency
        if self.use_ang:
            payload_dim += 1
        if self.use_raw:
            payload_dim += 2
        if payload_dim == 0:
            raise ValueError("At least one payload component must be enabled.")

        self.message_mlp = nn.Sequential(
            nn.Linear(payload_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, hidden_dim),
        )
        readout_dim = hidden_dim * n_channels if readout_mode == "flatten" else hidden_dim
        self.classifier = nn.Linear(readout_dim, n_classes)

    def _aggregate_per_node(self, msg: torch.Tensor) -> torch.Tensor:
        """Aggregate [B, E, H] messages to [B, C, H] by destination."""
        batch_size, _, hidden_dim = msg.shape
        agg = torch.zeros(
            batch_size,
            self.n_channels,
            hidden_dim,
            device=msg.device,
            dtype=msg.dtype,
        )
        agg.index_add_(1, self.dst_idx, msg)
        return agg

    def _aggregate_active_slots_per_node(self, gate_mask: torch.Tensor) -> torch.Tensor:
        """Count active edge-frequency slots per destination node."""
        active_per_edge = gate_mask.to(dtype=torch.float32).sum(dim=2)
        counts = torch.zeros(
            gate_mask.shape[0],
            self.n_channels,
            device=gate_mask.device,
            dtype=torch.float32,
        )
        counts.index_add_(1, self.dst_idx, active_per_edge)
        return counts

    def build_graph(
        self,
        raw_x: torch.Tensor,
        sampling_rate: int,
    ) -> dict:
        """Build a sparse graph representation using fcwt-based coherence.

        Returned dict contains keys: `batch_size`, `n_channels`, `nfreqs`,
        and `windows` which is a list of per-window dicts. Each window dict
        contains `window_start`, `window_end`, `t_center` and `active`, where
        `active` is a list of length `batch_size` with tuples `(edge_idx, freq_idx)`
        representing active edge-frequency slots (both are 1-D `torch.LongTensor`).
        Only slots passing the coherence and phase thresholds are included.
        """
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        # Compute wavelet coherence once using fcwt
        coh_full, mean_phase_full = compute_coherence_fcwt(
            raw_x,
            sampling_rate,
            self.lowest,
            self.highest,
            self.nfreqs,
        )
        
        gate_mask_full = (coh_full > self.coherence_threshold) & (
            mean_phase_full > self.phase_threshold_rad
        )

        windows: list[dict] = []
        for window_start in range(0, n_time - self.window_size + 1, self.window_size):
            window_end = window_start + self.window_size
            t_center = window_start + (self.window_size // 2)

            # Use time-averaged gate_mask for this window
            window_gate_mask = gate_mask_full  # (B, E, F) already computed

            # For each batch element, collect active (edge,freq) pairs
            active_per_batch: list[tuple[torch.Tensor, torch.Tensor]] = []
            for b in range(batch_size):
                nz = torch.nonzero(window_gate_mask[b], as_tuple=False)
                if nz.numel() == 0:
                    active_per_batch.append((
                        torch.empty(0, dtype=torch.long, device=gate_mask_full.device),
                        torch.empty(0, dtype=torch.long, device=gate_mask_full.device),
                    ))
                else:
                    edge_idx = nz[:, 0].to(dtype=torch.long)
                    freq_idx = nz[:, 1].to(dtype=torch.long)
                    active_per_batch.append((edge_idx, freq_idx))

            windows.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "t_center": t_center,
                    "active": active_per_batch,
                }
            )

        return {
            "batch_size": batch_size,
            "n_channels": n_channels,
            "nfreqs": self.nfreqs,
            "windows": windows,
        }

    def forward(
        self,
        raw_x: torch.Tensor | None = None,
        w_real: torch.Tensor | None = None,
        w_imag: torch.Tensor | None = None,
        graph: dict | None = None,
    ) -> tuple[torch.Tensor, float]:
        # w_real and w_imag are accepted for API compatibility but ignored
        # (June18 computes coherence directly from raw_x using fcwt)
        if graph is None:
            if raw_x is None:
                raise ValueError("Either provide `graph` or `raw_x`.")
            batch_size, n_channels, n_time = raw_x.shape
            if n_channels != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")
            # build graph on the fly
            graph = self.build_graph(raw_x, self.sampling_rate)
        else:
            batch_size = graph.get("batch_size")
            n_channels = graph.get("n_channels")

        # sparse graph processing requires raw signal to extract time-domain raw values
        if graph is not None and self.use_raw and raw_x is None:
            raise ValueError("`raw_x` is required when `use_raw` is True and a sparse `graph` is provided.")

        # allocate evidence on correct device/dtype
        if raw_x is not None:
            dev = raw_x.device
            dt = raw_x.dtype
        else:
            raise ValueError("Unable to infer device/dtype for tensor allocation.")

        evidence = torch.zeros(
            batch_size,
            self.n_channels,
            self.hidden_dim,
            device=dev,
            dtype=dt,
        )

        gate_sum = 0.0
        gate_count = 0.0
        window_count = 0
        num_edges = self.src_idx.numel()
        active_slots_per_node = torch.zeros(
            batch_size,
            self.n_channels,
            device=dev,
            dtype=torch.float32,
        )

        # iterate over sparse precomputed windows in the graph
        for win in graph["windows"]:
            window_start = win["window_start"]
            window_end = win["window_end"]
            t_center = win["t_center"]
            active_per_batch = win["active"]

            # gate_count is total possible slots per window
            gate_count += float(batch_size * num_edges * self.nfreqs)
            window_count += 1

            # iterate batches sparsely
            for b in range(batch_size):
                edge_idx_b, freq_idx_b = active_per_batch[b]
                n_active = int(edge_idx_b.numel())
                if n_active == 0:
                    continue

                gate_sum += float(n_active)

                # build payload for active slots
                feats: list[torch.Tensor] = []
                if self.use_mag:
                    # binary indicator that edge exists
                    binary_indicator = torch.ones(n_active, 1, device=dev, dtype=dt)
                    feats.append(binary_indicator)
                    # normalized frequency [0, 1]
                    freq_normalized = freq_idx_b.float().to(dtype=dt) / float(self.nfreqs)
                    feats.append(freq_normalized.unsqueeze(-1))
                if self.use_ang:
                    # would need to recompute wavelet features if use_ang=True
                    raise NotImplementedError(
                        "use_ang is not implemented with sparse graph; coherence is determined at build_graph() time."
                    )
                if self.use_raw:
                    if raw_x is None:
                        raise ValueError("`raw_x` is required when `use_raw` is True")
                    raw_t_b = raw_x[b, :, t_center]
                    src_raw_vals = raw_t_b[self.src_idx[edge_idx_b]].unsqueeze(-1)
                    dst_raw_vals = raw_t_b[self.dst_idx[edge_idx_b]].unsqueeze(-1)
                    feats.extend([src_raw_vals, dst_raw_vals])

                payload = torch.cat(feats, dim=-1)
                msg_b = self.message_mlp(payload)

                # aggregate into destination nodes for this batch
                agg_b = torch.zeros(
                    self.n_channels,
                    msg_b.shape[-1],
                    device=msg_b.device,
                    dtype=msg_b.dtype,
                )
                dst_for_active = self.dst_idx[edge_idx_b].to(device=msg_b.device)
                agg_b.index_add_(0, dst_for_active, msg_b)

                evidence[b] = evidence[b] + agg_b

                # update active_slots_per_node if needed
                if self.evidence_norm == "active_slots":
                    ones = torch.ones(n_active, device=msg_b.device, dtype=torch.float32)
                    counts_b = torch.zeros(self.n_channels, device=msg_b.device, dtype=torch.float32)
                    counts_b.index_add_(0, dst_for_active, ones)
                    active_slots_per_node[b] = active_slots_per_node[b] + counts_b

        if window_count > 0 and self.evidence_norm == "all_slots":
            slots_per_destination = (self.n_channels - 1) * self.nfreqs * window_count
            evidence = evidence / float(slots_per_destination)
        elif window_count > 0 and self.evidence_norm == "windows":
            evidence = evidence / float(window_count)
        elif self.evidence_norm == "active_slots":
            evidence = evidence / active_slots_per_node.clamp_min(1.0).unsqueeze(-1)

        readout = (
            evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
            if self.readout_mode == "flatten"
            else evidence.mean(dim=1)
        )
        logits = self.classifier(readout)
        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return logits, edge_density


class WCTEvidenceGNNClassifier(_BaseCWTGNNClassifier):
    """sklearn/MOABB wrapper around the non-recurrent WCT evidence GNN."""

    model_label = "WCT-Evidence"

    def __init__(
        self,
        sampling_rate: int = 250,
        lowest: float = 8.0,
        highest: float = 35.0,
        nfreqs: int = 16,
        cwt_resample_n_time: int | None = None,
        coherence_threshold: float = 0.7,
        phase_threshold_deg: float = 30.0,
        window_size: int = 25,
        use_mag: bool = True,
        use_ang: bool = False,
        use_raw: bool = True,
        readout_mode: str = "mean",
        evidence_norm: str = "all_slots",
        hidden_dim: int = 8,
        message_dim: int = 8,
        epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 0.1,
        normalize_input: bool = True,
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
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.readout_mode = readout_mode
        self.evidence_norm = evidence_norm
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
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            verbose=verbose,
        )

    def _build_model(self, n_channels: int, n_classes: int) -> WCTEvidenceGNNCore:
        return WCTEvidenceGNNCore(
            n_channels=n_channels,
            nfreqs=self.nfreqs,
            n_classes=n_classes,
            sampling_rate=self.sampling_rate,
            lowest=self.lowest,
            highest=self.highest,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            coherence_threshold=self.coherence_threshold,
            phase_threshold_deg=self.phase_threshold_deg,
            window_size=self.window_size,
            use_mag=self.use_mag,
            use_ang=self.use_ang,
            use_raw=self.use_raw,
            readout_mode=self.readout_mode,
            evidence_norm=self.evidence_norm,
        )
