"""Non-recurrent WCT evidence GNN classifier."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from coheriqs_contributions.nn_components import (
        ActConfig,
        DenseMLPConfig,
        InitConfig,
        NormConfig,
        RegConfig,
        build_dense_mlp,
        scoped_torch_init_seed,
    )
except ModuleNotFoundError:
    from nn_components import (
        ActConfig,
        DenseMLPConfig,
        InitConfig,
        NormConfig,
        RegConfig,
        build_dense_mlp,
        scoped_torch_init_seed,
    )

try:
    from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _compute_wct_window_features,
        _ordered_pair_indices,
    )
except ModuleNotFoundError:
    from moabb_pipelines.wct_phase_gnn_classifier import (
        _BaseCWTGNNClassifier,
        _compute_wct_window_features,
        _ordered_pair_indices,
    )


WCT_EVIDENCE_COMPONENT_PROFILES = (
    "legacy",
)


class WCTEvidenceGNNCore(nn.Module):
    """Torch core for windowed WCT message evidence accumulation."""

    def __init__(
        self,
        n_channels: int,
        nfreqs: int,
        n_classes: int,
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
        component_profile: str = "legacy",
        message_layer_norm: bool = False,
        model_init_seed: int | None = None,
        message_init_seed: int | None = None,
        readout_init_seed: int | None = None,
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
        self.component_profile = component_profile
        if component_profile not in WCT_EVIDENCE_COMPONENT_PROFILES:
            raise ValueError(
                f"Unsupported component_profile={component_profile!r}. "
                f"Expected one of {WCT_EVIDENCE_COMPONENT_PROFILES}."
            )

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
        if payload_dim == 0:
            raise ValueError("At least one payload component must be enabled.")

        with scoped_torch_init_seed(model_init_seed):
            self.message_mlp = _build_message_mlp(
                message_layer_norm=message_layer_norm,
                in_features=payload_dim,
                hidden_features=message_dim,
                out_features=hidden_dim,
                init_seed=message_init_seed,
            )
            readout_dim = (
                hidden_dim * n_channels if readout_mode == "flatten" else hidden_dim
            )
            self.classifier = _build_readout(
                in_features=readout_dim,
                n_classes=n_classes,
                init_seed=readout_init_seed,
            )

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

    def forward(
        self,
        raw_x: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, n_channels, n_time = raw_x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}.")

        evidence = torch.zeros(
            batch_size,
            self.n_channels,
            self.hidden_dim,
            device=raw_x.device,
            dtype=raw_x.dtype,
        )

        gate_sum = 0.0
        gate_count = 0.0
        window_count = 0
        num_edges = self.src_idx.numel()
        active_slots_per_node = torch.zeros(
            batch_size,
            self.n_channels,
            device=raw_x.device,
            dtype=torch.float32,
        )

        for window_start in range(0, n_time - self.window_size + 1, self.window_size):
            window_end = window_start + self.window_size
            t_center = window_start + (self.window_size // 2)

            src_r = w_real[:, self.src_idx, window_start:window_end, :]
            src_i = w_imag[:, self.src_idx, window_start:window_end, :]
            dst_r = w_real[:, self.dst_idx, window_start:window_end, :]
            dst_i = w_imag[:, self.dst_idx, window_start:window_end, :]

            mag, ang, coh, mean_phase = _compute_wct_window_features(
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
            window_count += 1
            if self.evidence_norm == "active_slots":
                active_slots_per_node = (
                    active_slots_per_node
                    + self._aggregate_active_slots_per_node(gate_mask)
                )

            features = []
            if self.use_mag:
                mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
                features.append(mag.unsqueeze(-1))
            if self.use_ang:
                ang = torch.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
                features.append(ang.unsqueeze(-1))
            if self.use_raw:
                raw_t = raw_x[:, :, t_center]
                src_raw = raw_t[:, self.src_idx].unsqueeze(-1).unsqueeze(-1)
                dst_raw = raw_t[:, self.dst_idx].unsqueeze(-1).unsqueeze(-1)
                features.extend(
                    [
                        src_raw.expand(batch_size, num_edges, self.nfreqs, 1),
                        dst_raw.expand(batch_size, num_edges, self.nfreqs, 1),
                    ]
                )

            msg = self.message_mlp(torch.cat(features, dim=-1))
            msg = msg * gate_mask.to(dtype=msg.dtype).unsqueeze(-1)
            evidence = evidence + self._aggregate_per_node(msg.sum(dim=2))

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
        component_profile: str = "legacy",
        message_layer_norm: bool = False,
        message_init_seed: int | None = None,
        readout_init_seed: int | None = None,
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
        self.component_profile = component_profile
        self.message_layer_norm = message_layer_norm
        self.message_init_seed = message_init_seed
        self.readout_init_seed = readout_init_seed
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
            component_profile=self.component_profile,
            message_layer_norm=self.message_layer_norm,
            model_init_seed=self.seed,
            message_init_seed=self.message_init_seed,
            readout_init_seed=self.readout_init_seed,
        )


def _build_message_mlp(
    *,
    message_layer_norm: bool,
    in_features: int,
    hidden_features: int,
    out_features: int,
    init_seed: int | None,
) -> nn.Module:
    return build_dense_mlp(
        DenseMLPConfig(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            activation=ActConfig(kind="relu"),
            norm=NormConfig(kind="layer" if message_layer_norm else None),
            regularization=RegConfig(),
            init=InitConfig(mode="torch_default"),
        ),
        init_seed=init_seed,
    )


def _build_readout(
    *,
    in_features: int,
    n_classes: int,
    init_seed: int | None,
) -> nn.Module:
    return build_dense_mlp(
        DenseMLPConfig(
            in_features=in_features,
            hidden_features=n_classes,
            out_features=n_classes,
            depth=1,
            activation=ActConfig(kind="identity"),
            norm=NormConfig(kind=None),
            regularization=RegConfig(),
            init=InitConfig(mode="torch_default"),
        ),
        init_seed=init_seed,
    )
