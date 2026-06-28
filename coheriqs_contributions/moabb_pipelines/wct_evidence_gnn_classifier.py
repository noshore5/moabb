"""Non-recurrent WCT evidence GNN classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from coheriqs_contributions.moabb_pipelines.common import make_gaussian_weight2d, resolve_torch_device

try:
    from coheriqs_contributions.nn_components import (
        ActConfig,
        Conv1dConfig,
        Conv2dConfig,
        DenseMLPConfig,
        InitConfig,
        NormConfig,
        RegConfig,
        build_dense_mlp,
        build_conv1d_block,
        build_conv2d_block,
        scoped_torch_init_seed, ResidualConfig,
)
except ModuleNotFoundError:
    from nn_components import (
        ActConfig,
        Conv1dConfig,
        Conv2dConfig,
        DenseMLPConfig,
        InitConfig,
        NormConfig,
        RegConfig,
        build_conv1d_block,
        build_dense_mlp,
        build_conv2d_block,
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
WCT_EVIDENCE_WINDOW_COMPUTE_MODES = (
    "auto",
    "single_pass_windowed",
    "single_pass_continuous",
    "chunked",
    "sequential",
)
DEFAULT_MAX_WINDOWS_PER_CHUNK = 4


@dataclass(frozen=True)
class _WindowLayout:
    starts: list[int]
    ends: list[int]
    centers: list[int]
    feature_indices: list[int]

    @property
    def n_windows(self) -> int:
        return len(self.starts)


def _window_layout(
    *,
    n_time: int,
    window_size: int,
    feature_time_steps: int,
) -> _WindowLayout:
    starts = list(range(0, n_time - window_size + 1, window_size))
    n_windows = len(starts)
    if feature_time_steps < n_windows:
        raise ValueError(
            "Number of feature time steps must be greater than or equal to "
            "the number of full windows."
        )

    ends = [start + window_size for start in starts]
    centers = [start + (window_size // 2) for start in starts]
    feature_time_ratio = feature_time_steps / n_time
    feature_indices = [round(feature_time_ratio * start) for start in starts]
    if any(
        later <= earlier
        for earlier, later in zip(feature_indices, feature_indices[1:], strict=False)
    ):
        raise ValueError("Corresponding feature time steps must be increasing.")

    return _WindowLayout(
        starts=starts,
        ends=ends,
        centers=centers,
        feature_indices=feature_indices,
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= max(int(dim), 0)
    return numel


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def _conv1d_length(length: int, *, kernel_size: int, stride: int) -> int:
    return ((int(length) - int(kernel_size)) // int(stride)) + 1


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
        use_freq: bool = True,
        use_time: bool = True,
        readout_mode: str = "mean",
        evidence_norm: str = "all_slots",
        component_profile: str = "legacy",
        message_layer_norm: bool = False,
        model_init_seed: int | None = None,
        message_init_seed: int | None = None,
        readout_init_seed: int | None = None,

        padding_time_dim: bool = False,
        padding_mode: Literal["reflect", "constant", "replicate"] = "reflect",
        smooth_kernel_sigma: tuple[float, float] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (None, 3),
        window_compute_mode: Literal[
            "auto",
            "single_pass_windowed",
            "single_pass_continuous",
            "chunked",
            "sequential",
        ] = "auto",
        max_windows_per_chunk: int | None = None,
        **kwargs,
       
    ) -> None:
        super().__init__()
        if smooth_kernel_size[0] is not None and smooth_kernel_size[0] <= 0:
            raise ValueError("smooth_kernel_size[0] must be > 0")
        if smooth_kernel_size[1] is None or smooth_kernel_size[1] <= 0:
            raise ValueError("smooth_kernel_size[1] must be > 0 and not None")
        if window_size <= 0:
            raise ValueError("window_size must be >= 1")
        if padding_mode not in {"reflect", "constant", "replicate"}:
            raise ValueError("padding_mode must be one of {'reflect', 'constant', 'replicate'}")
        if smooth_kernel_sigma[0] is not None and (smooth_kernel_sigma[0] <= 0.0):
            raise ValueError("smooth_kernel_sigma[0] must be > 0.0 or None")
        if smooth_kernel_sigma[1] is not None and (smooth_kernel_sigma[1] <= 0.0):
            raise ValueError("smooth_kernel_sigma[1] must be > 0.0 or None")
        if readout_mode not in {"mean", "flatten"}:
            raise ValueError("readout_mode must be one of {'mean', 'flatten'}")
        if evidence_norm not in {"all_slots", "windows", "active_slots", "none"}:
            raise ValueError(
                "evidence_norm must be one of "
                "{'all_slots', 'windows', 'active_slots', 'none'}"
            )
        if window_compute_mode not in WCT_EVIDENCE_WINDOW_COMPUTE_MODES:
            raise ValueError(
                f"window_compute_mode must be one of "
                f"{WCT_EVIDENCE_WINDOW_COMPUTE_MODES}."
            )
        if max_windows_per_chunk is not None and max_windows_per_chunk <= 0:
            raise ValueError("max_windows_per_chunk must be > 0 or None.")

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
        self.use_freq = use_freq
        self.use_time = use_time
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

        
        self.padding_time_dim = padding_time_dim
        self.padding_mode = padding_mode
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.smooth_kernel_size = (
            window_size if smooth_kernel_size[0] is None else smooth_kernel_size[0],
            smooth_kernel_size[1],
        )
        self.window_compute_mode = window_compute_mode
        self.max_windows_per_chunk = max_windows_per_chunk

        feature_dim = 4
        self.feature_dim = feature_dim
        payload_dim = feature_dim * 2
        if self.use_freq:
            payload_dim += 1
        if self.use_time:
            payload_dim += 1
        if self.use_mag:
            payload_dim += 1
        if self.use_ang:
            payload_dim += 1
        if self.use_raw:
            payload_dim += 2
        if payload_dim == 0:
            raise ValueError("At least one payload component must be enabled.")
        self.payload_dim = payload_dim
        self._summary_context: dict[str, object] | None = None

        with scoped_torch_init_seed(model_init_seed):

            self.feature_conv = _build_feature_conv(
                kernel_size=5,
                intermediate_channels=nfreqs,
                out_channels=feature_dim * nfreqs,
                pool_size=4,
            )

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

    def configure_summary_context(
        self,
        *,
        batch_size: int,
        n_time: int,
        dtype: torch.dtype,
        n_samples: int | None = None,
    ) -> None:
        self._summary_context = {
            "batch_size": int(batch_size),
            "n_time": int(n_time),
            "dtype": dtype,
            "n_samples": None if n_samples is None else int(n_samples),
        }

    def print_custom_summary(self, header: str = "Model") -> None:
        effective_mode = self._resolve_window_compute_mode()
        chunk_cap = (
            self.max_windows_per_chunk
            if self.max_windows_per_chunk is not None
            else DEFAULT_MAX_WINDOWS_PER_CHUNK
        )
        print(
            f"[{header}] WCTEvidence config "
            f"window_compute_mode={self.window_compute_mode} "
            f"effective_window_compute_mode={effective_mode} "
            f"max_windows_per_chunk={self.max_windows_per_chunk} "
            f"default_chunk_cap={DEFAULT_MAX_WINDOWS_PER_CHUNK} "
            f"chunk_cap_effective={chunk_cap}",
            flush=True,
        )
        print(
            f"[{header}] WCTEvidence config "
            f"window_size={self.window_size} nfreqs={self.nfreqs} "
            f"hidden_dim={self.hidden_dim} message_dim={self.message_dim} "
            f"payload_dim={self.payload_dim} "
            f"smooth_kernel_size={self.smooth_kernel_size} "
            f"padding_time_dim={self.padding_time_dim} "
            f"padding_mode={self.padding_mode}",
            flush=True,
        )

        context = self._summary_context
        if context is None:
            print(
                f"[{header}] WCTEvidence memory estimates unavailable: "
                "summary context was not configured.",
                flush=True,
            )
            return

        batch_size = int(context["batch_size"])
        n_time = int(context["n_time"])
        n_samples = context["n_samples"]
        dtype = context["dtype"]
        n_windows = n_time // self.window_size
        num_edges = self.src_idx.numel()
        feature_time_steps = self._estimate_feature_time_steps(n_time)
        dtype_bytes = _dtype_nbytes(dtype)
        print(
            f"[{header}] WCTEvidence dimensions "
            f"B={batch_size} C={self.n_channels} E={num_edges} "
            f"T={n_time} W={self.window_size} N={n_windows} F={self.nfreqs} "
            f"D={self.feature_dim} H={self.hidden_dim} "
            f"dtype={dtype} bytes_per_elem={dtype_bytes} "
            f"n_samples={n_samples}",
            flush=True,
        )
        print(
            f"[{header}] WCTEvidence memory estimates "
            "approximate tensor payloads only; autograd, optimizer state, "
            "allocator fragmentation, and convolution workspace are excluded.",
            flush=True,
        )
        for label, shape, copies in self._critical_tensor_estimates(
            batch_size=batch_size,
            n_time=n_time,
            n_windows=n_windows,
            feature_time_steps=feature_time_steps,
            effective_mode=effective_mode,
        ):
            numel = _shape_numel(shape) * copies
            copies_prefix = f"{copies} x " if copies != 1 else ""
            print(
                f"[{header}] WCTEvidence tensor {label}: "
                f"shape={copies_prefix}{shape} "
                f"elements={numel} "
                f"approx_memory={_format_bytes(numel * dtype_bytes)}",
                flush=True,
            )

    def _estimate_feature_time_steps(self, n_time: int) -> int:
        length = int(n_time)
        for kernel_size, stride in [(5, 1), (4, 4), (5, 1), (4, 4)]:
            length = _conv1d_length(length, kernel_size=kernel_size, stride=stride)
            if length <= 0:
                return 0
        return length

    def _critical_tensor_estimates(
        self,
        *,
        batch_size: int,
        n_time: int,
        n_windows: int,
        feature_time_steps: int,
        effective_mode: str,
    ) -> list[tuple[str, tuple[int, ...], int]]:
        num_edges = self.src_idx.numel()
        estimates = [
            (
                "feature_conv_output",
                (batch_size, self.nfreqs * self.feature_dim, self.n_channels, feature_time_steps),
                1,
            ),
            (
                "edge_conv_src_dst",
                (batch_size, num_edges, self.nfreqs, self.feature_dim, feature_time_steps),
                2,
            ),
        ]

        if effective_mode in {"single_pass_windowed", "single_pass_continuous"}:
            estimates.append(
                (
                    "full_edge_wct_maps",
                    (batch_size, num_edges, n_time, self.nfreqs),
                    5,
                )
            )
            if effective_mode == "single_pass_windowed":
                smooth_time = max(n_time - self.smooth_kernel_size[0] + 1, 0)
                positions = max(self.window_size - self.smooth_kernel_size[0] + 1, 0)
                estimates.extend(
                    [
                        (
                            "smoothed_wct_maps",
                            (batch_size, num_edges, smooth_time, self.nfreqs),
                            4,
                        ),
                        (
                            "windowed_coh_cross_selection",
                            (batch_size, num_edges, n_windows, positions, self.nfreqs),
                            2,
                        ),
                    ]
                )
            else:
                estimates.extend(
                    [
                        (
                            "continuous_smoothed_wct_maps",
                            (batch_size, num_edges, n_time, self.nfreqs),
                            4,
                        ),
                        (
                            "continuous_window_view",
                            (
                                batch_size,
                                num_edges,
                                n_windows,
                                self.window_size,
                                self.nfreqs,
                            ),
                            2,
                        ),
                    ]
                )
        else:
            chunk_windows = 1 if effective_mode == "sequential" else self._chunk_size(
                _WindowLayout(
                    starts=[0] * n_windows,
                    ends=[0] * n_windows,
                    centers=[0] * n_windows,
                    feature_indices=[0] * n_windows,
                )
            )
            estimates.extend(
                [
                    (
                        "chunk_window_inputs",
                        (
                            batch_size,
                            num_edges,
                            chunk_windows,
                            self.window_size,
                            self.nfreqs,
                        ),
                        4,
                    ),
                    (
                        "chunk_helper_maps",
                        (
                            batch_size * chunk_windows,
                            num_edges,
                            self.window_size,
                            self.nfreqs,
                        ),
                        4,
                    ),
                    (
                        "chunk_window_features",
                        (batch_size, num_edges, chunk_windows, self.nfreqs),
                        3,
                    ),
                ]
            )

        estimates.extend(
            [
                (
                    "message_payload",
                    (
                        batch_size,
                        num_edges,
                        n_windows,
                        self.nfreqs,
                        self.payload_dim,
                    ),
                    1,
                ),
                (
                    "messages",
                    (
                        batch_size,
                        num_edges,
                        n_windows,
                        self.nfreqs,
                        self.hidden_dim,
                    ),
                    1,
                ),
                ("evidence", (batch_size, self.n_channels, self.hidden_dim), 1),
            ]
        )
        return estimates

    def _resolve_window_compute_mode(self) -> str:
        if self.window_compute_mode != "auto":
            return self.window_compute_mode
        if self._single_pass_windowed_supported():
            return "single_pass_windowed"
        return "chunked"

    def _single_pass_windowed_supported(self) -> bool:
        return (
            not self.padding_time_dim
            and self.smooth_kernel_size[0] <= self.window_size
        )

    def _validate_single_pass_windowed_supported(self) -> None:
        if self._single_pass_windowed_supported():
            return
        raise ValueError(
            "single_pass_windowed requires padding_time_dim=False and "
            "smooth_kernel_size[0] <= window_size so window boundaries stay exact."
        )

    def _batched_freqs(self, freqs: torch.Tensor, batch_size: int) -> torch.Tensor:
        if freqs.ndim == 1:
            freqs = freqs.view(1, -1).expand(batch_size, -1)
        if freqs.shape != (batch_size, self.nfreqs):
            raise ValueError(
                f"Expected freqs shape {(batch_size, self.nfreqs)} or "
                f"{(self.nfreqs,)}, got {tuple(freqs.shape)}."
            )
        return freqs

    def _full_edge_wct_maps(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_r = w_real.index_select(1, self.src_idx)
        src_i = w_imag.index_select(1, self.src_idx)
        dst_r = w_real.index_select(1, self.dst_idx)
        dst_i = w_imag.index_select(1, self.dst_idx)

        xwt_real = src_r * dst_r + src_i * dst_i
        xwt_imag = src_i * dst_r - src_r * dst_i
        mag = torch.sqrt(xwt_real * xwt_real + xwt_imag * xwt_imag + 1e-12)
        auto1 = src_r * src_r + src_i * src_i
        auto2 = dst_r * dst_r + dst_i * dst_i

        inv_scale = freqs.view(freqs.shape[0], 1, 1, self.nfreqs)
        return (
            mag,
            xwt_real * inv_scale,
            xwt_imag * inv_scale,
            auto1 * inv_scale,
            auto2 * inv_scale,
        )

    def _smooth_wct_maps(
        self,
        xwt_real: torch.Tensor,
        xwt_imag: torch.Tensor,
        auto1: torch.Tensor,
        auto2: torch.Tensor,
        smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth_kernel, pad = smooth_kernel_and_pad
        batch_size, num_edges, n_time, nfreqs = xwt_real.shape

        maps = torch.stack([xwt_real, xwt_imag, auto1, auto2], dim=2)
        maps = maps.view(batch_size * num_edges * 4, 1, n_time, nfreqs)
        maps = torch.nn.functional.pad(maps, pad, mode=self.padding_mode)
        smoothed = torch.nn.functional.conv2d(maps, smooth_kernel)
        out_time, out_freq = smoothed.shape[-2:]
        smoothed = smoothed.view(batch_size, num_edges, 4, out_time, out_freq)
        smooth_cross = torch.complex(smoothed[:, :, 0], smoothed[:, :, 1])
        smooth_auto1 = smoothed[:, :, 2]
        smooth_auto2 = smoothed[:, :, 3]
        coh = (smooth_cross.abs() ** 2) / (smooth_auto1 * smooth_auto2 + 1e-12)
        return smooth_cross, coh.clamp(min=0.0, max=1.0), smooth_kernel

    def _window_mean(self, values: torch.Tensor, layout: _WindowLayout) -> torch.Tensor:
        batch_size, num_edges, _, nfreqs = values.shape
        if layout.n_windows == 0:
            return values.new_empty(batch_size, num_edges, 0, nfreqs)
        values = values[:, :, : layout.n_windows * self.window_size, :]
        values = values.reshape(
            batch_size,
            num_edges,
            layout.n_windows,
            self.window_size,
            nfreqs,
        )
        return values.mean(dim=3)

    def _compute_wct_features_single_pass_windowed(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
        layout: _WindowLayout,
        smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_single_pass_windowed_supported()
        mag, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
            w_real,
            w_imag,
            freqs,
        )
        mean_mag = self._window_mean(mag, layout)

        smooth_cross, coh_by_time, smooth_kernel = self._smooth_wct_maps(
            xwt_real,
            xwt_imag,
            auto1,
            auto2,
            smooth_kernel_and_pad,
        )
        kernel_time = smooth_kernel.shape[-2]
        positions_per_window = self.window_size - kernel_time + 1
        if positions_per_window <= 0:
            raise ValueError(
                "single_pass_windowed requires smooth_kernel_size[0] <= window_size."
            )

        batch_size, num_edges, _, nfreqs = coh_by_time.shape
        if layout.n_windows == 0:
            empty = coh_by_time.new_empty(batch_size, num_edges, 0, nfreqs)
            return empty, empty, empty

        starts = torch.tensor(layout.starts, device=coh_by_time.device)
        offsets = torch.arange(positions_per_window, device=coh_by_time.device)
        time_indices = (starts[:, None] + offsets[None, :]).reshape(-1)

        coh_windows = coh_by_time.index_select(2, time_indices).reshape(
            batch_size,
            num_edges,
            layout.n_windows,
            positions_per_window,
            nfreqs,
        )
        cross_windows = smooth_cross.index_select(2, time_indices).reshape(
            batch_size,
            num_edges,
            layout.n_windows,
            positions_per_window,
            nfreqs,
        )
        coh, coh_max_idx = torch.max(coh_windows, dim=3)
        smooth_cross = torch.gather(
            cross_windows,
            dim=3,
            index=coh_max_idx.unsqueeze(3),
        ).squeeze(3)
        mean_phase = torch.angle(smooth_cross)
        return mean_mag, mean_phase, coh

    def _compute_wct_features_single_pass_continuous(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
        layout: _WindowLayout,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag, xwt_real, xwt_imag, auto1, auto2 = self._full_edge_wct_maps(
            w_real,
            w_imag,
            freqs,
        )
        mean_mag = self._window_mean(mag, layout)

        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size,
            sigma=self.smooth_kernel_sigma,
            pad_h=None,
            device=w_real.device,
            dtype=w_real.dtype,
        )
        smooth_cross, coh_by_time, _ = self._smooth_wct_maps(
            xwt_real,
            xwt_imag,
            auto1,
            auto2,
            smooth_kernel_and_pad,
        )

        batch_size, num_edges, _, nfreqs = coh_by_time.shape
        if layout.n_windows == 0:
            empty = coh_by_time.new_empty(batch_size, num_edges, 0, nfreqs)
            return empty, empty, empty

        coh_windows = coh_by_time[:, :, : layout.n_windows * self.window_size, :]
        coh_windows = coh_windows.reshape(
            batch_size,
            num_edges,
            layout.n_windows,
            self.window_size,
            nfreqs,
        )
        cross_windows = smooth_cross[:, :, : layout.n_windows * self.window_size, :]
        cross_windows = cross_windows.reshape(
            batch_size,
            num_edges,
            layout.n_windows,
            self.window_size,
            nfreqs,
        )
        coh, coh_max_idx = torch.max(coh_windows, dim=3)
        smooth_cross = torch.gather(
            cross_windows,
            dim=3,
            index=coh_max_idx.unsqueeze(3),
        ).squeeze(3)
        mean_phase = torch.angle(smooth_cross)
        return mean_mag, mean_phase, coh

    def _compute_wct_features_chunked(
        self,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
        layout: _WindowLayout,
        smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]],
        *,
        max_windows_per_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = w_real.shape[0]
        num_edges = self.src_idx.numel()
        chunks = []
        for first in range(0, layout.n_windows, max_windows_per_chunk):
            last = min(first + max_windows_per_chunk, layout.n_windows)
            chunk_starts = layout.starts[first:last]
            chunk_ends = layout.ends[first:last]
            chunk_size = last - first

            src_r = torch.stack(
                [
                    w_real[:, self.src_idx, window_start:window_end, :]
                    for window_start, window_end in zip(chunk_starts, chunk_ends, strict=True)
                ],
                dim=2,
            )
            src_i = torch.stack(
                [
                    w_imag[:, self.src_idx, window_start:window_end, :]
                    for window_start, window_end in zip(chunk_starts, chunk_ends, strict=True)
                ],
                dim=2,
            )
            dst_r = torch.stack(
                [
                    w_real[:, self.dst_idx, window_start:window_end, :]
                    for window_start, window_end in zip(chunk_starts, chunk_ends, strict=True)
                ],
                dim=2,
            )
            dst_i = torch.stack(
                [
                    w_imag[:, self.dst_idx, window_start:window_end, :]
                    for window_start, window_end in zip(chunk_starts, chunk_ends, strict=True)
                ],
                dim=2,
            )

            src_r = src_r.permute(0, 2, 1, 3, 4).reshape(
                batch_size * chunk_size,
                num_edges,
                self.window_size,
                self.nfreqs,
            )
            src_i = src_i.permute(0, 2, 1, 3, 4).reshape_as(src_r)
            dst_r = dst_r.permute(0, 2, 1, 3, 4).reshape_as(src_r)
            dst_i = dst_i.permute(0, 2, 1, 3, 4).reshape_as(src_r)
            chunk_freqs = freqs[:, None, :].expand(
                batch_size,
                chunk_size,
                self.nfreqs,
            )
            chunk_freqs = chunk_freqs.reshape(batch_size * chunk_size, self.nfreqs)

            mean_mag, mean_phase, coh = _compute_wct_window_features(
                src_r,
                src_i,
                dst_r,
                dst_i,
                chunk_freqs,
                smooth_kernel_and_pad=smooth_kernel_and_pad,
                padding_mode=self.padding_mode,
            )
            chunks.append(
                (
                    mean_mag.reshape(batch_size, chunk_size, num_edges, self.nfreqs)
                    .permute(0, 2, 1, 3),
                    mean_phase.reshape(batch_size, chunk_size, num_edges, self.nfreqs)
                    .permute(0, 2, 1, 3),
                    coh.reshape(batch_size, chunk_size, num_edges, self.nfreqs)
                    .permute(0, 2, 1, 3),
                )
            )

        if not chunks:
            empty = w_real.new_empty(batch_size, num_edges, 0, self.nfreqs)
            return empty, empty, empty

        mean_mag = torch.cat([chunk[0] for chunk in chunks], dim=2)
        mean_phase = torch.cat([chunk[1] for chunk in chunks], dim=2)
        coh = torch.cat([chunk[2] for chunk in chunks], dim=2)
        return mean_mag, mean_phase, coh

    def _compute_wct_features_for_mode(
        self,
        *,
        mode: str,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
        freqs: torch.Tensor,
        layout: _WindowLayout,
        smooth_kernel_and_pad: tuple[torch.Tensor, tuple[int, int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "single_pass_windowed":
            return self._compute_wct_features_single_pass_windowed(
                w_real,
                w_imag,
                freqs,
                layout,
                smooth_kernel_and_pad,
            )
        if mode == "single_pass_continuous":
            return self._compute_wct_features_single_pass_continuous(
                w_real,
                w_imag,
                freqs,
                layout,
            )
        if mode in {"chunked", "sequential"}:
            chunk_size = 1 if mode == "sequential" else self._chunk_size(layout)
            return self._compute_wct_features_chunked(
                w_real,
                w_imag,
                freqs,
                layout,
                smooth_kernel_and_pad,
                max_windows_per_chunk=chunk_size,
            )
        raise ValueError(f"Unsupported resolved window compute mode: {mode!r}.")

    def _chunk_size(self, layout: _WindowLayout) -> int:
        requested = (
            self.max_windows_per_chunk
            if self.max_windows_per_chunk is not None
            else DEFAULT_MAX_WINDOWS_PER_CHUNK
        )
        return min(requested, max(layout.n_windows, 1))

    def _accumulate_evidence_from_window_features(
        self,
        *,
        raw_x: torch.Tensor,
        edge_src_conv: torch.Tensor,
        edge_dst_conv: torch.Tensor,
        freqs: torch.Tensor,
        layout: _WindowLayout,
        mean_mag: torch.Tensor,
        mean_phase: torch.Tensor,
        coh: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        batch_size, _, n_time = raw_x.shape
        num_edges = self.src_idx.numel()
        window_count = layout.n_windows

        if window_count == 0:
            evidence = raw_x.new_zeros(batch_size, self.n_channels, self.hidden_dim)
            return self._readout(evidence), 0.0

        gate_mask = (coh > self.coherence_threshold) & (
            mean_phase > self.phase_threshold_rad
        )
        gate_sum = float(gate_mask.sum().item())
        gate_count = float(gate_mask.numel())

        feature_indices = torch.tensor(
            layout.feature_indices,
            device=edge_src_conv.device,
            dtype=torch.long,
        )
        centers = torch.tensor(layout.centers, device=raw_x.device, dtype=torch.long)

        features = []
        src_conv = edge_src_conv.index_select(4, feature_indices).permute(0, 1, 4, 2, 3)
        dst_conv = edge_dst_conv.index_select(4, feature_indices).permute(0, 1, 4, 2, 3)
        features.append(src_conv)
        features.append(dst_conv)

        if self.use_freq:
            freq_features = (1.0 / freqs).view(batch_size, 1, 1, self.nfreqs, 1)
            features.append(
                freq_features.expand(
                    batch_size,
                    num_edges,
                    window_count,
                    self.nfreqs,
                    1,
                )
            )

        if self.use_time:
            time_centers = raw_x.new_tensor(layout.centers, dtype=raw_x.dtype)
            time_centers = time_centers / float(max(n_time - 1, 1))
            features.append(
                time_centers.view(1, 1, window_count, 1, 1).expand(
                    batch_size,
                    num_edges,
                    window_count,
                    self.nfreqs,
                    1,
                )
            )

        if self.use_mag:
            mean_mag = torch.nan_to_num(mean_mag, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(mean_mag.unsqueeze(-1))
        if self.use_ang:
            mean_phase = torch.nan_to_num(
                mean_phase,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            features.append(mean_phase.unsqueeze(-1))
        if self.use_raw:
            raw_t = raw_x.index_select(2, centers)
            src_raw = raw_t[:, self.src_idx, :].unsqueeze(3).unsqueeze(4)
            dst_raw = raw_t[:, self.dst_idx, :].unsqueeze(3).unsqueeze(4)
            features.extend(
                [
                    src_raw.expand(
                        batch_size,
                        num_edges,
                        window_count,
                        self.nfreqs,
                        1,
                    ),
                    dst_raw.expand(
                        batch_size,
                        num_edges,
                        window_count,
                        self.nfreqs,
                        1,
                    ),
                ]
            )

        msg = self.message_mlp(torch.cat(features, dim=-1))
        msg = msg * gate_mask.to(dtype=msg.dtype).unsqueeze(-1)
        evidence = self._aggregate_per_node(msg.sum(dim=(2, 3)))

        if self.evidence_norm == "all_slots":
            slots_per_destination = (self.n_channels - 1) * self.nfreqs * window_count
            evidence = evidence / float(slots_per_destination)
        elif self.evidence_norm == "windows":
            evidence = evidence / float(window_count)
        elif self.evidence_norm == "active_slots":
            active_per_edge = gate_mask.to(dtype=torch.float32).sum(dim=(2, 3))
            active_slots_per_node = torch.zeros(
                batch_size,
                self.n_channels,
                device=raw_x.device,
                dtype=torch.float32,
            )
            active_slots_per_node.index_add_(1, self.dst_idx, active_per_edge)
            evidence = evidence / active_slots_per_node.clamp_min(1.0).unsqueeze(-1)

        edge_density = (gate_sum / gate_count) if gate_count > 0 else 0.0
        return self._readout(evidence), edge_density

    def _readout(self, evidence: torch.Tensor) -> torch.Tensor:
        batch_size = evidence.shape[0]
        readout = (
            evidence.reshape(batch_size, self.n_channels * self.hidden_dim)
            if self.readout_mode == "flatten"
            else evidence.mean(dim=1)
        )
        return self.classifier(readout)

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
        dtype = raw_x.dtype

        smooth_kernel_and_pad = make_gaussian_weight2d(
            kernel_size=self.smooth_kernel_size,
            sigma=self.smooth_kernel_sigma,
            pad_h=0 if not self.padding_time_dim else None,
            device=device,
            dtype=dtype,
        )

        # [B, 1, C, T] -> [B, F*D, C, T']
        # C' = feature_dim
        conv_features = self.feature_conv(raw_x.unsqueeze(1))

        feature_time_steps = conv_features.shape[3]

        layout = _window_layout(
            n_time=n_time,
            window_size=self.window_size,
            feature_time_steps=feature_time_steps,
        )

        # [B, F*D, C, T'] -> [B, F, D, C, T']
        conv_by_freq = conv_features.view(
            batch_size,
            self.nfreqs,
            self.feature_dim,
            n_channels,
            feature_time_steps,
        )
        # [B, F, D, C, T'] -> [B, C, F, D, T']
        conv_by_freq = conv_by_freq.permute(0, 3, 1, 2, 4)

        # in edge form: [B, C, F, D, T'] -> [B, E, F, D, T']
        edge_src_conv = conv_by_freq.index_select(1, self.src_idx)
        edge_dst_conv = conv_by_freq.index_select(1, self.dst_idx)

        freqs = self._batched_freqs(freqs, batch_size).to(
            device=raw_x.device,
            dtype=raw_x.dtype,
        )
        mode = self._resolve_window_compute_mode()
        mean_mag, mean_phase, coh = self._compute_wct_features_for_mode(
            mode=mode,
            w_real=w_real,
            w_imag=w_imag,
            freqs=freqs,
            layout=layout,
            smooth_kernel_and_pad=smooth_kernel_and_pad,
        )
        return self._accumulate_evidence_from_window_features(
            raw_x=raw_x,
            edge_src_conv=edge_src_conv,
            edge_dst_conv=edge_dst_conv,
            freqs=freqs,
            layout=layout,
            mean_mag=mean_mag,
            mean_phase=mean_phase,
            coh=coh,
        )


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
        use_freq: bool = True,
        use_time: bool = True,
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
        component_profile: str = "legacy",
        message_layer_norm: bool = False,
        message_init_seed: int | None = None,
        readout_init_seed: int | None = None,
        padding_time_dim: bool = False,
        padding_mode: Literal["reflect", "constant", "replicate"] = "reflect",
        smooth_kernel_sigma: tuple[float, float] = (None, None),
        smooth_kernel_size: tuple[int | None, int] = (None, 3),
        window_compute_mode: Literal[
            "auto",
            "single_pass_windowed",
            "single_pass_continuous",
            "chunked",
            "sequential",
        ] = "auto",
        max_windows_per_chunk: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.coherence_threshold = coherence_threshold
        self.phase_threshold_deg = phase_threshold_deg
        self.window_size = window_size
        self.use_mag = use_mag
        self.use_ang = use_ang
        self.use_raw = use_raw
        self.use_freq = use_freq
        self.use_time = use_time
        self.readout_mode = readout_mode
        self.evidence_norm = evidence_norm
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.component_profile = component_profile
        self.message_layer_norm = message_layer_norm
        self.message_init_seed = message_init_seed
        self.readout_init_seed = readout_init_seed
        self.padding_time_dim = padding_time_dim
        self.padding_mode = padding_mode
        self.smooth_kernel_sigma = smooth_kernel_sigma
        self.smooth_kernel_size = smooth_kernel_size
        self.window_compute_mode = window_compute_mode
        self.max_windows_per_chunk = max_windows_per_chunk
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

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> WCTEvidenceGNNCore:
        raw_x = features[0] if isinstance(features, tuple) else features
        model = self._build_model(
            n_channels=int(raw_x.shape[1]),
            n_classes=n_classes,
            **kwargs,
        )
        model.configure_summary_context(
            batch_size=int(self.batch_size),
            n_time=int(raw_x.shape[2]),
            dtype=raw_x.dtype,
            n_samples=int(raw_x.shape[0]),
        )
        return model

    def _build_model(self, n_channels: int, n_classes: int, **kwargs) -> WCTEvidenceGNNCore:
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
            use_freq=self.use_freq,
            use_time=self.use_time,
            readout_mode=self.readout_mode,
            evidence_norm=self.evidence_norm,
            component_profile=self.component_profile,
            message_layer_norm=self.message_layer_norm,
            model_init_seed=self.seed,
            message_init_seed=self.message_init_seed,
            readout_init_seed=self.readout_init_seed,
            padding_time_dim=self.padding_time_dim,
            padding_mode=self.padding_mode,
            smooth_kernel_sigma=self.smooth_kernel_sigma,
            smooth_kernel_size=self.smooth_kernel_size,
            window_compute_mode=self.window_compute_mode,
            max_windows_per_chunk=self.max_windows_per_chunk,
            **kwargs,
        )


def _build_feature_conv(
    *,
    kernel_size: int,
    intermediate_channels: int,
    out_channels: int,
    pool_size: int,
) -> nn.Module:
    conv_blocks = []
    conv1 = build_conv2d_block(
        Conv2dConfig(
            in_channels=1,
            out_channels=intermediate_channels,
            kernel_size=(1, kernel_size),
            padding=0,
            regularization=RegConfig(0.5, 0.0),
            norm=NormConfig("layer"),
            activation=ActConfig(kind="gelu"),
        ),
    )
    max_pool1 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))

    conv_blocks.append(conv1)
    conv_blocks.append(max_pool1)
    conv2 = build_conv2d_block(
        Conv2dConfig(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=0,
            regularization=RegConfig(0.0, 0.5),
            norm=NormConfig("layer"),
            activation=ActConfig(kind="gelu"),
        ),
    )
    max_pool2 = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
    conv_blocks.append(conv2)
    conv_blocks.append(max_pool2)

    return nn.Sequential(*conv_blocks)


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
            depth=2,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            activation=ActConfig(kind="silu"),
            norm=NormConfig(kind="layer" if message_layer_norm else None, affine=True),
            regularization=RegConfig(dropout=0.0),
            init=InitConfig(mode="torch_default"),
        ),
        # residual=ResidualConfig(norm_position="sandwich", shortcut="auto", rezero=True),
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
