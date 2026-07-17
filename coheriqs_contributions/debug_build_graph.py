#!/usr/bin/env python3
"""Debug harness for WCT-Evidence graph construction.

This script isolates the expensive graph-building path used by
`WCTEvidenceGNNCore.build_graph()` so you can debug CPU and memory issues
without running the full MOABB training loop.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

# Make local package imports work whether the script is run from repo root
# or directly from within `coheriqs_contributions/`.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from coheriqs_contributions.moabb_pipelines.wct_june18 import WCTEvidenceGNNCore
    from coheriqs_contributions.moabb_pipelines.torch_coherence_utils import (
        compute_coherence_fcwt,
        gaussian_smooth_2d,
    )
except ModuleNotFoundError:
    from moabb_pipelines.wct_june18 import WCTEvidenceGNNCore
    from moabb_pipelines.torch_coherence_utils import (
        compute_coherence_fcwt,
        gaussian_smooth_2d,
    )


def make_tiny_signal(batch_size: int, n_channels: int, n_time: int, device: str) -> torch.Tensor:
    """Generate a small synthetic EEG-like tensor."""
    t = torch.linspace(0, 1, n_time, device=device)
    base = torch.stack(
        [
            torch.sin(2 * np.pi * 8.0 * t),
            torch.sin(2 * np.pi * 12.0 * t + 0.3),
            torch.sin(2 * np.pi * 20.0 * t + 0.7),
            torch.sin(2 * np.pi * 6.0 * t + 1.1),
        ], dim=0,
    )
    base = base[:n_channels]
    x = base.unsqueeze(0).repeat(batch_size, 1, 1)
    x = x + 0.05 * torch.randn_like(x)
    return x


def load_real_sample(subject: int, sample_index: int, device: str) -> tuple[torch.Tensor, int, int]:
    """Load one real EEG epoch from BNCI2014_001."""
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    epochs, y, _metadata = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
    X = epochs.get_data()
    if not (0 <= sample_index < len(X)):
        raise IndexError(f"sample_index={sample_index} out of range for {len(X)} samples")
    raw_np = X[sample_index].astype(np.float32)
    raw_x = torch.from_numpy(raw_np).unsqueeze(0).to(device)
    return raw_x, int(sample_index), int(len(X))


def summarize_graph(graph: dict) -> None:
    windows = graph.get("windows", [])
    total_active = 0
    for idx, win in enumerate(windows):
        active = win.get("active", [])
        window_active = sum(int(edge_idx.numel()) for edge_idx, _ in active)
        total_active += window_active
        print(
            f"[graph] window={idx} start={win.get('window_start')} end={win.get('window_end')} "
            f"center={win.get('t_center')} active_slots={window_active}",
            flush=True,
        )
    print(f"[graph] windows={len(windows)} total_active_slots={total_active}", flush=True)


def run_debug(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.real_sample:
        raw_x, sample_idx, total_samples = load_real_sample(args.subject, args.sample_index, args.device)
        n_channels = int(raw_x.shape[1])
        n_time = int(raw_x.shape[2])
        print(
            f"[input] real sample subject={args.subject} sample_index={sample_idx}/{total_samples - 1} "
            f"raw_x.shape={tuple(raw_x.shape)} device={raw_x.device} dtype={raw_x.dtype}",
            flush=True,
        )
    else:
        n_channels = args.n_channels
        n_time = args.n_time
        raw_x = make_tiny_signal(args.batch_size, n_channels, n_time, args.device)
        print(f"[input] raw_x.shape={tuple(raw_x.shape)} device={raw_x.device} dtype={raw_x.dtype}", flush=True)

    core = WCTEvidenceGNNCore(
        n_channels=n_channels,
        nfreqs=args.nfreqs,
        n_classes=args.n_classes,
        sampling_rate=args.sampling_rate,
        lowest=args.lowest,
        highest=args.highest,
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        coherence_threshold=args.coherence_threshold,
        phase_threshold_deg=args.phase_threshold_deg,
        window_size=args.window_size,
        use_mag=True,
        use_ang=False,
        use_raw=True,
        readout_mode=args.readout_mode,
        evidence_norm=args.evidence_norm,
    )

    print("[debug] timing compute_coherence_fcwt()", flush=True)
    t0 = time.perf_counter()
    coh, mean_phase = compute_coherence_fcwt(
        raw_x,
        args.sampling_rate,
        args.lowest,
        args.highest,
        args.nfreqs,
    )
    t1 = time.perf_counter()
    print(
        f"[debug] coherence done in {t1 - t0:.3f}s coh.shape={tuple(coh.shape)} "
        f"mean_phase.shape={tuple(mean_phase.shape)}",
        flush=True,
    )

    print("[debug] timing gaussian_smooth_2d() on a tiny slice", flush=True)
    tiny = torch.rand(1, 1, min(8, args.n_time), min(8, args.nfreqs), device=raw_x.device)
    t2 = time.perf_counter()
    smoothed = gaussian_smooth_2d(tiny)
    t3 = time.perf_counter()
    print(f"[debug] gaussian_smooth_2d done in {t3 - t2:.3f}s output.shape={tuple(smoothed.shape)}", flush=True)

    print("[debug] timing build_graph()", flush=True)
    t4 = time.perf_counter()
    graph = core.build_graph(raw_x, args.sampling_rate)
    t5 = time.perf_counter()
    print(f"[debug] build_graph done in {t5 - t4:.3f}s", flush=True)
    summarize_graph(graph)

    print("[debug] timing forward()", flush=True)
    t6 = time.perf_counter()
    logits, edge_density = core(raw_x)
    t7 = time.perf_counter()
    print(
        f"[debug] forward done in {t7 - t6:.3f}s logits.shape={tuple(logits.shape)} "
        f"edge_density={edge_density:.6f}",
        flush=True,
    )

    plot_coherence_pair(coh, mean_phase, n_channels=n_channels, ch_i=0, ch_j=1)
    return 0

def plot_coherence_pair(coh, mean_phase, n_channels, ch_i=0, ch_j=1, freq_idx=None, lowest=8.0, highest=35.0, nfreqs=8):
    """Plot coherence (and optionally phase) for one channel pair across all time windows.

    Parameters
    ----------
    coh : torch.Tensor
        Shape (batch, n_pairs_x2, nfreqs) as returned by compute_coherence_fcwt.
    mean_phase : torch.Tensor
        Same shape as coh.
    n_channels : int
        Number of channels in the original signal (used to compute pair index).
    ch_i, ch_j : int
        Channel indices (0-based) for the pair you want to plot. ch_i < ch_j.
    freq_idx : int or None
        Which frequency bin to plot. If None, averages across all frequency bins.
    lowest, highest, nfreqs : float, float, int
        Used to label the frequency axis if freq_idx is None (averaging) vs a single bin.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    assert ch_i < ch_j, "ch_i must be less than ch_j"
    assert 0 <= ch_i < n_channels and 0 <= ch_j < n_channels, "channel indices out of range"

    # Map (ch_i, ch_j) -> flat pair index using upper-triangle ordering
    # pair_index counts how many pairs come before (ch_i, ch_j) in row-major
    # upper-triangle order: for i in range(n_channels): for j in range(i+1, n_channels)
    pair_index = 0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if i == ch_i and j == ch_j:
                break
            pair_index += 1
        else:
            continue
        break

    n_pairs = n_channels * (n_channels - 1) // 2
    total_slots = coh.shape[1]
    # Handle the x2 stacking (e.g. mag/phase or upper/lower) seen in coh.shape[1] == n_pairs * 2
    if total_slots == n_pairs * 2:
        # Assume layout is [pair_0, pair_1, ..., pair_{n_pairs-1}, dup_0, dup_1, ...]
        # If your actual layout interleaves differently, adjust this line.
        slot_index = pair_index
    elif total_slots == n_pairs:
        slot_index = pair_index
    else:
        raise ValueError(
            f"Unexpected coh.shape[1]={total_slots} for n_channels={n_channels} "
            f"(expected {n_pairs} or {n_pairs * 2}). Check pair-indexing convention."
        )

    coh_np = coh[0, slot_index, :].detach().cpu().numpy()  # shape (nfreqs,)
    phase_np = mean_phase[0, slot_index, :].detach().cpu().numpy()

    freqs = np.linspace(lowest, highest, nfreqs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(freqs, coh_np, marker="o")
    axes[0].set_title(f"Coherence: ch{ch_i}-ch{ch_j}")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Coherence")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.7, color="r", linestyle="--", alpha=0.5, label="threshold=0.7")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs, np.degrees(phase_np), marker="o", color="orange")
    axes[1].set_title(f"Mean phase diff: ch{ch_i}-ch{ch_j}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"[plot_coherence_pair] pair=({ch_i},{ch_j}) slot_index={slot_index} "
          f"coh min/mean/max={coh_np.min():.3f}/{coh_np.mean():.3f}/{coh_np.max():.3f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug WCT graph construction in isolation.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--n-time", type=int, default=256)
    parser.add_argument("--nfreqs", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--lowest", type=float, default=8.0)
    parser.add_argument("--highest", type=float, default=35.0)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--coherence-threshold", type=float, default=0.7)
    parser.add_argument("--phase-threshold-deg", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--message-dim", type=int, default=8)
    parser.add_argument("--readout-mode", choices=["mean", "flatten"], default="mean")
    parser.add_argument(
        "--evidence-norm",
        choices=["all_slots", "windows", "active_slots", "none"],
        default="active_slots",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-sample", action="store_true", help="Load one real epoch from BNCI2014_001 instead of synthetic data")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_debug(args)
    except Exception as exc:  # noqa: BLE001 - intentional debug harness
        print("[debug] exception raised during graph build:", file=sys.stderr, flush=True)
        traceback.print_exc()
        print(f"[debug] exception type={type(exc).__name__}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
