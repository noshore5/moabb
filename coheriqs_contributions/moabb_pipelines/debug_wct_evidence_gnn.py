"""
Debug / visualization harness for WCTEvidenceGNNCore.

This script wraps WCTEvidenceGNNCore in a subclass that captures every
intermediate tensor produced during a single forward pass (CWT magnitude,
cross-spectrum, smoothed coherence, phase, gate mask, per-edge evidence,
final logits) and renders a set of diagnostic plots for one sample.

USAGE
-----
1. Fill in `load_model_and_sample()` below to load your trained/untrained
   model (or build one with WCTEvidenceGNNCore(...) directly) and a single
   example's (raw_x, w_real, w_imag, freqs) tensors, shaped:
       raw_x  : [1, C, T]
       w_real : [1, C, T, F]
       w_imag : [1, C, T, F]
       freqs  : [1, F]  or [F]

2. Run:
       python debug_wct_evidence_gnn.py --edge 0 --out debug_plots/

3. Inspect the PNGs written to --out, or import this module and use
   `DebugWCTEvidenceGNNCore` / `run_debug_forward` interactively
   (e.g. in a notebook) to explore model.debug_cache directly.

Everything here only *reads* intermediate tensors via method overrides -
no pipeline math is duplicated or reimplemented, so plots always reflect
exactly what the real forward pass computed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless / script use; drop this line in notebooks
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path

# Make the package root importable regardless of how this script is launched
_PKG_ROOT = Path(__file__).resolve().parents[2]  # .../moabb
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
# ---------------------------------------------------------------------------
# Import the real model. Adjust this import to match your environment.
# ---------------------------------------------------------------------------
try:
    from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
        WCTEvidenceGNNCore,
    )
except ModuleNotFoundError:
    from moabb_pipelines.wct_evidence_gnn_classifier import WCTEvidenceGNNCore


# ---------------------------------------------------------------------------
# 1. Debug subclass: captures intermediates without touching pipeline logic
# ---------------------------------------------------------------------------
class DebugWCTEvidenceGNNCore(WCTEvidenceGNNCore):
    """Subclass of WCTEvidenceGNNCore that stashes intermediate tensors.

    After calling forward() once, everything captured is available in
    `self.debug_cache` as CPU tensors (or floats for scalars), keyed by
    pipeline stage. Nothing about the math is changed - each override
    just calls super() and records the result before returning it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_cache: dict[str, object] = {}

    def _full_edge_wct_maps(self, w_real, w_imag, freqs, *, compute_mag):
        mag, xwt_real, xwt_imag, auto1, auto2 = super()._full_edge_wct_maps(
            w_real, w_imag, freqs, compute_mag=compute_mag
        )
        self.debug_cache["mag"] = None if mag is None else mag.detach().cpu()
        self.debug_cache["xwt_real"] = xwt_real.detach().cpu()
        self.debug_cache["xwt_imag"] = xwt_imag.detach().cpu()
        self.debug_cache["auto1"] = auto1.detach().cpu()
        self.debug_cache["auto2"] = auto2.detach().cpu()
        return mag, xwt_real, xwt_imag, auto1, auto2

    def _smooth_wct_maps(self, *args, **kwargs):
        smooth_cross, coh, kernel = super()._smooth_wct_maps(*args, **kwargs)
        # multiple calls can happen (e.g. chunked mode loops); keep a list
        self.debug_cache.setdefault("smooth_cross_chunks", []).append(
            smooth_cross.detach().cpu()
        )
        self.debug_cache.setdefault("coh_chunks", []).append(coh.detach().cpu())
        self.debug_cache["smooth_kernel"] = kernel.detach().cpu()
        return smooth_cross, coh, kernel

    def _accumulate_evidence_from_window_features(self, **kwargs):
        for key in ("mean_mag", "mean_phase", "coh"):
            v = kwargs.get(key)
            if v is not None:
                self.debug_cache[f"gate_{key}"] = v.detach().cpu()
        logits, edge_density = super()._accumulate_evidence_from_window_features(**kwargs)
        self.debug_cache["evidence_logits"] = logits.detach().cpu()
        self.debug_cache["edge_density"] = float(edge_density)
        return logits, edge_density

    def forward(self, raw_x, w_real, w_imag, freqs):
        self.debug_cache["raw_x"] = raw_x.detach().cpu()
        self.debug_cache["w_real"] = w_real.detach().cpu()
        self.debug_cache["w_imag"] = w_imag.detach().cpu()
        self.debug_cache["freqs"] = freqs.detach().cpu()
        self.debug_cache["resolved_mode"] = self._resolve_window_compute_mode()
        return super().forward(raw_x, w_real, w_imag, freqs)


def run_debug_forward(model: DebugWCTEvidenceGNNCore, raw_x, w_real, w_imag, freqs):
    """Run one forward pass in eval mode, no grad, and return debug_cache."""
    model.eval()
    with torch.no_grad():
        logits, edge_density = model(raw_x, w_real, w_imag, freqs)
    print(f"resolved window_compute_mode: {model.debug_cache['resolved_mode']}")
    print(f"edge_density: {edge_density:.4f}")
    print(f"logits: {logits.numpy()}")
    return model.debug_cache


# ---------------------------------------------------------------------------
# 2. Plotting helpers
# ---------------------------------------------------------------------------
def _freq_labels(freqs: torch.Tensor) -> list[str]:
    f = freqs[0] if freqs.ndim == 2 else freqs
    return [f"{v:.1f}" for v in f.numpy()]


def plot_raw_and_cwt(cache: dict, channel_idx: int, out_dir: Path):
    raw_x = cache["raw_x"]
    w_real = cache["w_real"]
    w_imag = cache["w_imag"]
    freqs = cache["freqs"]
    mag = torch.sqrt(w_real[0, channel_idx] ** 2 + w_imag[0, channel_idx] ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(raw_x[0, channel_idx].numpy())
    axes[0].set_title(f"Raw signal - channel {channel_idx}")
    axes[0].set_xlabel("time")

    im = axes[1].imshow(mag.T.numpy(), aspect="auto", origin="lower", cmap="magma")
    labels = _freq_labels(freqs)
    step = max(len(labels) // 10, 1)
    axes[1].set_yticks(range(0, len(labels), step))
    axes[1].set_yticklabels(labels[::step])
    axes[1].set_title(f"CWT magnitude - channel {channel_idx}")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("frequency")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / f"cwt_channel_{channel_idx}.png", dpi=150)
    plt.close(fig)


def plot_edge_heatmap(tensor: torch.Tensor, edge_idx: int, title: str, out_path: Path, cmap="viridis"):
    arr = tensor[0, edge_idx]  # [time_or_windows, freq]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(arr.T.numpy(), aspect="auto", origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, label=title)
    ax.set_xlabel("time / window index")
    ax.set_ylabel("frequency bin")
    ax.set_title(f"{title} - edge {edge_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gate_mask(model: DebugWCTEvidenceGNNCore, cache: dict, edge_idx: int, out_dir: Path):
    if "gate_coh" not in cache or "gate_mean_phase" not in cache:
        print("gate_coh / gate_mean_phase not in cache, skipping gate mask plot")
        return
    coh = cache["gate_coh"][0, edge_idx]        # [windows, freq]
    phase = cache["gate_mean_phase"][0, edge_idx]
    gate = (coh > model.coherence_threshold) & (phase > model.phase_threshold_rad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(coh.T.numpy(), aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"Coherence (edge {edge_idx})")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(phase.T.numpy(), aspect="auto", origin="lower", cmap="twilight")
    axes[1].set_title(f"Phase, rad (edge {edge_idx})")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(gate.T.numpy(), aspect="auto", origin="lower", cmap="gray")
    axes[2].set_title(
        f"Gate mask (coh>{model.coherence_threshold}, "
        f"phase>{model.phase_threshold_rad:.2f})"
    )
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("window")
        ax.set_ylabel("freq bin")
    fig.tight_layout()
    fig.savefig(out_dir / f"gate_mask_edge_{edge_idx}.png", dpi=150)
    plt.close(fig)


def plot_evidence_summary(cache: dict, out_dir: Path):
    logits = cache["evidence_logits"][0].numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(logits)), logits)
    ax.set_xlabel("class index")
    ax.set_ylabel("logit")
    ax.set_title(f"Output logits (edge_density={cache['edge_density']:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / "evidence_logits.png", dpi=150)
    plt.close(fig)


def plot_all_edges_gate_density(model: DebugWCTEvidenceGNNCore, cache: dict, out_dir: Path):
    """Bar chart of per-edge gate-pass fraction, to spot dead/saturated edges."""
    if "gate_coh" not in cache or "gate_mean_phase" not in cache:
        print("gate_coh / gate_mean_phase not in cache, skipping per-edge density plot")
        return
    coh = cache["gate_coh"][0]        # [E, windows, freq]
    phase = cache["gate_mean_phase"][0]
    gate = (coh > model.coherence_threshold) & (phase > model.phase_threshold_rad)
    density = gate.float().mean(dim=(1, 2)).numpy()  # [E]

    fig, ax = plt.subplots(figsize=(max(8, len(density) * 0.3), 4))
    ax.bar(range(len(density)), density)
    ax.set_xlabel("edge index")
    ax.set_ylabel("fraction of (window, freq) slots gated ON")
    ax.set_title("Per-edge gate density")
    fig.tight_layout()
    fig.savefig(out_dir / "per_edge_gate_density.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Load your model + a sample here
# ---------------------------------------------------------------------------
def load_model_and_sample():
    """
    Loads one epoch from a MOABB dataset/paradigm, and runs it through the
    REAL pipeline's own `_prepare_features` (from _BaseCWTGNNClassifier) to
    get (raw_x, w_real, w_imag, freqs) — no separate CWT implementation here.
    """
    import numpy as np
    from moabb.datasets import BNCI2014_001  # swap for your dataset
    from moabb.paradigms import MotorImagery   # swap for your paradigm

    try:
        from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
            WCTEvidenceGNNClassifier,
        )
    except ModuleNotFoundError:
        from moabb_pipelines.wct_evidence_gnn_classifier import WCTEvidenceGNNClassifier

    # 1. Pull epoched data from MOABB
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=35, resample=250)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    # X: [n_epochs, n_channels, n_times] (numpy)

    # 2. Instantiate the REAL sklearn wrapper with whatever hyperparams
    #    match your actual training config (sampling_rate/lowest/highest/
    #    nfreqs/cwt_resample_n_time especially — these drive the transform).
    clf = WCTEvidenceGNNClassifier(
        sampling_rate=250,
        lowest=8.0,
        highest=35.0,
        nfreqs=16,
        cwt_resample_n_time=None,
        window_size=25,
    )

    # `_prepare_features` needs some attrs normally set up in `_init_torch_classifier`
    # (self.verbose, self.transform_, self.seed, etc.) — calling `_init_cwt_gnn_classifier`
    # once ensures those exist without duplicating its logic.
    clf._init_cwt_gnn_classifier(
        sampling_rate=clf.sampling_rate,
        lowest=clf.lowest,
        highest=clf.highest,
        nfreqs=clf.nfreqs,
        cwt_resample_n_time=clf.cwt_resample_n_time,
        normalize_input=True,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=None,
        device="cpu",
        seed=0,
        verbose=0,
    )

    # 3. Run the REAL feature pipeline (fit=True so z-score stats + transform_
    #    get initialized from this data) on a single-epoch slice.
    sample_idx = 0
    X_one = X[sample_idx : sample_idx + 1]  # keep batch dim: [1, C, T]
    features = clf._prepare_features(X_one, fit=True)
    # `features` is whatever compute_cwt_real_imag_tensors returns — inspect
    # once to confirm shape/order (raw_x, w_real, w_imag, freqs) or a variant.
    raw_x, w_real, w_imag, freqs = features

    n_channels = int(raw_x.shape[1])

    # 4. Build the debug model matching this data's shape/config
    model = DebugWCTEvidenceGNNCore(
        n_channels=n_channels,
        nfreqs=clf.nfreqs,
        n_classes=len(np.unique(labels)),
        window_size=clf.window_size,
    )

    return model, raw_x, w_real, w_imag, freqs

# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Debug plots for WCTEvidenceGNNCore")
    parser.add_argument("--edge", type=int, default=0, help="Edge index to visualize in detail")
    parser.add_argument("--channel", type=int, default=0, help="Channel index for CWT plot")
    parser.add_argument("--out", type=str, default="debug_plots", help="Output directory for PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, raw_x, w_real, w_imag, freqs = load_model_and_sample()
    cache = run_debug_forward(model, raw_x, w_real, w_imag, freqs)

    plot_raw_and_cwt(cache, channel_idx=args.channel, out_dir=out_dir)

    if "mag" in cache and cache["mag"] is not None:
        plot_edge_heatmap(cache["mag"], args.edge, "Cross-wavelet magnitude", out_dir / f"mag_edge_{args.edge}.png")

    if "coh_chunks" in cache:
        # concat along whatever axis chunks were produced on (usually window axis, dim=2)
        coh_full = torch.cat(cache["coh_chunks"], dim=2) if len(cache["coh_chunks"]) > 1 else cache["coh_chunks"][0]
        plot_edge_heatmap(coh_full, args.edge, "Smoothed coherence", out_dir / f"coh_edge_{args.edge}.png")

    plot_gate_mask(model, cache, args.edge, out_dir)
    plot_all_edges_gate_density(model, cache, out_dir)
    plot_evidence_summary(cache, out_dir)

    print(f"\nPlots written to: {out_dir.resolve()}")
    print(f"Available debug_cache keys: {sorted(k for k in cache.keys())}")


if __name__ == "__main__":
    main()
