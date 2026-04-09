"""Minimal mock-data smoke script for XWTPhaseGNNClassifier.

Usage (from repo root):
    .\moabb-env-win\Scripts\python.exe my_contributions\smoke_xwt_phase_gnn_mock.py
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from my_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
        XWTPhaseGNNClassifier,
    )
except ModuleNotFoundError:
    from moabb_pipelines.xwt_phase_gnn_classifier import XWTPhaseGNNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--n-time", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfreqs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = rng.standard_normal(
        (args.n_samples, args.n_channels, args.n_time), dtype=np.float32
    )
    y = np.arange(args.n_samples) % args.n_classes

    clf = XWTPhaseGNNClassifier(
        sampling_rate=128,
        lowest=8.0,
        highest=35.0,
        nfreqs=args.nfreqs,
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        device="cpu",
        seed=args.seed,
    )

    print("Fitting on mock data...", flush=True)
    clf.fit(X, y)

    pred = clf.predict(X)
    train_acc = float(np.mean(pred == y))
    proba = clf.predict_proba(X[: min(3, args.n_samples)])

    print("\n=== Mock Smoke Results ===", flush=True)
    print(f"X shape: {X.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print(f"pred shape: {pred.shape}", flush=True)
    print(f"train accuracy: {train_acc:.4f}", flush=True)
    print(f"proba sample shape: {proba.shape}", flush=True)
    print(f"loss history: {clf.train_loss_history_}", flush=True)
    print(f"accuracy history: {clf.train_accuracy_history_}", flush=True)
    print(f"edge density history: {clf.edge_density_history_}", flush=True)


if __name__ == "__main__":
    main()
