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
        XWTPhaseGNNV2Classifier,
    )
except ModuleNotFoundError:
    from moabb_pipelines.xwt_phase_gnn_classifier import (
        XWTPhaseGNNClassifier,
        XWTPhaseGNNV2Classifier,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v1", "v2"], default="v2")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--n-time", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nfreqs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--message-dim", type=int, default=3)
    parser.add_argument("--hidden-state-dim", type=int, default=32)
    parser.add_argument("--encoder-dim", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument(
        "--normalize-input",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-encoder-batch-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-local-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-prev-state-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--encoder-dropout", type=float, default=0.5)
    parser.add_argument("--gru-input-dropout", type=float, default=0.0)
    parser.add_argument("--readout-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-raw-in-message",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cwt-resample-n-time", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--validation-group-column", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X = rng.standard_normal(
        (args.n_samples, args.n_channels, args.n_time), dtype=np.float32
    )
    y = np.arange(args.n_samples) % args.n_classes

    if args.model == "v1":
        clf = XWTPhaseGNNClassifier(
            sampling_rate=128,
            lowest=8.0,
            highest=35.0,
            nfreqs=args.nfreqs,
            cwt_resample_n_time=args.cwt_resample_n_time,
            hidden_dim=args.hidden_dim,
            message_dim=args.message_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            verbose=args.verbose,
            device="cpu",
            seed=args.seed,
            normalize_input=args.normalize_input,
            validation_split=args.validation_split,
            validation_group_column=args.validation_group_column,
            early_stopping_patience=args.early_stopping_patience,
        )
    else:
        clf = XWTPhaseGNNV2Classifier(
            sampling_rate=128,
            lowest=8.0,
            highest=35.0,
            nfreqs=args.nfreqs,
            cwt_resample_n_time=args.cwt_resample_n_time,
            message_dim=args.message_dim,
            hidden_state_dim=args.hidden_state_dim,
            encoder_dim=args.encoder_dim,
            use_encoder_batch_norm=args.use_encoder_batch_norm,
            encoder_dropout=args.encoder_dropout,
            use_local_residual=args.use_local_residual,
            use_prev_state_mean=args.use_prev_state_mean,
            gru_input_dropout=args.gru_input_dropout,
            readout_dropout=args.readout_dropout,
            use_raw_in_message=args.use_raw_in_message,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            verbose=args.verbose,
            device="cpu",
            seed=args.seed,
            normalize_input=args.normalize_input,
            validation_split=args.validation_split,
            validation_group_column=args.validation_group_column,
            early_stopping_patience=args.early_stopping_patience,
        )

    print("Fitting on mock data...", flush=True)
    clf.fit(X, y)

    pred = clf.predict(X)
    train_acc = float(np.mean(pred == y))
    proba = clf.predict_proba(X[: min(3, args.n_samples)])

    print("\n=== Mock Smoke Results ===", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"X shape: {X.shape}", flush=True)
    print(f"y shape: {y.shape}", flush=True)
    print(f"pred shape: {pred.shape}", flush=True)
    print(f"train accuracy: {train_acc:.4f}", flush=True)
    print(f"proba sample shape: {proba.shape}", flush=True)
    print(f"loss history: {clf.train_loss_history_}", flush=True)
    print(f"accuracy history: {clf.train_accuracy_history_}", flush=True)
    print(f"roc_auc history: {clf.train_roc_auc_history_}", flush=True)
    print(f"val loss history: {clf.val_loss_history_}", flush=True)
    print(f"val accuracy history: {clf.val_accuracy_history_}", flush=True)
    print(f"val roc_auc history: {clf.val_roc_auc_history_}", flush=True)
    print(f"best epoch: {clf.best_epoch_}", flush=True)
    print(f"best val loss: {clf.best_val_loss_}", flush=True)
    print(f"edge density history: {clf.edge_density_history_}", flush=True)


if __name__ == "__main__":
    main()
