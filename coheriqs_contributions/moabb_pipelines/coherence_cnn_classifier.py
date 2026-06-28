"""Coherence-based CNN classifier for EEG signal classification."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    from coheriqs_contributions.moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_minmax,
        fit_minmax_stats,
        prepare_cwt_tf,
        resolve_coherence_utils,
        upper_pair_indices,
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_minmax,
        fit_minmax_stats,
        prepare_cwt_tf,
        resolve_coherence_utils,
        upper_pair_indices,
    )


log = logging.getLogger(__name__)


class CoherenceCNN(nn.Module):
    """CNN over pairwise wavelet-coherence maps."""

    def __init__(self, n_classes: int, input_shape: tuple[int, int, int], **kwargs):
        super().__init__()
        n_pairs, nfreqs, n_timepoints = input_shape
        self.conv1 = nn.Conv2d(n_pairs, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * (nfreqs // 8) * (n_timepoints // 8), 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout_fc1(torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(torch.relu(self.bn_fc2(self.fc2(x))))
        return self.fc_out(x)


class CoherenceCNNClassifier(TorchEEGClassifier):
    """sklearn/MOABB wrapper for coherence-map CNN."""

    model_label = "Coherence-CNN"

    def __init__(
        self,
        lowest: float = 4,
        highest: float = 40,
        nfreqs: int = 50,
        sampling_rate: int = 250,
        cwt_resample_n_time: int = 100,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        device: str = "cpu",
        seed: int = 42,
        use_class_weights: bool = False,
        verbose: int = 0,
    ) -> None:
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.sampling_rate = sampling_rate
        self.cwt_resample_n_time = cwt_resample_n_time
        self.transform_ = None
        self.coherence_fn_ = None
        self.coherence_min_: float | None = None
        self.coherence_max_: float | None = None
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            validation_split=validation_split,
            validation_group_column=validation_group_column,
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=seed,
            use_class_weights=use_class_weights,
            verbose=verbose,
        )

    def _compute_channel_cwt(self, X: np.ndarray):
        if self.transform_ is None or self.coherence_fn_ is None:
            self.transform_, self.coherence_fn_ = resolve_coherence_utils()
        n_samples, n_channels, _ = X.shape
        coeffs_by_channel = {}
        freqs_by_channel = {}
        with tqdm(
            total=n_samples * n_channels,
            desc="Coherence CWT",
            disable=self.verbose < 1,
            leave=False,
        ) as pbar:
            for sample_idx in range(n_samples):
                for ch_idx in range(n_channels):
                    try:
                        coeffs, freqs = self.transform_(
                            X[sample_idx, ch_idx, :],
                            self.sampling_rate,
                            self.highest,
                            self.lowest,
                            nfreqs=self.nfreqs,
                        )
                        coeffs_by_channel[(sample_idx, ch_idx)] = prepare_cwt_tf(
                            coeffs,
                            nfreqs=self.nfreqs,
                            n_time=self.cwt_resample_n_time,
                        ).T
                        freqs_by_channel[(sample_idx, ch_idx)] = freqs
                    except Exception as exc:
                        log.debug(
                            "CWT failed for sample %s channel %s: %s",
                            sample_idx,
                            ch_idx,
                            exc,
                        )
                    pbar.update(1)
        return coeffs_by_channel, freqs_by_channel

    def _compute_coherence_matrices(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        pairs = upper_pair_indices(n_channels)
        coeffs_by_channel, freqs_by_channel = self._compute_channel_cwt(X)
        out = np.zeros(
            (n_samples, len(pairs), self.nfreqs, self.cwt_resample_n_time),
            dtype=np.float32,
        )
        for sample_idx in tqdm(
            range(n_samples),
            desc="Coherence",
            disable=self.verbose < 1,
            leave=False,
        ):
            for pair_idx, (ch_i, ch_j) in enumerate(pairs):
                coeffs_i = coeffs_by_channel.get((sample_idx, ch_i))
                coeffs_j = coeffs_by_channel.get((sample_idx, ch_j))
                if coeffs_i is None or coeffs_j is None:
                    continue
                try:
                    coh, _, _ = self.coherence_fn_(
                        coeffs_i,
                        coeffs_j,
                        freqs_by_channel[(sample_idx, ch_i)],
                    )
                    coh = np.asarray(coh)
                    if coh.shape[0] != self.nfreqs and coh.shape[-1] == self.nfreqs:
                        coh = coh.T
                    if coh.shape != (self.nfreqs, self.cwt_resample_n_time):
                        coh = prepare_cwt_tf(
                            coh,
                            nfreqs=self.nfreqs,
                            n_time=self.cwt_resample_n_time,
                        ).T
                    out[sample_idx, pair_idx] = np.nan_to_num(
                        coh, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32)
                except Exception as exc:
                    log.debug(
                        "Coherence failed for sample %s channels (%s, %s): %s",
                        sample_idx,
                        ch_i,
                        ch_j,
                        exc,
                    )
        return out

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        features = self._compute_coherence_matrices(X)
        if fit:
            self.coherence_min_, self.coherence_max_ = fit_minmax_stats(
                features[train_idx]
            )
        if self.coherence_min_ is None or self.coherence_max_ is None:
            raise ValueError("Feature normalization stats are not initialized.")
        return apply_minmax(features, self.coherence_min_, self.coherence_max_).astype(
            np.float32
        )

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        return CoherenceCNN(n_classes, tuple(features.shape[1:]), **kwargs)
