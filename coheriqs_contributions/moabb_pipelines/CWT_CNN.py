"""CWT-based CNN classifier for EEG signal classification."""

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
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_minmax,
        fit_minmax_stats,
        prepare_cwt_tf,
        resolve_coherence_utils,
    )


log = logging.getLogger(__name__)


class CWTCNN(nn.Module):
    """CNN over per-channel CWT magnitude maps."""

    def __init__(self, n_classes: int, input_shape: tuple[int, int, int]):
        super().__init__()
        n_channels, _, _ = input_shape
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=(1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.25)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.dropout_fc1(torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(torch.relu(self.bn_fc2(self.fc2(x))))
        return self.fc_out(x)


class CWTCNNClassifier(TorchEEGClassifier):
    """sklearn/MOABB wrapper for CWT-CNN."""

    model_label = "CWT-CNN"

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
        self.transform_min_: float | None = None
        self.transform_max_: float | None = None
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

    def _compute_wavelet_transforms(self, X: np.ndarray) -> np.ndarray:
        if self.transform_ is None:
            self.transform_, _ = resolve_coherence_utils()
        n_samples, n_channels, _ = X.shape
        out = np.zeros(
            (n_samples, n_channels, self.nfreqs, self.cwt_resample_n_time),
            dtype=np.float32,
        )
        with tqdm(
            total=n_samples * n_channels,
            desc="CWT-CNN CWT",
            disable=self.verbose < 1,
            leave=False,
        ) as pbar:
            for sample_idx in range(n_samples):
                for ch_idx in range(n_channels):
                    try:
                        coeffs, _ = self.transform_(
                            X[sample_idx, ch_idx, :],
                            self.sampling_rate,
                            self.highest,
                            self.lowest,
                            nfreqs=self.nfreqs,
                        )
                        out[sample_idx, ch_idx] = np.abs(
                            prepare_cwt_tf(
                                coeffs,
                                nfreqs=self.nfreqs,
                                n_time=self.cwt_resample_n_time,
                            ).T
                        ).astype(np.float32)
                    except Exception as exc:
                        log.debug(
                            "CWT failed for sample %s channel %s: %s",
                            sample_idx,
                            ch_idx,
                            exc,
                        )
                    pbar.update(1)
        return out

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        features = self._compute_wavelet_transforms(X)
        if fit:
            self.transform_min_, self.transform_max_ = fit_minmax_stats(
                features[train_idx]
            )
        if self.transform_min_ is None or self.transform_max_ is None:
            raise ValueError("Feature normalization stats are not initialized.")
        return apply_minmax(features, self.transform_min_, self.transform_max_).astype(
            np.float32
        )

    def _build_model_from_features(self, features, n_classes: int) -> nn.Module:
        return CWTCNN(n_classes, tuple(features.shape[1:]))
