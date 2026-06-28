"""EEGNet classifier for EEG signal classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from coheriqs_contributions.moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        fit_global_zscore_stats,
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import (
        TorchEEGClassifier,
        apply_global_zscore,
        fit_global_zscore_stats,
    )

class EEGNetModel(nn.Module):
    """Compact EEGNet-style CNN."""

    def __init__(self, n_classes: int, n_channels: int, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        f1 = 8
        f2 = 16
        depth_multiplier = 2

        self.conv1 = nn.Conv2d(1, f1, kernel_size=(1, 51), padding=(0, 25), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.elu = nn.ELU(alpha=1.0)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.sep_conv = nn.Conv2d(
            f1 * depth_multiplier, f2, kernel_size=(1, 15), padding=(0, 7), bias=False
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(f2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.pool1(self.elu(self.bn2(self.depthwise(x))))
        x = self.dropout1(x)
        x = self.pool2(self.elu(self.bn3(self.sep_conv(x))))
        x = self.dropout2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)


class EEGNetClassifier(TorchEEGClassifier):
    """sklearn/MOABB wrapper for EEGNet."""

    model_label = "EEGNet"

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        validation_split: float | list | tuple | None = 0.2,
        validation_group_column: str | None = None,
        early_stopping_patience: int | None = None,
        dropout_rate: float = 0.5,
        device: str = "cpu",
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.X_mean_: float | None = None
        self.X_std_: float | None = None
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
            verbose=verbose,
        )

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        if fit:
            self.X_mean_, self.X_std_ = fit_global_zscore_stats(X[train_idx])
        if self.X_mean_ is None or self.X_std_ is None:
            raise ValueError("Input normalization stats are not initialized.")
        return apply_global_zscore(X, self.X_mean_, self.X_std_).astype(np.float32)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        return EEGNetModel(
            n_classes=n_classes,
            n_channels=int(features.shape[1]),
            dropout_rate=self.dropout_rate,
            **kwargs,
        )
