"""Custom sklearn-compatible classifiers for local MOABB runs."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

try:
    from coheriqs_contributions.moabb_pipelines.common import (
        prepare_cwt_tf,
        resolve_coherence_utils,
        upper_pair_indices,
        validate_eeg_X,
    )
except ModuleNotFoundError:
    from moabb_pipelines.common import (
        prepare_cwt_tf,
        resolve_coherence_utils,
        upper_pair_indices,
        validate_eeg_X,
    )


log = logging.getLogger(__name__)


class WaveletTransformClassifier(ClassifierMixin, BaseEstimator):
    """Random forest over per-channel CWT and pairwise coherence statistics."""

    def __init__(
        self,
        lowest=4,
        highest=40,
        nfreqs=50,
        sampling_rate=250,
        cwt_resample_n_time=100,
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    ):
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.sampling_rate = sampling_rate
        self.cwt_resample_n_time = cwt_resample_n_time
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.transform_ = None
        self.coherence_ = None

    def _helpers(self):
        if self.transform_ is None or self.coherence_ is None:
            self.transform_, self.coherence_ = resolve_coherence_utils()
        return self.transform_, self.coherence_

    def _append_scale_stats(self, values: np.ndarray, out: list[float]) -> None:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.ndim == 1 and values.shape[0] == self.nfreqs:
            values = values[:, None]
        elif values.ndim == 2 and values.shape[1] == self.nfreqs:
            values = values.T
        elif values.ndim != 2 or values.shape[0] != self.nfreqs:
            out.extend([0.0] * (self.nfreqs * 3))
            return

        out.extend(np.mean(values, axis=1).tolist())
        out.extend(np.max(values, axis=1).tolist())
        out.extend(np.std(values, axis=1).tolist())

    def _compute_wavelet_features(self, X):
        X = validate_eeg_X(X)
        n_samples, n_channels, _ = X.shape
        n_time = int(self.cwt_resample_n_time)
        if n_time <= 0:
            raise ValueError("cwt_resample_n_time must be a positive integer.")

        transform_fn, coherence_fn = self._helpers()
        pairs = upper_pair_indices(n_channels)
        total_transforms = n_samples * n_channels
        total_coherences = n_samples * len(pairs)

        log.info(
            "Computing wavelet RF features: samples=%s channels=%s transforms=%s coherences=%s",
            n_samples,
            n_channels,
            total_transforms,
            total_coherences,
        )
        if self.verbose:
            print(
                f"   Starting {total_transforms} CWT transforms and "
                f"{total_coherences} coherence computations...",
                flush=True,
            )

        wavelet_coeffs: dict[tuple[int, int], tuple[np.ndarray | None, np.ndarray | None]] = {}
        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                try:
                    coeffs, freqs = transform_fn(
                        X[sample_idx, ch_idx, :],
                        self.sampling_rate,
                        self.highest,
                        self.lowest,
                        nfreqs=self.nfreqs,
                    )
                    coeffs_ft = prepare_cwt_tf(
                        coeffs,
                        nfreqs=self.nfreqs,
                        n_time=n_time,
                    ).T
                    wavelet_coeffs[(sample_idx, ch_idx)] = (coeffs_ft, freqs)
                except Exception as exc:
                    log.debug(
                        "CWT failed for sample=%s channel=%s: %s",
                        sample_idx,
                        ch_idx,
                        exc,
                    )
                    wavelet_coeffs[(sample_idx, ch_idx)] = (None, None)

        features = []
        iterator = tqdm(range(n_samples), disable=not self.verbose, leave=False)
        for sample_idx in iterator:
            sample_features: list[float] = []

            for ch_idx in range(n_channels):
                coeffs, _ = wavelet_coeffs[(sample_idx, ch_idx)]
                if coeffs is None:
                    sample_features.extend([0.0] * (self.nfreqs * 3))
                else:
                    self._append_scale_stats(np.abs(coeffs), sample_features)

            for ch_i, ch_j in pairs:
                coeffs_i, freqs_i = wavelet_coeffs[(sample_idx, ch_i)]
                coeffs_j, _ = wavelet_coeffs[(sample_idx, ch_j)]
                if coeffs_i is None or coeffs_j is None or freqs_i is None:
                    sample_features.extend([0.0] * (self.nfreqs * 3))
                    continue
                try:
                    coh, _, _ = coherence_fn(coeffs_i, coeffs_j, freqs_i)
                    self._append_scale_stats(np.asarray(coh), sample_features)
                except Exception as exc:
                    log.debug(
                        "Coherence failed for sample=%s channels=(%s,%s): %s",
                        sample_idx,
                        ch_i,
                        ch_j,
                        exc,
                    )
                    sample_features.extend([0.0] * (self.nfreqs * 3))

            features.append(sample_features)

        X_features = np.asarray(features, dtype=np.float32)
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
        log.info("Computed wavelet RF features shape: %s", X_features.shape)
        if self.verbose:
            print("   Completed CWT and coherence feature extraction.", flush=True)
        return X_features

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_features = self._compute_wavelet_features(X)
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model_.fit(X_features, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model_.predict(self._compute_wavelet_features(X))

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model_.predict_proba(self._compute_wavelet_features(X))

    def score(self, X, y):
        return float(accuracy_score(y, self.predict(X)))
