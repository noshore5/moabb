"""Custom classifiers and components for benchmarking pipelines.

This module contains custom implementations for use in MOABB pipelines.
"""

import logging
import sys

import numpy as np
from scipy.signal import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, "../Coherent_Multiplex")
from utils.coherence_utils import coherence, transform


log = logging.getLogger(__name__)


class WaveletTransformClassifier(BaseEstimator, ClassifierMixin):
    """Classifier using Wavelet Transform (FCWT) for EEG feature extraction.

    This classifier applies Continuous Wavelet Transform (CWT) to EEG channels
    and extracts time-frequency features for classification.
    Uses the FCWT library from Coherent_Multiplex.

    Parameters
    ----------
    lowest : float, default=4
        Lowest frequency (Hz) for the wavelet transform

    highest : float, default=40
        Highest frequency (Hz) for the wavelet transform

    nfreqs : int, default=50
        Number of frequency scales for the wavelet transform

    sampling_rate : int, default=250
        Sampling rate of the EEG signal (Hz)

    Examples
    --------
    >>> clf = WaveletTransformClassifier(lowest=4, highest=40, nfreqs=50)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, lowest=4, highest=40, nfreqs=50, sampling_rate=250):
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.sampling_rate = sampling_rate
        self.model_ = None

    def _compute_wavelet_features(self, X):
        """Compute wavelet transform features from EEG data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            EEG data

        Returns
        -------
        X_features : array-like, shape (n_samples, n_wavelet_features)
            Wavelet-based features including individual channel features and
            coherence between unique channel pairs
        """
        n_samples, n_channels, n_timepoints = X.shape
        features_list = []

        # Calculate total operations: individual transforms + pairwise coherence
        individual_transforms = n_samples * n_channels
        unique_pairs = n_channels * (n_channels - 1) // 2
        pairwise_coherences = n_samples * unique_pairs
        total_operations = individual_transforms + pairwise_coherences

        log.info(
            f"Computing wavelet features for {n_samples} samples, {n_channels} channels"
        )
        log.info(
            f"Individual transforms: {individual_transforms}, Pairwise coherences: {pairwise_coherences}"
        )
        log.info(f"Total operations: {total_operations}")
        print(
            f"   Starting {individual_transforms} wavelet transforms and {pairwise_coherences} coherence computations...",
            flush=True,
        )

        # Pre-compute all wavelet transforms for all samples and channels
        log.info("Pre-computing all wavelet transforms...")
        wavelet_coeffs = {}  # Store (sample_idx, ch_idx) -> (coeffs, freqs)

        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                signal = X[sample_idx, ch_idx, :]

                try:
                    coeffs, freqs = transform(
                        signal,
                        self.sampling_rate,
                        self.highest,
                        self.lowest,
                        nfreqs=self.nfreqs,
                    )
                    # Downsample the wavelet coefficients to 100 timepoints
                    if coeffs.ndim == 2:
                        # Shape: (nfreqs, n_timepoints)
                        coeffs = resample(coeffs, 100, axis=1)
                    elif coeffs.ndim == 1:
                        # Shape: (n_timepoints,)
                        coeffs = resample(coeffs, 100)

                    wavelet_coeffs[(sample_idx, ch_idx)] = (coeffs, freqs)
                except Exception as e:
                    log.debug(
                        f"Error in wavelet transform for sample {sample_idx}, channel {ch_idx}: {e}"
                    )
                    wavelet_coeffs[(sample_idx, ch_idx)] = (None, None)

        log.info("Computing features from wavelet transforms and coherence...")

        for sample_idx in tqdm(range(n_samples)):
            sample_features = []

            # Extract features from individual channel wavelet transforms
            for ch_idx in range(n_channels):
                coeffs, freqs = wavelet_coeffs.get((sample_idx, ch_idx), (None, None))

                if coeffs is not None:
                    abs_coeffs = np.abs(coeffs)

                    if abs_coeffs.ndim == 2:
                        # 2D: (nfreqs, n_timepoints)
                        sample_features.extend(
                            np.mean(abs_coeffs, axis=1)
                        )  # mean per scale
                        sample_features.extend(
                            np.max(abs_coeffs, axis=1)
                        )  # max per scale
                        sample_features.extend(
                            np.std(abs_coeffs, axis=1)
                        )  # std per scale
                    elif abs_coeffs.ndim == 1:
                        # 1D: just extract stats
                        sample_features.append(np.mean(abs_coeffs))
                        sample_features.append(np.max(abs_coeffs))
                        sample_features.append(np.std(abs_coeffs))
                    else:
                        log.warning(f"Unexpected coefficient shape: {abs_coeffs.shape}")
                        sample_features.extend([0] * (self.nfreqs * 3))
                else:
                    # Fallback to zeros
                    sample_features.extend([0] * (self.nfreqs * 3))

            # Compute coherence for each unique pair of channels
            for ch_i in range(n_channels):
                for ch_j in range(ch_i + 1, n_channels):
                    coeffs_i, freqs_i = wavelet_coeffs.get(
                        (sample_idx, ch_i), (None, None)
                    )
                    coeffs_j, freqs_j = wavelet_coeffs.get(
                        (sample_idx, ch_j), (None, None)
                    )

                    if coeffs_i is not None and coeffs_j is not None:
                        try:
                            # Compute wavelet coherence between channels i and j
                            coh, _, _ = coherence(coeffs_i, coeffs_j, freqs_i)

                            # Extract statistics from coherence
                            if coh.ndim == 2:
                                # 2D coherence (nfreqs, n_timepoints)
                                sample_features.extend(
                                    np.mean(coh, axis=1)
                                )  # mean coherence per scale
                                sample_features.extend(
                                    np.max(coh, axis=1)
                                )  # max coherence per scale
                                sample_features.extend(
                                    np.std(coh, axis=1)
                                )  # std coherence per scale
                            elif coh.ndim == 1:
                                # 1D coherence
                                sample_features.append(np.mean(coh))
                                sample_features.append(np.max(coh))
                                sample_features.append(np.std(coh))
                            else:
                                log.warning(f"Unexpected coherence shape: {coh.shape}")
                                sample_features.extend([0] * (self.nfreqs * 3))
                        except Exception as e:
                            log.debug(
                                f"Error computing coherence for sample {sample_idx}, channels ({ch_i}, {ch_j}): {e}"
                            )
                            # Fallback to zeros
                            sample_features.extend([0] * (self.nfreqs * 3))
                    else:
                        # Missing data for one or both channels
                        sample_features.extend([0] * (self.nfreqs * 3))

            features_list.append(sample_features)

        X_features = np.array(features_list)
        log.info(f"Computed features shape: {X_features.shape}")
        print(
            "   ✓ Completed all wavelet transforms and coherence computations!",
            flush=True,
        )

        # Handle any NaN values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

        return X_features

    def fit(self, X, y):
        """Fit the classifier using wavelet features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Training EEG data
        y : array-like, shape (n_samples,)
            Target labels

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)

        # Compute wavelet features
        log.info("Computing wavelet features for training data...")
        X_features = self._compute_wavelet_features(X)
        log.info(f"Training classifier on features with shape: {X_features.shape}")

        # Train classifier on wavelet features
        self.model_ = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.model_.fit(X_features, y)
        log.info("Classifier training complete")

        return self

    def predict(self, X):
        """Predict class labels using wavelet features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test EEG data

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")

        # Compute wavelet features
        X_features = self._compute_wavelet_features(X)

        # Make predictions
        return self.model_.predict(X_features)

    def score(self, X, y):
        """Compute the mean accuracy score.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test data
        y : array-like, shape (n_samples,)
            Target labels

        Returns
        -------
        score : float
            Mean accuracy score
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))
