"""Torch-based wavelet coherence utilities using fcwt."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make the repository package importable when this file is executed directly.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import fcwt
except ImportError:
    fcwt = None

try:
    from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
        _ordered_pair_indices,
    )
except ModuleNotFoundError:
    from moabb_pipelines.wct_phase_gnn_classifier import (
        _ordered_pair_indices,
    )


def gaussian_smooth_2d(tensor: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    """Apply 2D Gaussian smoothing (time and frequency dims).
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (batch, edges, time, freq).
    sigma : float
        Standard deviation of Gaussian kernel. Default 6.0 matches coherence_utils.py.
    
    Returns
    -------
    torch.Tensor
        Smoothed tensor of same shape.
    """
    batch, edges, time, freq = tensor.shape
    # Create Gaussian kernel for both dimensions
    size = int(6 * sigma + 1)
    x = torch.linspace(-3 * sigma, 3 * sigma, size, device=tensor.device, dtype=tensor.dtype)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply separable convolution
    # First: smooth along time dimension
    kernel_time = kernel_1d.view(1, 1, size, 1)
    pad_time = size // 2
    tensor_padded = F.pad(tensor.view(batch * edges, 1, time, freq), (0, 0, pad_time, pad_time), mode='constant', value=0)
    smoothed = F.conv2d(tensor_padded, kernel_time)
    smoothed = smoothed.view(batch, edges, -1, freq)
    
    # Second: smooth along frequency dimension
    kernel_freq = kernel_1d.view(1, 1, 1, size)
    pad_freq = size // 2
    smoothed_padded = F.pad(smoothed.view(batch * edges, 1, time, freq), (pad_freq, pad_freq, 0, 0), mode='constant', value=0)
    smoothed = F.conv2d(smoothed_padded, kernel_freq)
    smoothed = smoothed.view(batch, edges, time, freq)
    
    return smoothed


def compute_coherence_fcwt(
    raw_x: torch.Tensor,
    sampling_rate: int,
    lowest: float,
    highest: float,
    nfreqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute wavelet coherence using fcwt and Gaussian smoothing.
    
    Parameters
    ----------
    raw_x : torch.Tensor
        Raw EEG signal, shape (B, C, T) where B=batch, C=channels, T=time.
    sampling_rate : int
        Sampling rate in Hz.
    lowest : float
        Lowest frequency for CWT.
    highest : float
        Highest frequency for CWT.
    nfreqs : int
        Number of frequency bins.
    
    Returns
    -------
    coh : torch.Tensor
        Coherence, shape (B, E, F) where E=edges, F=frequencies.
    mean_phase : torch.Tensor
        Mean phase, shape (B, E, F).
    """
    if fcwt is None:
        raise ImportError("fcwt is required for coherence computation.")
    
    batch_size, n_channels, n_time = raw_x.shape
    device = raw_x.device
    dtype = raw_x.dtype
    
    # Compute CWT for each channel using fcwt
    src_idx, dst_idx = _ordered_pair_indices(n_channels)
    num_edges = src_idx.numel()
    
    # Process on CPU with fcwt, then move back to device
    raw_x_np = raw_x.detach().cpu().numpy().astype('float64')
    
    # Store all channel wavelets
    coeffs_all = np.zeros((batch_size, n_channels, nfreqs, n_time), dtype='complex128')
    
    for b in range(batch_size):
        for ch in range(n_channels):
            freqs, coeffs = fcwt.cwt(
                raw_x_np[b, ch, :],
                sampling_rate,
                lowest,
                highest,
                nfreqs,
                nthreads=4,
                scaling='log'
            )
            # Resample coefficients to match n_time if needed
            if coeffs.shape[1] != n_time:
                from scipy.interpolate import interp1d
                t_orig = np.linspace(0, 1, coeffs.shape[1])
                t_new = np.linspace(0, 1, n_time)
                coeffs_resampled = np.zeros((nfreqs, n_time), dtype='complex128')
                for f in range(nfreqs):
                    interp_fn = interp1d(t_orig, coeffs[f, :], kind='linear', fill_value='extrapolate')
                    coeffs_resampled[f, :] = interp_fn(t_new)
                coeffs_all[b, ch, :, :] = coeffs_resampled
            else:
                coeffs_all[b, ch, :, :] = coeffs
    
    # Convert to torch tensors
    coeffs_torch = torch.from_numpy(coeffs_all).to(device=device, dtype=torch.complex64)
    
    # Compute cross-wavelet and coherence
    xwt_real = (coeffs_torch[:, src_idx, :, :].real * coeffs_torch[:, dst_idx, :, :].real +
                 coeffs_torch[:, src_idx, :, :].imag * coeffs_torch[:, dst_idx, :, :].imag)
    xwt_imag = (coeffs_torch[:, src_idx, :, :].imag * coeffs_torch[:, dst_idx, :, :].real -
                 coeffs_torch[:, src_idx, :, :].real * coeffs_torch[:, dst_idx, :, :].imag)
    
    # Power spectra
    power_src = (coeffs_torch[:, src_idx, :, :].abs() ** 2).to(dtype=dtype)  # (B, E, F, T)
    power_dst = (coeffs_torch[:, dst_idx, :, :].abs() ** 2).to(dtype=dtype)  # (B, E, F, T)
    xwt_mag_sq = (xwt_real ** 2 + xwt_imag ** 2).to(dtype=dtype)  # (B, E, F, T)
    
    # Permute to (B, E, T, F) for gaussian smoothing
    power_src = power_src.permute(0, 1, 3, 2)  # (B, E, T, F)
    power_dst = power_dst.permute(0, 1, 3, 2)  # (B, E, T, F)
    xwt_mag_sq = xwt_mag_sq.permute(0, 1, 3, 2)  # (B, E, T, F)
    
    # Gaussian smooth
    power_src_smooth = gaussian_smooth_2d(power_src)
    power_dst_smooth = gaussian_smooth_2d(power_dst)
    xwt_mag_sq_smooth = gaussian_smooth_2d(xwt_mag_sq)
    
    # Coherence (now B, E, T, F)
    coh = xwt_mag_sq_smooth / (power_src_smooth * power_dst_smooth + 1e-12)
    coh = torch.clamp(coh, 0.0, 1.0)
    # Average over time to get (B, E, F)
    coh = coh.mean(dim=2)
    
    # Mean phase (ang is currently B, E, F, T - need to average over T to get B, E, F)
    ang = torch.atan2(xwt_imag.to(dtype=dtype), xwt_real.to(dtype=dtype))  # (B, E, F, T)
    mean_phase = ang.mean(dim=3)  # average over time (dim 3 in B, E, F, T)
    
    return coh, mean_phase
