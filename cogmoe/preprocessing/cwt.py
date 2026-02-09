"""
CWT compute & invert for CogMoE preprocessing.
"""

import numpy as np
from scipy.signal import cwt, morlet2

# Admissibility constant for Morlet wavelet (approximate)
C_PSI = 0.776


def calculate_scales(fs, f_min=0.5, f_max=40.0, num_freqs=32, morlet_w=5.0):
    """
    Compute CWT scales and corresponding frequencies.
    
    Args:
        fs (float): Sampling rate of the signal (Hz).
        f_min (float): Minimum frequency (Hz).
        f_max (float): Maximum frequency (Hz).
        num_freqs (int): Number of frequency bins.
        morlet_w (float): Morlet wavelet width parameter.

    Returns:
        scales (np.ndarray): Scales for CWT.
        freqs  (np.ndarray): Frequencies corresponding to each scale.
    """
    dt = 1.0 / fs
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_freqs)
    scales = morlet_w / (2 * np.pi * freqs * dt)
    return scales, freqs


def compute_cwt_map(signal_data, fs, f_min=0.5, f_max=40.0, num_freqs=32, morlet_w=5.0):
    """
    Compute the continuous wavelet transform (CWT) of a 1D signal using a Morlet wavelet.

    Args:
        signal_data (np.ndarray): 1D input signal.
        fs (float): Sampling rate of the signal (Hz).
        f_min, f_max, num_freqs, morlet_w: see calculate_scales.

    Returns:
        cwt_matrix (np.ndarray): Complex CWT coefficients of shape (num_freqs, len(signal)).
        freqs      (np.ndarray): Frequencies corresponding to rows of cwt_matrix.
    """
    # Prepare scales and freqs
    scales, freqs = calculate_scales(fs, f_min, f_max, num_freqs, morlet_w)

    # Impute NaNs by mean
    if np.isnan(signal_data).any():
        signal_data = np.nan_to_num(signal_data, nan=np.nanmean(signal_data))

    # Compute CWT coefficients
    cwt_matrix = cwt(signal_data, morlet2, scales, w=morlet_w)
    return cwt_matrix, freqs


def inverse_cwt(cwt_matrix, fs, f_min=0.5, f_max=40.0, num_freqs=32, morlet_w=5.0):
    """
    Approximate inverse continuous wavelet transform to reconstruct a 1D signal.

    Args:
        cwt_matrix (np.ndarray): Complex CWT coefficients (num_freqs, time).
        fs (float): Sampling rate used for forward CWT.
        f_min, f_max, num_freqs, morlet_w: see calculate_scales.

    Returns:
        reconstructed (np.ndarray): Reconstructed 1D signal array.
    """
    n_samples = cwt_matrix.shape[1]
    scales, _ = calculate_scales(fs, f_min, f_max, num_freqs, morlet_w)

    # Sum real parts weighted by 1/sqrt(scale)
    reconstructed = np.zeros(n_samples, dtype=float)
    for i, s in enumerate(scales):
        reconstructed += np.real(cwt_matrix[i, :] / np.sqrt(s))

    # Normalize by admissibility constant and signal length
    reconstructed /= (C_PSI * np.sqrt(n_samples))
    return reconstructed
