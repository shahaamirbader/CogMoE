"""
2D cross-correlation alignment for CogMoE preprocessing.
"""

import numpy as np
from scipy.signal import correlate2d


def find_shifts(W_ref, W_cur, prefer_small_shifts=True, top_ratio=0.9):
    """
    Compute optimal time and frequency shifts between two CWT magnitude maps via 2D cross-correlation.

    Args:
        W_ref (np.ndarray): Reference CWT magnitude map (freqs x time).
        W_cur (np.ndarray): Current modality CWT magnitude map (freqs x time).
        prefer_small_shifts (bool): If True, among top correlations favor smallest net shift.
        top_ratio (float): Ratio of global max to include for small-shift selection.

    Returns:
        time_shift (int): Number of time samples to shift W_cur to align with W_ref.
        freq_shift (int): Number of frequency bins to shift W_cur to align with W_ref.
    """
    # Compute full 2D cross-correlation
    corr2d = correlate2d(W_ref, W_cur, mode='full')
    # All local peaks detection
    # Pad boundaries to avoid edge effects
    peaks = []  # list of tuples (value, freq_offset, time_offset)
    f0, t0 = W_ref.shape[0] - 1, W_ref.shape[1] - 1
    # Search interior for local maxima
    for i in range(1, corr2d.shape[0] - 1):
        for j in range(1, corr2d.shape[1] - 1):
            val = corr2d[i, j]
            if val > corr2d[i-1, j] and val > corr2d[i+1, j] and val > corr2d[i, j-1] and val > corr2d[i, j+1]:
                freq_off = i - f0
                time_off = j - t0
                peaks.append((val, freq_off, time_off))

    # If no local, use global max
    if not peaks:
        idx = np.unravel_index(np.argmax(corr2d), corr2d.shape)
        freq_off = idx[0] - f0
        time_off = idx[1] - t0
        return time_off, freq_off

    # Sort peaks by correlation strength
    peaks.sort(key=lambda x: x[0], reverse=True)
    max_val = peaks[0][0]

    # Select among top peaks
    if prefer_small_shifts:
        # Filter within top_ratio of max
        candidates = [p for p in peaks if p[0] >= top_ratio * max_val]
        # Choose by smallest Euclidean shift
        _, freq_off, time_off = min(candidates, key=lambda p: (p[1]**2 + p[2]**2))
    else:
        _, freq_off, time_off = peaks[0]

    return time_off, freq_off


def shift_matrix(mat, time_shift, freq_shift):
    """
    Shift a 2D matrix along time (axis=1) and frequency (axis=0) by padding with zeros.

    Args:
        mat (np.ndarray): Input 2D array (freqs x time).
        time_shift (int): Positive shifts right, negative left.
        freq_shift (int): Positive shifts down, negative up.

    Returns:
        shifted (np.ndarray): Shifted matrix of same shape.
    """
    freqs, times = mat.shape
    shifted = np.zeros_like(mat)

    # Frequency shift
    if freq_shift >= 0:
        f_src_start = 0
        f_src_end = freqs - freq_shift
        f_dst_start = freq_shift
        f_dst_end = freqs
    else:
        f_src_start = -freq_shift
        f_src_end = freqs
        f_dst_start = 0
        f_dst_end = freqs + freq_shift
    # Time shift
    if time_shift >= 0:
        t_src_start = 0
        t_src_end = times - time_shift
        t_dst_start = time_shift
        t_dst_end = times
    else:
        t_src_start = -time_shift
        t_src_end = times
        t_dst_start = 0
        t_dst_end = times + time_shift

    shifted[f_dst_start:f_dst_end, t_dst_start:t_dst_end] = \
        mat[f_src_start:f_src_end, t_src_start:t_src_end]

    return shifted


def align_cwt_maps(W_maps, anchor_key=None, prefer_small_shifts=True):
    """
    Align multiple modalities' CWT maps to a common reference via 2D shifts.

    Args:
        W_maps (dict): {modality: np.ndarray (freqs x time)} of magnitude maps.
        anchor_key (str): Key of reference modality. Defaults to first in dict.
        prefer_small_shifts (bool): As above.

    Returns:
        aligned_maps (dict): Same keys as W_maps, but shifted arrays.
        shifts (dict): {modality: (time_shift, freq_shift)} relative to anchor.
    """
    keys = list(W_maps.keys())
    if not keys:
        return {}, {}
    if anchor_key is None:
        anchor_key = keys[0]

    aligned = {anchor_key: W_maps[anchor_key].copy()}
    shifts = {anchor_key: (0, 0)}

    # Align each other modality
    for k in keys:
        if k == anchor_key:
            continue
        t_off, f_off = find_shifts(
            np.abs(W_maps[anchor_key]),
            np.abs(W_maps[k]),
            prefer_small_shifts=prefer_small_shifts
        )
        shifts[k] = (t_off, f_off)
        aligned[k] = shift_matrix(W_maps[k], t_off, f_off)

    return aligned, shifts
