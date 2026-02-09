"""
Anti-alias filtering and resampling utilities for CogMoE.
"""
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


def anti_aliased_resample(signal_data, orig_fs, target_fs, order=5):
    """
    Resample a 1D signal from orig_fs to target_fs with anti-alias filtering.

    Args:
        signal_data (np.ndarray): Input signal.
        orig_fs     (float): Original sampling rate (Hz).
        target_fs   (float): Desired sampling rate (Hz).
        order       (int): Butterworth filter order.
    Returns:
        np.ndarray: Resampled signal.
    """
    # If upsampling, no anti-alias needed
    if target_fs >= orig_fs:
        # Nearest polyphase resample suffices
        gcd = np.gcd(int(orig_fs), int(target_fs))
        up = int(target_fs) // gcd
        down = int(orig_fs) // gcd
        return resample_poly(signal_data, up, down)

    # Design lowpass filter with cutoff at half of target
    nyq = 0.5 * orig_fs
    cutoff = 0.5 * target_fs
    b, a = butter(order, cutoff/nyq, btype='low')
    filtered = filtfilt(b, a, signal_data)

    # Polyphase resample
    gcd = np.gcd(int(orig_fs), int(target_fs))
    up = int(target_fs) // gcd
    down = int(orig_fs) // gcd
    resampled = resample_poly(filtered, up, down)
    return resampled
