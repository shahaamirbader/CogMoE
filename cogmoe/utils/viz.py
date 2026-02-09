"""
Visualization helpers for CogMoE: standard plots for preprocessing inspection.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(original, processed, fs, title, labels=('Original','Processed')):
    """
    Plot a 1D signal before and after processing.

    Args:
        original (np.ndarray): Original time-domain signal.
        processed (np.ndarray): Processed or reconstructed signal.
        fs        (float): Sampling frequency (Hz).
        title     (str): Plot title.
        labels    (tuple): Legend labels.
    """
    t = np.arange(len(original)) / fs
    plt.figure(figsize=(10,3))
    plt.plot(t, original, label=labels[0], alpha=0.6)
    plt.plot(t, processed, label=labels[1], linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cwt_map(cwt_matrix, freqs, fs, title, cmap='viridis'):
    """
    Plot magnitude of a CWT coefficient matrix.

    Args:
        cwt_matrix (np.ndarray): CWT coefficients (freqs x time).
        freqs      (np.ndarray): Frequency axis values.
        fs         (float): Sampling rates used in transform.
        title      (str): Plot title.
        cmap       (str): Matplotlib colormap.
    """
    plt.figure(figsize=(8,5))
    plt.imshow(np.abs(cwt_matrix),
               aspect='auto',
               origin='lower',
               extent=[0, cwt_matrix.shape[1] / fs, freqs[0], freqs[-1]],
               cmap=cmap)
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def plot_mask(mask, freqs, fs, title):
    """
    Visualize a boolean mask over time-frequency.

    Args:
        mask  (np.ndarray): Boolean array (freqs x time).
        freqs (np.ndarray): Frequency axis.
        fs    (float): Sampling freq used to compute TF size.
        title (str): Plot title.
    """
    plt.figure(figsize=(8,4))
    plt.imshow(mask.astype(int),
               aspect='auto',
               origin='lower',
               extent=[0, mask.shape[1] / fs, freqs[0], freqs[-1]],
               cmap='gray_r')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
