"""
Cross-modal interpolation and low-rank completion for CogMoE preprocessing.
"""
import numpy as np


def low_rank_completion(matrix, mask, rank_param=0.1, max_iter=10):
    """
    Iterative SVD-based low-rank matrix completion on masked entries.

    Args:
        matrix (np.ndarray): 2D array (freqs x time) with masked entries unchanged.
        mask (np.ndarray of bool): True where entries should be reconstructed.
        rank_param (float): Fraction of singular values to retain (0-1).
        max_iter (int): Number of SVD iterations.

    Returns:
        np.ndarray: Completed matrix of same shape.
    """
    X = matrix.copy().astype(complex)
    # Initialize masked entries with zero
    X[mask] = 0

    for _ in range(max_iter):
        # Compute SVD
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        # Determine number of singular values to keep
        total = np.sum(s)
        if total == 0:
            break
        cum = np.cumsum(s) / total
        k = np.searchsorted(cum, 1 - rank_param) + 1
        k = min(max(k, 1), len(s))
        # Low-rank approximation
        X_approx = (U[:, :k] * s[:k]) @ Vh[:k, :]
        # Update only masked entries
        X[mask] = X_approx[mask]

    return X.real if not np.iscomplexobj(matrix) else X


def recover_cwt_maps(W_maps, masks, rank_param=0.1):
    """
    Recover masked high-energy regions via cross-modal interpolation then low-rank completion.

    Args:
        W_maps (dict): {modality: dict(channel: np.ndarray of shape (freqs, time))}
        masks  (dict): {modality: same shape bool array indicating masked entries}
        rank_param (float): Passed to low_rank_completion.

    Returns:
        recovered (dict): {modality: dict(channel: completed np.ndarray)}
    """
    recovered = {}
    modalities = list(W_maps.keys())

    for m in modalities:
        rec_ch = {}
        mask = masks[m]
        # prepare cross-modal average across all other modalities/channels
        others = [o for o in modalities if o != m]
        for ch, W in W_maps[m].items():
            # cross-modal interpolation: average corresponding pixels from others
            H = np.zeros_like(W, dtype=complex)
            count = 0
            for o in others:
                for W_o in W_maps[o].values():
                    H += W_o
                    count += 1
            if count > 0:
                H /= count
            # apply interpolation at masked loci
            W_interp = W.copy()
            W_interp[mask] = H[mask]
            # low-rank complete
            W_complete = low_rank_completion(W_interp, mask, rank_param)
            rec_ch[ch] = W_complete
        recovered[m] = rec_ch

    return recovered
