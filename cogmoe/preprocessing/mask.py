"""
Mask generation for CogMoE preprocessing: identify high-energy regions across modalities.
"""

import numpy as np

def generate_masks(W_maps, mask_k=1.0):
    """
    For each modality, generate a boolean mask of shape (freqs, time)
    marking regions where cross-modal energy is high.

    Args:
        W_maps (dict): {modality: dict(channel: np.ndarray of shape (freqs, time))}
                       or {modality: np.ndarray of shape (freqs, time)} for single-channel.
        mask_k (float): threshold multiplier; mask = H_sum > mean(H_sum) + k*std(H_sum).

    Returns:
        masks (dict): {modality: np.ndarray (freqs, time) of bool}
    """
    modalities = list(W_maps.keys())
    masks = {}

    # Determine common shape (min freqs & time across all maps)
    freqs = min(
        W_maps[m].shape[0] if isinstance(W_maps[m], np.ndarray) else \
        min(w.shape[0] for w in W_maps[m].values())
        for m in modalities
    )
    times = min(
        W_maps[m].shape[1] if isinstance(W_maps[m], np.ndarray) else \
        min(w.shape[1] for w in W_maps[m].values())
        for m in modalities
    )

    for m in modalities:
        # Accumulate squared magnitude from all other modalities
        H_sum = np.zeros((freqs, times), dtype=float)
        count = 0
        for o in modalities:
            if o == m:
                continue
            # Aggregate channels
            if isinstance(W_maps[o], dict):
                for w in W_maps[o].values():
                    W_mag = np.abs(w[:freqs, :times])
                    H_sum += W_mag**2
                    count += 1
            else:
                W_mag = np.abs(W_maps[o][:freqs, :times])
                H_sum += W_mag**2
                count += 1
        if count > 0:
            H_sum /= count
            thr = H_sum.mean() + mask_k * H_sum.std()
            masks[m] = H_sum > thr
        else:
            masks[m] = np.zeros((freqs, times), dtype=bool)

    return masks
