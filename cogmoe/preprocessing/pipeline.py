"""
End-to-end Stage 1 preprocessing pipeline for CogMoE.
Orchestrates: CWT -> Alignment -> Masking -> Recovery -> iCWT.
"""

import numpy as np
from cogmoe.preprocessing.cwt import calculate_scales, compute_cwt_map, inverse_cwt
from cogmoe.preprocessing.align import align_cwt_maps
from cogmoe.preprocessing.mask import generate_masks
from cogmoe.preprocessing.recover import recover_cwt_maps


class PreprocessingPipeline:
    """
    End-to-end Stage 1: raw signals -> CWT -> align -> mask -> recover -> iCWT.
    Used when running from raw signal arrays (before feature extraction).

    Args:
        cfg: dict - preprocessing configuration section from YAML config.
    """

    def __init__(self, cfg):
        self.cwt_cfg = cfg.get("cwt", {})
        self.align_cfg = cfg.get("alignment", {})
        self.mask_k = cfg.get("masking", {}).get("mask_k", 1.0)
        self.rank_param = cfg.get("recovery", {}).get("rank_param", 0.1)

    def process_segment(self, raw_signals, fs):
        """
        Process a single time-segment through CWT, alignment, masking, recovery, iCWT.

        Args:
            raw_signals: dict {modality_name: 1D np.ndarray} per modality.
            fs: float - sampling rate (after resampling to common rate).

        Returns:
            recovered_signals: dict {modality_name: 1D np.ndarray} - recovered signals.
        """
        f_min = self.cwt_cfg.get("f_min", 0.5)
        f_max = self.cwt_cfg.get("f_max", 40.0)
        num_freqs = self.cwt_cfg.get("num_freqs", 32)
        morlet_w = self.cwt_cfg.get("morlet_w", 5.0)

        # 1) Compute CWT magnitude maps
        W_maps = {}
        for mod, signal in raw_signals.items():
            cwt_mat, _ = compute_cwt_map(
                signal, fs=fs, f_min=f_min, f_max=f_max,
                num_freqs=num_freqs, morlet_w=morlet_w
            )
            W_maps[mod] = np.abs(cwt_mat)

        # 2) Align all modalities to anchor
        anchor = self.align_cfg.get("anchor", list(raw_signals.keys())[0])
        prefer_small = self.align_cfg.get("prefer_small_shifts", True)
        aligned, shifts = align_cwt_maps(
            W_maps, anchor_key=anchor, prefer_small_shifts=prefer_small
        )

        # 3) Generate cross-modal masks
        masks = generate_masks(aligned, mask_k=self.mask_k)

        # 4) Recover missing/high-energy regions
        wrapped = {m: {"ch0": aligned[m]} for m in aligned}
        rec_maps = recover_cwt_maps(wrapped, masks, rank_param=self.rank_param)

        # 5) Invert CWT back to time-domain signals
        recovered_signals = {}
        for m in rec_maps:
            recovered_signals[m] = inverse_cwt(
                rec_maps[m]["ch0"], fs=fs, f_min=f_min, f_max=f_max,
                num_freqs=num_freqs, morlet_w=morlet_w
            )

        return recovered_signals

    def process_dataset(self, all_segments, fs):
        """
        Process all segments through the full pipeline.

        Args:
            all_segments: list of dict {modality_name: 1D np.ndarray}.
            fs: float - common sampling rate.

        Returns:
            list of dict {modality_name: 1D np.ndarray} - recovered signals.
        """
        results = []
        for seg in all_segments:
            results.append(self.process_segment(seg, fs))
        return results
