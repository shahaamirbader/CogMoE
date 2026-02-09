"""
Unit tests for preprocessing pipeline components.
"""

import numpy as np
import pytest


class TestCWT:
    def test_cwt_output_shape(self):
        from cogmoe.preprocessing.cwt import compute_cwt_map
        signal = np.sin(np.linspace(0, 10, 1000))
        cwt_mat, freqs = compute_cwt_map(signal, fs=100, num_freqs=32)
        assert cwt_mat.shape[0] == 32
        assert cwt_mat.shape[1] == len(signal)
        assert len(freqs) == 32

    def test_cwt_roundtrip(self):
        from cogmoe.preprocessing.cwt import compute_cwt_map, inverse_cwt
        signal = np.sin(np.linspace(0, 10, 500))
        cwt_mat, _ = compute_cwt_map(signal, fs=100)
        reconstructed = inverse_cwt(cwt_mat, fs=100)
        assert reconstructed.shape == signal.shape

    def test_handles_nans(self):
        from cogmoe.preprocessing.cwt import compute_cwt_map
        signal = np.sin(np.linspace(0, 10, 500))
        signal[100:110] = np.nan
        cwt_mat, freqs = compute_cwt_map(signal, fs=100)
        assert not np.any(np.isnan(cwt_mat))


class TestAlignment:
    def test_identity_alignment(self):
        from cogmoe.preprocessing.align import align_cwt_maps
        W = np.random.randn(32, 100)
        maps = {"A": W.copy(), "B": W.copy()}
        aligned, shifts = align_cwt_maps(maps, anchor_key="A")
        assert shifts["A"] == (0, 0)

    def test_shifted_recovery(self):
        from cogmoe.preprocessing.align import shift_matrix
        mat = np.eye(10)
        shifted = shift_matrix(mat, time_shift=2, freq_shift=0)
        assert shifted.shape == mat.shape


class TestMask:
    def test_mask_shape(self):
        from cogmoe.preprocessing.mask import generate_masks
        maps = {
            "A": np.random.randn(32, 100),
            "B": np.random.randn(32, 100),
        }
        masks = generate_masks(maps, mask_k=1.0)
        assert masks["A"].shape == (32, 100)
        assert masks["A"].dtype == bool


class TestRecovery:
    def test_low_rank_completion(self):
        from cogmoe.preprocessing.recover import low_rank_completion
        # Create low-rank matrix
        mat = np.outer(np.random.randn(32), np.random.randn(100))
        mask = np.zeros((32, 100), dtype=bool)
        mask[10:15, 30:40] = True
        completed = low_rank_completion(mat, mask, rank_param=0.1)
        assert completed.shape == mat.shape
