"""
Unit tests for signal quality scoring.
"""

import numpy as np
import pytest


class TestSNR:
    def test_clean_signal(self):
        from cogmoe.utils.quality import compute_snr
        # High-variance signal should have high SNR
        signal = np.sin(np.linspace(0, 10, 100)) * 10
        snr = compute_snr(signal)
        assert snr > 0

    def test_empty_signal(self):
        from cogmoe.utils.quality import compute_snr
        snr = compute_snr(np.array([]))
        assert snr == 0.0


class TestMissingProportion:
    def test_complete(self):
        from cogmoe.utils.quality import compute_missing_proportion
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        p = compute_missing_proportion(signal)
        assert p == 1.0

    def test_with_nans(self):
        from cogmoe.utils.quality import compute_missing_proportion
        signal = np.array([1.0, np.nan, 3.0, 0.0])
        p = compute_missing_proportion(signal)
        assert 0 < p < 1


class TestAutocorrelation:
    def test_periodic_signal(self):
        from cogmoe.utils.quality import compute_autocorrelation
        # Highly autocorrelated signal
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        r = compute_autocorrelation(signal, lag_window=10)
        assert r > 0.5

    def test_random_signal(self):
        from cogmoe.utils.quality import compute_autocorrelation
        np.random.seed(42)
        signal = np.random.randn(200)
        r = compute_autocorrelation(signal, lag_window=10)
        # Random signal has near-zero autocorrelation
        assert r < 0.3


class TestQualityVector:
    def test_normalized(self):
        from cogmoe.utils.quality import compute_quality_vector
        feats = {
            "ECG": np.random.randn(33),
            "EEG": np.random.randn(16),
            "Gaze": np.random.randn(30),
            "EDA": np.random.randn(10),
        }
        q = compute_quality_vector(feats)
        assert q.shape == (4,)
        # Should sum to 1 (normalized)
        assert abs(q.sum() - 1.0) < 1e-6 or q.sum() == 0.0

    def test_quality_score_non_negative(self):
        from cogmoe.utils.quality import compute_quality_score
        score = compute_quality_score(np.random.randn(50))
        assert score >= 0
