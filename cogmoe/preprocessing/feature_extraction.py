# File: preprocessing/feature_extraction.py

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy as _entropy
import neurokit2 as nk

def extract_eeg_features(signal, fs):
    """
    Extract EEG features for one channel:
      – Raw stats: mean, min, max, median, var, std
      – Spectral entropy
      – Hjorth mobility & complexity
      – Nonlinear complexity: LZ and Higuchi FD
      – Band‐power stats for Δ(0.5–4), θ(4–8), α(8–12), β(12–30), γ(30–75)
    Args:
        signal (1D np.ndarray), fs (float)
    Returns:
        pd.Series of ~40 features
    """
    feats = {}
    x = np.nan_to_num(signal, nan=np.nanmean(signal))

    # Time‐domain
    feats['eeg_mean']   = np.mean(x)
    feats['eeg_min']    = np.min(x)
    feats['eeg_max']    = np.max(x)
    feats['eeg_median'] = np.median(x)
    feats['eeg_var']    = np.var(x)
    feats['eeg_std']    = np.std(x)

    # PSD via Welch
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), 256))
    psd_norm = psd / np.sum(psd)
    feats['eeg_spectral_entropy'] = _entropy(psd_norm)

    # Hjorth parameters
    hj = nk.complexity.hjorth(x)
    feats['eeg_hjorth_mobility']   = hj['Mobility']
    feats['eeg_hjorth_complexity'] = hj['Complexity']

    # Nonlinear complexity
    feats['eeg_lz_complexity']     = nk.complexity.lempel_ziv_complexity(x)
    feats['eeg_higuchi_fd']        = nk.complexity.higuchi_fd(x)

    # Band‐power statistics
    bands = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  12),
        'beta':  (12, 30),
        'gamma': (30, 75)
    }
    for band, (f_lo, f_hi) in bands.items():
        idx = np.logical_and(freqs >= f_lo, freqs <= f_hi)
        bp = psd[idx]
        feats[f'eeg_{band}_power']        = np.sum(bp)
        feats[f'eeg_{band}_power_mean']   = np.mean(bp) if bp.size else 0.0
        feats[f'eeg_{band}_power_max']    = np.max(bp) if bp.size else 0.0
        feats[f'eeg_{band}_power_min']    = np.min(bp) if bp.size else 0.0
        feats[f'eeg_{band}_power_median'] = np.median(bp) if bp.size else 0.0

    return pd.Series(feats)


def extract_ecg_features(ecg_signal, fs):
    """
    Extract ECG/HRV features via NeuroKit2:
      – Time‐domain HRV
      – Poincaré metrics
      – Nonlinear HRV (entropy)
    Args:
        ecg_signal (1D np.ndarray), fs (float)
    Returns:
        pd.Series of ~53 features
    """
    # Process ECG to find R‐peaks
    processed = nk.ecg_process(ecg_signal, sampling_rate=fs)
    peaks     = processed[1]['ECG_R_Peaks']

    # Time‐domain HRV
    hrv_time     = nk.hrv_time(peaks, sampling_rate=fs, show=False)
    # Poincaré
    hrv_poincare = nk.hrv_poincare(peaks, sampling_rate=fs, show=False)
    # Nonlinear
    hrv_nonlinear= nk.hrv_nonlinear(peaks, sampling_rate=fs, show=False)

    # Merge into one series
    df = pd.concat([hrv_time, hrv_poincare, hrv_nonlinear], axis=1)
    return df.iloc[0].add_prefix('ecg_')


def extract_eda_features(eda_signal, fs):
    """
    Extract EDA features on raw, phasic, and tonic components via NeuroKit2:
      – For each component: mean, median, std, skew, kurtosis,
        entropy (histogram), IQR, AUC, squared AUC, MAD
    Args:
        eda_signal (1D np.ndarray), fs (float)
    Returns:
        pd.Series of ~30 features
    """
    signals, info = nk.eda_process(eda_signal, sampling_rate=fs)
    comps = {
        'raw':   signals['EDA_Raw'],
        'phasic':signals['EDA_Phasic'],
        'tonic': signals['EDA_Tonic']
    }
    feats = {}
    for label, x in comps.items():
        x = np.nan_to_num(x, nan=np.nanmean(x))
        feats[f'eda_{label}_mean']   = np.mean(x)
        feats[f'eda_{label}_median'] = np.median(x)
        feats[f'eda_{label}_std']    = np.std(x)
        feats[f'eda_{label}_skew']   = skew(x)
        feats[f'eda_{label}_kurtosis']= kurtosis(x)

        # Shannon entropy of amplitude distribution
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        feats[f'eda_{label}_entropy'] = -np.sum(hist * np.log(hist))

        feats[f'eda_{label}_iqr']    = np.percentile(x, 75) - np.percentile(x, 25)
        feats[f'eda_{label}_auc']    = np.trapz(np.abs(x))
        feats[f'eda_{label}_auc2']   = np.trapz(x**2)
        feats[f'eda_{label}_mad']    = np.median(np.abs(x - np.median(x)))

    return pd.Series(feats)


def extract_gaze_features(gaze_dict, fs):
    """
    Extract Gaze/Event features:
      – Pupil: max, min, mean
      – Blink: count, max duration, mean duration
      – Fixation: count, max/min/mean duration; dispersion max/min/mean
      – Saccade: count, duration max/min/mean; amplitude max/min/mean;
                 peak velocity/accel/decel max/min/mean; direction max/min/mean
    Args:
        gaze_dict (dict): {
            'pupil': 1D np.ndarray,
            'blink_mask': 1D bool mask,
            'fixation_mask': 1D bool mask,
            'dispersion': 1D np.ndarray,
            'saccade_amplitude': 1D np.ndarray,
            'saccade_duration': 1D np.ndarray,
            'saccade_peak_velocity': 1D np.ndarray,
            'saccade_peak_acceleration': 1D np.ndarray,
            'saccade_peak_deceleration': 1D np.ndarray,
            'saccade_direction': 1D np.ndarray
        }
        fs (float): sampling rate for mask→duration conversion
    Returns:
        pd.Series of ~32 features
    """
    feats = {}
    # Pupil
    pupil = gaze_dict.get('pupil', np.array([]))
    if pupil.size:
        feats['gaze_pupil_max']  = np.max(pupil)
        feats['gaze_pupil_min']  = np.min(pupil)
        feats['gaze_pupil_mean'] = np.mean(pupil)

    def event_stats(mask, name):
        # mask: bool array where event==True
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends   = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends   = np.r_[ends, mask.size]
        durations = (ends - starts) / fs
        return len(durations), durations.max() if durations.size else 0.0, durations.mean() if durations.size else 0.0

    # Blink
    b_cnt, b_max, b_mean = event_stats(gaze_dict.get('blink_mask', np.zeros(0, dtype=bool)), 'blink')
    feats['gaze_blink_count']         = b_cnt
    feats['gaze_blink_max_duration']  = b_max
    feats['gaze_blink_mean_duration'] = b_mean

    # Fixation
    f_cnt, f_max, f_mean = event_stats(gaze_dict.get('fixation_mask', np.zeros(0, dtype=bool)), 'fixation')
    feats['gaze_fixation_count']        = f_cnt
    feats['gaze_fixation_max_duration'] = f_max
    feats['gaze_fixation_mean_duration']= f_mean

    # Dispersion
    disp = gaze_dict.get('dispersion', np.array([]))
    if disp.size:
        feats['gaze_dispersion_max']   = np.max(disp)
        feats['gaze_dispersion_min']   = np.min(disp)
        feats['gaze_dispersion_mean']  = np.mean(disp)

    # Saccade events (arrays per event)
    sac_keys = [
        'saccade_amplitude', 'saccade_duration',
        'saccade_peak_velocity', 'saccade_peak_acceleration',
        'saccade_peak_deceleration', 'saccade_direction'
    ]
    for key in sac_keys:
        arr = gaze_dict.get(key, np.array([]))
        if arr.size:
            feats[f'gaze_{key}_count'] = arr.size
            feats[f'gaze_{key}_max']   = np.max(arr)
            feats[f'gaze_{key}_min']   = np.min(arr)
            feats[f'gaze_{key}_mean']  = np.mean(arr)

    return pd.Series(feats)
