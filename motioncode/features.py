"""Handcrafted features for the linear / forest baselines.

These are deliberately boring (rate stats, amplitude, spectral energy).
They exist so the CNN is not the only model on the board.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch


FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "skew",
    "kurtosis",
    "n_peaks",
    "mean_ibi",
    "std_ibi",
    "cv_ibi",
    "spec_low",
    "spec_mid",
    "spec_high",
    "spec_centroid",
)


def _safe_stats(x: np.ndarray) -> tuple[float, float, float]:
    m = float(x.mean())
    s = float(x.std())
    if s < 1e-12:
        return m, 0.0, 0.0
    z = (x - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return m, skew, kurt


def extract_features(X: np.ndarray, fs: float) -> np.ndarray:
    """Map (n, length) windows → (n, n_features) float64."""
    if X.ndim != 2:
        raise ValueError(f"expected (n, length), got {X.shape}")
    n = X.shape[0]
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i, row in enumerate(X):
        mean, skew, kurt = _safe_stats(row)
        std = float(row.std())
        rms = float(np.sqrt(np.mean(row**2)))
        ptp = float(row.max() - row.min())
        # Peak detection on a mild threshold relative to std.
        height = max(0.4 * std, 1e-3) if std > 0 else 1e-3
        peaks, _ = find_peaks(row, height=height, distance=max(int(0.25 * fs), 1))
        n_peaks = float(len(peaks))
        if len(peaks) >= 2:
            ibi = np.diff(peaks) / fs
            mean_ibi = float(ibi.mean())
            std_ibi = float(ibi.std())
            cv_ibi = float(std_ibi / mean_ibi) if mean_ibi > 1e-8 else 0.0
        else:
            mean_ibi = 0.0
            std_ibi = 0.0
            cv_ibi = 0.0
        freqs, psd = welch(row, fs=fs, nperseg=min(128, len(row)))
        psd_sum = float(psd.sum()) + 1e-12
        low = float(psd[(freqs >= 0.5) & (freqs < 5)].sum() / psd_sum)
        mid = float(psd[(freqs >= 5) & (freqs < 15)].sum() / psd_sum)
        high = float(psd[(freqs >= 15) & (freqs <= 40)].sum() / psd_sum)
        centroid = float(np.sum(freqs * psd) / psd_sum)
        out[i] = [
            mean,
            std,
            rms,
            ptp,
            skew,
            kurt,
            n_peaks,
            mean_ibi,
            std_ibi,
            cv_ibi,
            low,
            mid,
            high,
            centroid,
        ]
    return out
