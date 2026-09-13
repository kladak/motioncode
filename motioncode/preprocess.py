"""Per-record cleanup. No statistics are fit on val/test."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

from motioncode.data.schema import RecordBatch


def _bandpass_sos(fs: float, low: float, high: float) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(low / nyq, 1e-4)
    hi = min(high / nyq, 0.99)
    if lo >= hi:
        raise ValueError(f"invalid bandpass {low}-{high} at fs={fs}")
    return butter(3, [lo, hi], btype="band", output="sos")


def preprocess_batch(
    batch: RecordBatch,
    *,
    do_detrend: bool = True,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    normalize: str = "zscore",
) -> RecordBatch:
    """Return a copy with cleaned windows.

    Normalization is per-record (mean/std of that window). That does not
    leak across the split. A global scaler, if added later, must be fit on
    train only.
    """
    X = batch.X.astype(np.float64, copy=True)
    if do_detrend:
        X = np.stack([detrend(row, type="linear") for row in X], axis=0)
    if bandpass is not None:
        low, high = bandpass
        # fs=128 → Nyquist 64; cap high at 40 or 0.45*fs, whichever is smaller.
        high = min(high, 0.45 * batch.fs)
        sos = _bandpass_sos(batch.fs, low, high)
        X = sosfiltfilt(sos, X, axis=1)
    if normalize == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = (X - mu) / sd
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"unknown normalize={normalize}")
    return RecordBatch(
        X=X.astype(np.float32),
        y=batch.y.copy(),
        record_id=batch.record_id.copy(),
        class_name=batch.class_name.copy(),
        fs=batch.fs,
        source=batch.source,
    )
