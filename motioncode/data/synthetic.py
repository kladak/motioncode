"""Offline ECG-like waveform generator.

The shapes are Gaussian P/QRS/T caricatures plus class-specific RR schedules.
They are good enough to stress a pipeline. They are not lead-II physiology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motioncode.data.schema import CLASSES, CLASS_TO_ID, RecordBatch


@dataclass(frozen=True)
class SyntheticConfig:
    n_per_class: int = 60
    length: int = 512
    fs: float = 128.0
    seed: int = 42


def _gaussian(t: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _beat_template(
    fs: float,
    qrs_width_s: float,
    qrs_amp: float,
    t_amp: float,
    p_amp: float,
) -> np.ndarray:
    """One P-QRS-T caricature, ~600 ms, zero-padded by the caller."""
    duration = 0.60
    t = np.arange(0.0, duration, 1.0 / fs)
    qrs_sigma = max(qrs_width_s / 2.35, 1.5 / fs)
    p = _gaussian(t, 0.12, 0.025, p_amp)
    q = _gaussian(t, 0.20, qrs_sigma * 0.45, -0.18 * qrs_amp)
    r = _gaussian(t, 0.22, qrs_sigma, qrs_amp)
    s = _gaussian(t, 0.25, qrs_sigma * 0.50, -0.28 * qrs_amp)
    tw = _gaussian(t, 0.40, 0.050, t_amp)
    return (p + q + r + s + tw).astype(np.float32)


def _rr_schedule(
    class_name: str,
    n_beats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return RR intervals in seconds."""
    if class_name == "regular":
        base = rng.uniform(0.75, 0.95)
        return rng.normal(base, 0.012, size=n_beats).clip(0.55, 1.3)
    if class_name == "irregular":
        # High RR variance; still positive. Not a model of AF electrophysiology.
        return rng.uniform(0.42, 1.15, size=n_beats)
    if class_name == "wide":
        base = rng.uniform(0.80, 1.00)
        return rng.normal(base, 0.014, size=n_beats).clip(0.60, 1.3)
    if class_name == "burst":
        base = rng.uniform(0.75, 0.95)
        rr = rng.normal(base, 0.012, size=n_beats).clip(0.55, 1.3)
        # Sprinkle premature-like short intervals.
        n_extra = max(2, n_beats // 6)
        idx = rng.choice(n_beats, size=n_extra, replace=False)
        rr[idx] = rng.uniform(0.32, 0.48, size=n_extra)
        return rr
    raise ValueError(f"unknown class {class_name}")


def _render_record(
    class_name: str,
    length: int,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if class_name == "wide":
        qrs_width = rng.uniform(0.14, 0.18)
        qrs_amp = rng.uniform(0.85, 1.15)
    else:
        qrs_width = rng.uniform(0.06, 0.09)
        qrs_amp = rng.uniform(0.90, 1.20)
    t_amp = rng.uniform(0.18, 0.32)
    p_amp = rng.uniform(0.08, 0.14)
    tmpl = _beat_template(fs, qrs_width, qrs_amp, t_amp, p_amp)

    n_beats = 18
    rr = _rr_schedule(class_name, n_beats, rng)
    x = np.zeros(length, dtype=np.float32)
    t = 0.15 * fs  # start a bit after t=0
    for interval in rr:
        start = int(round(t))
        if start >= length:
            break
        end = min(length, start + tmpl.size)
        x[start:end] += tmpl[: end - start]
        t += interval * fs

    # Baseline wander + sensor-ish noise. Burst gets a bit more muscle noise.
    t_idx = np.arange(length, dtype=np.float32) / fs
    wander = 0.04 * np.sin(2 * np.pi * rng.uniform(0.15, 0.35) * t_idx + rng.uniform(0, 6))
    noise_std = 0.045 if class_name == "burst" else 0.025
    noise = rng.normal(0.0, noise_std, size=length)
    x = x + wander.astype(np.float32) + noise.astype(np.float32)
    return x.astype(np.float32)


def generate_synthetic(cfg: SyntheticConfig | None = None) -> RecordBatch:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_per_class * len(CLASSES)
    X = np.zeros((n, cfg.length), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    record_id = np.empty(n, dtype=object)
    class_name = np.empty(n, dtype=object)

    i = 0
    for name in CLASSES:
        for k in range(cfg.n_per_class):
            X[i] = _render_record(name, cfg.length, cfg.fs, rng)
            y[i] = CLASS_TO_ID[name]
            record_id[i] = f"syn-{name}-{k:04d}"
            class_name[i] = name
            i += 1

    # Shuffle so class blocks do not leak into naive head/tail splits.
    perm = rng.permutation(n)
    return RecordBatch(
        X=X[perm],
        y=y[perm],
        record_id=np.asarray(record_id[perm], dtype="U32"),
        class_name=np.asarray(class_name[perm], dtype="U16"),
        fs=float(cfg.fs),
        source="synthetic",
    )
