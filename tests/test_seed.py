"""Determinism: same seed → same data and same baseline predictions."""

from __future__ import annotations

import numpy as np

from motioncode.data.splits import grouped_split
from motioncode.data.synthetic import SyntheticConfig, generate_synthetic
from motioncode.models.baselines import train_baselines
from motioncode.preprocess import preprocess_batch


def test_synthetic_is_deterministic():
    a = generate_synthetic(SyntheticConfig(n_per_class=8, length=128, seed=99))
    b = generate_synthetic(SyntheticConfig(n_per_class=8, length=128, seed=99))
    c = generate_synthetic(SyntheticConfig(n_per_class=8, length=128, seed=100))
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, c.X)


def test_baseline_predictions_deterministic():
    batch = generate_synthetic(SyntheticConfig(n_per_class=16, length=128, seed=42))
    clean = preprocess_batch(batch)
    splits = grouped_split(clean, seed=42)

    def fit_and_pred():
        models = train_baselines(
            splits.train.X,
            splits.train.y,
            fs=clean.fs,
            seed=42,
            which=["logistic", "forest"],
        )
        return {k: v.predict(splits.test.X) for k, v in models.items()}

    p1 = fit_and_pred()
    p2 = fit_and_pred()
    for name in p1:
        assert np.array_equal(p1[name], p2[name]), name
