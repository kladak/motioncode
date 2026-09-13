"""Shape and schema contracts."""

from __future__ import annotations

import numpy as np
import pytest

from motioncode.data.schema import CLASSES, RecordBatch
from motioncode.data.synthetic import SyntheticConfig, generate_synthetic
from motioncode.features import FEATURE_NAMES, extract_features
from motioncode.preprocess import preprocess_batch


def test_synthetic_shapes_and_unique_ids():
    batch = generate_synthetic(SyntheticConfig(n_per_class=5, length=256, seed=11))
    assert batch.X.shape == (20, 256)
    assert batch.y.shape == (20,)
    assert batch.X.dtype == np.float32
    assert len(set(batch.record_id.tolist())) == 20
    assert set(batch.class_name.tolist()) == set(CLASSES)


def test_record_batch_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        RecordBatch(
            X=np.zeros((2, 8), dtype=np.float32),
            y=np.zeros(2, dtype=np.int64),
            record_id=np.array(["a", "a"], dtype="U8"),
            class_name=np.array(["regular", "regular"], dtype="U16"),
            fs=128.0,
            source="synthetic",
        )


def test_preprocess_preserves_shape():
    batch = generate_synthetic(SyntheticConfig(n_per_class=4, length=256, seed=2))
    out = preprocess_batch(batch)
    assert out.X.shape == batch.X.shape
    # Per-record z-score → near unit variance.
    assert np.allclose(out.X.std(axis=1), 1.0, atol=0.05)


def test_feature_matrix_shape():
    batch = generate_synthetic(SyntheticConfig(n_per_class=3, length=256, seed=5))
    feats = extract_features(batch.X, batch.fs)
    assert feats.shape == (batch.n, len(FEATURE_NAMES))
    assert np.isfinite(feats).all()
