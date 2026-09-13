"""CNN forward/train smoke on a tiny tensor."""

from __future__ import annotations

import numpy as np

from motioncode.models.cnn1d import CNN1D, train_cnn1d


def test_cnn_forward_shapes():
    model = CNN1D.init(n_classes=4, channels=(4, 8), seed=1)
    X = np.random.default_rng(0).normal(size=(5, 64)).astype(np.float32)
    probs, cache = model.forward(X)
    assert probs.shape == (5, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert cache["gap"].shape == (5, 8)


def test_cnn_trains_a_few_epochs():
    rng = np.random.default_rng(1)
    # Make two classes with different mean frequency so training can move.
    t = np.linspace(0, 1, 64, endpoint=False)
    X0 = np.sin(2 * np.pi * 3 * t)[None, :] + 0.05 * rng.normal(size=(20, 64))
    X1 = np.sin(2 * np.pi * 9 * t)[None, :] + 0.05 * rng.normal(size=(20, 64))
    X = np.concatenate([X0, X1]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)
    model, history = train_cnn1d(
        X[:30],
        y[:30],
        X[30:],
        y[30:],
        n_classes=2,
        channels=(4, 8),
        epochs=5,
        lr=0.05,
        batch_size=8,
        seed=0,
    )
    assert len(history) == 5
    preds = model.predict(X[30:])
    assert preds.shape == (10,)
