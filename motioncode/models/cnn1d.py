"""Tiny teaching 1D CNN in pure numpy.

Two conv blocks → ReLU → global average pool → dense → softmax.
No PyTorch/TensorFlow dependency so CI stays light. This is not a
literature architecture and is not meant to be compared to one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (B, C_in, L), w: (C_out, C_in, K), b: (C_out,) → (B, C_out, L-K+1)."""
    bsz, _, length = x.shape
    c_out, c_in, k = w.shape
    out_len = length - k + 1
    # im2col-ish via stride tricks would be nicer; keep it readable.
    out = np.zeros((bsz, c_out, out_len), dtype=np.float64)
    for oi in range(out_len):
        window = x[:, :, oi : oi + k]  # (B, C_in, K)
        # (B, C_out) = tensordot over (C_in, K)
        out[:, :, oi] = np.tensordot(window, w, axes=([1, 2], [1, 2])) + b
    return out


@dataclass
class CNN1D:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w_fc: np.ndarray
    b_fc: np.ndarray
    n_classes: int

    @classmethod
    def init(
        cls,
        *,
        n_classes: int,
        channels: tuple[int, int] = (8, 16),
        kernel: int = 7,
        seed: int = 42,
    ) -> "CNN1D":
        rng = np.random.default_rng(seed)
        c1, c2 = channels
        # He-ish scale for ReLU.
        w1 = rng.normal(0, np.sqrt(2 / (1 * kernel)), size=(c1, 1, kernel))
        b1 = np.zeros(c1)
        w2 = rng.normal(0, np.sqrt(2 / (c1 * kernel)), size=(c2, c1, kernel))
        b2 = np.zeros(c2)
        w_fc = rng.normal(0, np.sqrt(2 / c2), size=(n_classes, c2))
        b_fc = np.zeros(n_classes)
        return cls(w1, b1, w2, b2, w_fc, b_fc, n_classes)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """X: (B, L) → probs (B, C). Also returns cache for backward."""
        x = X[:, None, :].astype(np.float64)  # (B, 1, L)
        z1 = _conv1d(x, self.w1, self.b1)
        a1 = _relu(z1)
        z2 = _conv1d(a1, self.w2, self.b2)
        a2 = _relu(z2)
        # Global average pool over time.
        gap = a2.mean(axis=2)  # (B, C2)
        logits = gap @ self.w_fc.T + self.b_fc
        probs = _softmax(logits)
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "gap": gap, "probs": probs}
        return probs, cache

    def predict_proba(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        outs = []
        for start in range(0, X.shape[0], batch_size):
            probs, _ = self.forward(X[start : start + batch_size])
            outs.append(probs)
        return np.concatenate(outs, axis=0)

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        return self.predict_proba(X, batch_size=batch_size).argmax(axis=1)


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())


def train_cnn1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_classes: int,
    channels: tuple[int, int] = (8, 16),
    epochs: int = 25,
    lr: float = 0.02,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[CNN1D, list[dict]]:
    """SGD with a hand-rolled backward pass. Slow but dependency-free."""
    model = CNN1D.init(n_classes=n_classes, channels=channels, seed=seed)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    n = X_train.shape[0]
    k = model.w1.shape[2]

    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            probs, cache = model.forward(xb)
            losses.append(_cross_entropy(probs, yb))
            # Softmax + CE gradient.
            bsz = yb.shape[0]
            dlogits = probs.copy()
            dlogits[np.arange(bsz), yb] -= 1.0
            dlogits /= bsz

            gap = cache["gap"]
            dw_fc = dlogits.T @ gap
            db_fc = dlogits.sum(axis=0)
            dgap = dlogits @ model.w_fc  # (B, C2)

            a2 = cache["a2"]
            tlen = a2.shape[2]
            da2 = np.repeat(dgap[:, :, None], tlen, axis=2) / tlen
            dz2 = da2 * (cache["z2"] > 0)

            # Conv2 grads.
            a1 = cache["a1"]
            c2, c1, _ = model.w2.shape
            out_len2 = dz2.shape[2]
            dw2 = np.zeros_like(model.w2)
            for oi in range(out_len2):
                window = a1[:, :, oi : oi + k]  # (B, C1, K)
                # dw2[co, ci, kk] += sum_b dz2[b, co, oi] * window[b, ci, kk]
                dw2 += np.tensordot(dz2[:, :, oi], window, axes=([0], [0]))
            db2 = dz2.sum(axis=(0, 2))

            # da1 via full-ish conv transpose (valid forward → full-ish back).
            da1 = np.zeros_like(a1)
            for oi in range(out_len2):
                # dz2[b, co, oi] * w2[co, ci, :] accumulates into a1[b, ci, oi:oi+k]
                # tensordot over co: (B, C1, K)
                contrib = np.tensordot(dz2[:, :, oi], model.w2, axes=([1], [0]))
                da1[:, :, oi : oi + k] += contrib

            dz1 = da1 * (cache["z1"] > 0)
            x = cache["x"]
            out_len1 = dz1.shape[2]
            dw1 = np.zeros_like(model.w1)
            for oi in range(out_len1):
                window = x[:, :, oi : oi + k]
                dw1 += np.tensordot(dz1[:, :, oi], window, axes=([0], [0]))
            db1 = dz1.sum(axis=(0, 2))

            model.w_fc -= lr * dw_fc
            model.b_fc -= lr * db_fc
            model.w2 -= lr * dw2
            model.b2 -= lr * db2
            model.w1 -= lr * dw1
            model.b1 -= lr * db1

        val_probs = model.predict_proba(X_val)
        val_loss = _cross_entropy(val_probs, y_val)
        val_acc = float((val_probs.argmax(1) == y_val).mean())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
    return model, history
