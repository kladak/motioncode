"""Sklearn baselines on handcrafted features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from motioncode.features import extract_features


@dataclass
class BaselineBundle:
    name: str
    pipeline: Pipeline
    feature_fs: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = extract_features(X, self.feature_fs)
        return self.pipeline.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = extract_features(X, self.feature_fs)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(feats)
        # Dummy always has predict_proba; keep a guard for future models.
        classes = self.pipeline.classes_
        n = X.shape[0]
        out = np.zeros((n, len(classes)), dtype=np.float64)
        preds = self.pipeline.predict(feats)
        for i, p in enumerate(preds):
            out[i, list(classes).index(p)] = 1.0
        return out


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fs: float,
    seed: int,
    which: list[str] | None = None,
) -> dict[str, BaselineBundle]:
    which = which or ["dummy", "logistic", "forest"]
    feats = extract_features(X_train, fs)
    out: dict[str, BaselineBundle] = {}
    if "dummy" in which:
        pipe = Pipeline(
            [
                ("clf", DummyClassifier(strategy="most_frequent", random_state=seed)),
            ]
        )
        pipe.fit(feats, y_train)
        out["dummy"] = BaselineBundle("dummy", pipe, fs)
    if "logistic" in which:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(feats, y_train)
        out["logistic"] = BaselineBundle("logistic", pipe, fs)
    if "forest" in which:
        pipe = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        pipe.fit(feats, y_train)
        out["forest"] = BaselineBundle("forest", pipe, fs)
    return out
