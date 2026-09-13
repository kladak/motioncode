"""Honest metric sink: only numbers computed from y_true / y_pred / proba."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from motioncode.data.schema import CLASSES


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    *,
    class_names: tuple[str, ...] = CLASSES,
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == len(class_names):
        # AUROC needs all classes present in y_true for ovr safely; still try.
        present = set(np.unique(y_true).tolist())
        if present == set(labels):
            out["auroc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        else:
            out["auroc_ovr"] = None
            out["auroc_note"] = (
                f"skipped: y_true missing classes {set(labels) - present}"
            )
    else:
        out["auroc_ovr"] = None
    out["error_pairs"] = _top_error_pairs(cm, class_names)
    return out


def _top_error_pairs(
    cm: np.ndarray, class_names: tuple[str, ...], top_k: int = 5
) -> list[dict[str, Any]]:
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append(
                    {
                        "true": class_names[i],
                        "pred": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["pred"]))
    return pairs[:top_k]


def write_confusion_plot(
    cm: list[list[int]] | np.ndarray,
    class_names: tuple[str, ...],
    path: str | Path,
    *,
    title: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, interpolation="nearest", cmap="Greens")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ylabel="true",
        xlabel="pred",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    thresh = mat.max() / 2.0 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                str(mat[i, j]),
                ha="center",
                va="center",
                color="white" if mat[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def example_misclassified(
    record_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = CLASSES,
    limit: int = 8,
) -> list[dict[str, str]]:
    out = []
    for rid, yt, yp in zip(record_ids, y_true, y_pred, strict=True):
        if yt != yp:
            out.append(
                {
                    "record_id": str(rid),
                    "true": class_names[int(yt)],
                    "pred": class_names[int(yp)],
                }
            )
        if len(out) >= limit:
            break
    return out
