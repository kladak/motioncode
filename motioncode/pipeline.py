"""End-to-end experiment: load → preprocess → split → train → metrics JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from motioncode.config import ExperimentConfig
from motioncode.data.schema import CLASSES
from motioncode.data.splits import grouped_split
from motioncode.data.synthetic import SyntheticConfig, generate_synthetic
from motioncode.eval.metrics import (
    evaluate_predictions,
    example_misclassified,
    write_confusion_plot,
)
from motioncode.models.baselines import train_baselines
from motioncode.models.cnn1d import train_cnn1d
from motioncode.preprocess import preprocess_batch


def load_batch(cfg: ExperimentConfig):
    if cfg.source == "synthetic":
        return generate_synthetic(
            SyntheticConfig(
                n_per_class=cfg.n_per_class,
                length=cfg.length,
                fs=cfg.fs,
                seed=cfg.seed,
            )
        )
    raise ValueError(
        f"source={cfg.source!r} is not available in the offline path. "
        "Use source=synthetic, or adapt scripts/download_physionet.py locally."
    )


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = load_batch(cfg)
    clean = preprocess_batch(
        raw,
        do_detrend=cfg.detrend,
        bandpass=cfg.bandpass,
        normalize=cfg.normalize,
    )
    splits = grouped_split(
        clean,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )

    reports_dir = Path(cfg.reports_dir)
    artifacts_dir = Path(cfg.artifacts_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "project": "motioncode",
        "version": "0.1.0",
        "disclaimer": (
            "Educational research tooling. Not a medical device. "
            "Synthetic morphology labels are not diagnoses. "
            "No reconstructible historical 45%→69.3% claim exists for this repo."
        ),
        "config": {
            "seed": cfg.seed,
            "source": cfg.source,
            "n_per_class": cfg.n_per_class,
            "length": cfg.length,
            "fs": cfg.fs,
            "models": cfg.models,
            "cnn_epochs": cfg.cnn_epochs,
        },
        "split_sizes": {
            "train": splits.train.n,
            "val": splits.val.n,
            "test": splits.test.n,
        },
        "leakage_ok": splits.leakage.ok,
        "class_names": list(CLASSES),
        "models": {},
    }

    baseline_names = [m for m in cfg.models if m in ("dummy", "logistic", "forest")]
    if baseline_names:
        baselines = train_baselines(
            splits.train.X,
            splits.train.y,
            fs=cfg.fs,
            seed=cfg.seed,
            which=baseline_names,
        )
        for name, bundle in baselines.items():
            y_pred = bundle.predict(splits.test.X)
            y_proba = bundle.predict_proba(splits.test.X)
            metrics = evaluate_predictions(splits.test.y, y_pred, y_proba)
            metrics["misclassified_examples"] = example_misclassified(
                splits.test.record_id, splits.test.y, y_pred
            )
            plot_path = reports_dir / f"confusion_{name}.png"
            write_confusion_plot(
                metrics["confusion_matrix"],
                CLASSES,
                plot_path,
                title=f"{name} — test confusion",
            )
            metrics["confusion_plot"] = str(plot_path)
            results["models"][name] = metrics
            joblib.dump(bundle, artifacts_dir / f"{name}.joblib")

    if "cnn1d" in cfg.models:
        cnn, history = train_cnn1d(
            splits.train.X,
            splits.train.y,
            splits.val.X,
            splits.val.y,
            n_classes=len(CLASSES),
            channels=cfg.cnn_channels,
            epochs=cfg.cnn_epochs,
            lr=cfg.cnn_lr,
            batch_size=cfg.cnn_batch_size,
            seed=cfg.seed,
        )
        y_proba = cnn.predict_proba(splits.test.X)
        y_pred = y_proba.argmax(axis=1)
        metrics = evaluate_predictions(splits.test.y, y_pred, y_proba)
        metrics["misclassified_examples"] = example_misclassified(
            splits.test.record_id, splits.test.y, y_pred
        )
        metrics["train_history_tail"] = history[-3:]
        plot_path = reports_dir / "confusion_cnn1d.png"
        write_confusion_plot(
            metrics["confusion_matrix"],
            CLASSES,
            plot_path,
            title="cnn1d — test confusion",
        )
        metrics["confusion_plot"] = str(plot_path)
        results["models"]["cnn1d"] = metrics
        # Save weight dict (numpy) for the inference CLI.
        np.savez(
            artifacts_dir / "cnn1d.npz",
            w1=cnn.w1,
            b1=cnn.b1,
            w2=cnn.w2,
            b2=cnn.b2,
            w_fc=cnn.w_fc,
            b_fc=cnn.b_fc,
            n_classes=cnn.n_classes,
        )

    results["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    out_path = reports_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    results["metrics_path"] = str(out_path)
    return results
