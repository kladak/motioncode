"""YAML experiment config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    source: str = "synthetic"
    n_per_class: int = 60
    length: int = 512
    fs: float = 128.0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    detrend: bool = True
    bandpass: tuple[float, float] | None = (0.5, 40.0)
    normalize: str = "zscore"
    models: list[str] = field(default_factory=lambda: ["dummy", "logistic", "forest", "cnn1d"])
    cnn_epochs: int = 40
    cnn_lr: float = 0.05
    cnn_batch_size: int = 16
    cnn_channels: tuple[int, int] = (8, 16)
    reports_dir: str = "reports"
    artifacts_dir: str = "artifacts"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        ds = raw.get("dataset", {})
        split = raw.get("split", {})
        pre = raw.get("preprocess", {})
        cnn = raw.get("cnn", {})
        band = pre.get("bandpass", [0.5, 40.0])
        bandpass = None if band in (None, "none", False) else (float(band[0]), float(band[1]))
        ch = cnn.get("channels", [8, 16])
        return cls(
            seed=int(raw.get("seed", 42)),
            source=str(ds.get("source", "synthetic")),
            n_per_class=int(ds.get("n_per_class", 60)),
            length=int(ds.get("length", 512)),
            fs=float(ds.get("fs", 128.0)),
            train_frac=float(split.get("train", 0.70)),
            val_frac=float(split.get("val", 0.15)),
            test_frac=float(split.get("test", 0.15)),
            detrend=bool(pre.get("detrend", True)),
            bandpass=bandpass,
            normalize=str(pre.get("normalize", "zscore")),
            models=list(raw.get("models", ["dummy", "logistic", "forest", "cnn1d"])),
            cnn_epochs=int(cnn.get("epochs", 40)),
            cnn_lr=float(cnn.get("lr", 0.05)),
            cnn_batch_size=int(cnn.get("batch_size", 16)),
            cnn_channels=(int(ch[0]), int(ch[1])),
            reports_dir=str(raw.get("reports_dir", "reports")),
            artifacts_dir=str(raw.get("artifacts_dir", "artifacts")),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)
