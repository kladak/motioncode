"""Shared record contract for synthetic and (optional) PhysioNet loaders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Simulator morphology tags — not rhythm diagnoses.
CLASSES: tuple[str, ...] = ("regular", "irregular", "wide", "burst")
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_ID.items()}


@dataclass
class RecordBatch:
    """One window per record in v0. `record_id` is the split unit."""

    X: np.ndarray  # (n, length) float32
    y: np.ndarray  # (n,) int64
    record_id: np.ndarray  # (n,) unicode
    class_name: np.ndarray  # (n,) unicode
    fs: float
    source: str

    def __post_init__(self) -> None:
        n = int(self.X.shape[0])
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2d (n, length), got {self.X.shape}")
        if self.y.shape != (n,):
            raise ValueError(f"y length {self.y.shape} != n={n}")
        if self.record_id.shape != (n,) or self.class_name.shape != (n,):
            raise ValueError("record_id / class_name must be length n")
        if len(set(self.record_id.tolist())) != n:
            raise ValueError("record_id values must be unique within a batch")

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def length(self) -> int:
        return int(self.X.shape[1])

    def take(self, idx: np.ndarray) -> "RecordBatch":
        idx = np.asarray(idx)
        return RecordBatch(
            X=self.X[idx],
            y=self.y[idx],
            record_id=self.record_id[idx],
            class_name=self.class_name[idx],
            fs=self.fs,
            source=self.source,
        )
