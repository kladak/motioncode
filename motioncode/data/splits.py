"""Grouped, stratified splits plus leakage checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split

from motioncode.data.schema import RecordBatch


@dataclass
class LeakageIssue:
    kind: str
    detail: str


@dataclass
class LeakageReport:
    ok: bool
    issues: list[LeakageIssue] = field(default_factory=list)

    def raise_if_leaked(self) -> None:
        if not self.ok:
            msg = "; ".join(f"{i.kind}: {i.detail}" for i in self.issues)
            raise LeakageError(msg)


class LeakageError(RuntimeError):
    pass


@dataclass
class SplitBundle:
    train: RecordBatch
    val: RecordBatch
    test: RecordBatch
    leakage: LeakageReport


def _waveform_hash(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def check_leakage(train: RecordBatch, val: RecordBatch, test: RecordBatch) -> LeakageReport:
    issues: list[LeakageIssue] = []
    groups = {
        "train": set(train.record_id.tolist()),
        "val": set(val.record_id.tolist()),
        "test": set(test.record_id.tolist()),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            sample = sorted(overlap)[:5]
            issues.append(
                LeakageIssue(
                    "record_id_overlap",
                    f"{a}/{b} share {len(overlap)} ids e.g. {sample}",
                )
            )

    hashes = {
        name: {_waveform_hash(row) for row in batch.X}
        for name, batch in (("train", train), ("val", val), ("test", test))
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes[a] & hashes[b]
        if overlap:
            issues.append(
                LeakageIssue(
                    "waveform_hash_overlap",
                    f"{a}/{b} share {len(overlap)} identical windows",
                )
            )
    return LeakageReport(ok=len(issues) == 0, issues=issues)


def grouped_split(
    batch: RecordBatch,
    *,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitBundle:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    idx = np.arange(batch.n)
    y = batch.y
    # First carve out test, then split remainder into train/val.
    rest_idx, test_idx = train_test_split(
        idx,
        test_size=test_frac,
        random_state=seed,
        stratify=y,
    )
    val_rel = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_rel,
        random_state=seed,
        stratify=y[rest_idx],
    )
    train = batch.take(np.sort(train_idx))
    val = batch.take(np.sort(val_idx))
    test = batch.take(np.sort(test_idx))
    leakage = check_leakage(train, val, test)
    leakage.raise_if_leaked()
    return SplitBundle(train=train, val=val, test=test, leakage=leakage)
