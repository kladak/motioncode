"""Leakage and split invariants."""

from __future__ import annotations

import numpy as np
import pytest

from motioncode.data.schema import RecordBatch
from motioncode.data.splits import LeakageError, check_leakage, grouped_split
from motioncode.data.synthetic import SyntheticConfig, generate_synthetic


def _tiny_batch(seed: int = 0) -> RecordBatch:
    return generate_synthetic(SyntheticConfig(n_per_class=20, length=128, seed=seed))


def test_grouped_split_no_record_overlap():
    batch = _tiny_batch()
    splits = grouped_split(batch, seed=7)
    ids = (
        set(splits.train.record_id),
        set(splits.val.record_id),
        set(splits.test.record_id),
    )
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert splits.leakage.ok


def test_split_sizes_sum_to_n():
    batch = _tiny_batch(seed=3)
    splits = grouped_split(batch, seed=3)
    assert splits.train.n + splits.val.n + splits.test.n == batch.n


def test_leakage_checker_catches_shared_id():
    batch = _tiny_batch(seed=1)
    train = batch.take(np.arange(10))
    val = batch.take(np.arange(5, 15))  # overlap on purpose
    test = batch.take(np.arange(15, 25))
    report = check_leakage(train, val, test)
    assert not report.ok
    assert any(i.kind == "record_id_overlap" for i in report.issues)


def test_leakage_checker_catches_duplicate_waveform():
    batch = _tiny_batch(seed=2)
    train = batch.take(np.arange(10))
    # Clone a train waveform into val under a new id.
    X = batch.X[10:20].copy()
    X[0] = train.X[0]
    val = RecordBatch(
        X=X,
        y=batch.y[10:20].copy(),
        record_id=np.array([f"val-clone-{i}" for i in range(10)], dtype="U32"),
        class_name=batch.class_name[10:20].copy(),
        fs=batch.fs,
        source=batch.source,
    )
    test = batch.take(np.arange(20, 30))
    report = check_leakage(train, val, test)
    assert not report.ok
    assert any(i.kind == "waveform_hash_overlap" for i in report.issues)


def test_grouped_split_raises_on_forced_leak(monkeypatch):
    batch = _tiny_batch(seed=4)

    def bad_check(train, val, test):
        from motioncode.data.splits import LeakageIssue, LeakageReport

        return LeakageReport(
            ok=False, issues=[LeakageIssue("record_id_overlap", "forced")]
        )

    monkeypatch.setattr("motioncode.data.splits.check_leakage", bad_check)
    with pytest.raises(LeakageError):
        grouped_split(batch, seed=4)
