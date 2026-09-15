"""End-to-end smoke on the CI-sized config."""

from __future__ import annotations

import json
from pathlib import Path

from motioncode.config import load_config
from motioncode.pipeline import run_experiment


def test_ci_config_produces_metrics(tmp_path: Path):
    cfg = load_config("configs/ci.yaml")
    cfg.reports_dir = str(tmp_path / "reports")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    # Keep the smoke test fast: skip CNN here; dedicated CNN shape test elsewhere.
    cfg.models = ["dummy", "logistic", "forest"]
    results = run_experiment(cfg)
    path = Path(results["metrics_path"])
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["leakage_ok"] is True
    assert "data_note" in payload
    for name in ("dummy", "logistic", "forest"):
        m = payload["models"][name]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert "macro_f1" in m
        assert "confusion_matrix" in m
