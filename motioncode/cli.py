"""CLI entrypoints: train, infer, and a tiny demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from motioncode.config import load_config
from motioncode.data.schema import CLASSES, ID_TO_CLASS
from motioncode.data.synthetic import SyntheticConfig, generate_synthetic
from motioncode.models.cnn1d import CNN1D
from motioncode.pipeline import run_experiment
from motioncode.preprocess import preprocess_batch


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.reports_dir:
        cfg.reports_dir = args.reports_dir
    if args.artifacts_dir:
        cfg.artifacts_dir = args.artifacts_dir
    results = run_experiment(cfg)
    print(json.dumps({k: results["models"][k].get("accuracy") for k in results["models"]}, indent=2))
    print(f"wrote {results['metrics_path']}")
    return 0


def _load_cnn(path: Path) -> CNN1D:
    data = np.load(path)
    return CNN1D(
        w1=data["w1"],
        b1=data["b1"],
        w2=data["w2"],
        b2=data["b2"],
        w_fc=data["w_fc"],
        b_fc=data["b_fc"],
        n_classes=int(data["n_classes"]),
    )


def cmd_infer(args: argparse.Namespace) -> int:
    artifacts = Path(args.artifacts_dir)
    model_name = args.model
    # Build a one-off synthetic sample when no input npz is given (demo path).
    if args.input:
        arr = np.load(args.input)
        X = arr["X"] if "X" in arr.files else arr[arr.files[0]]
        fs = float(arr["fs"]) if "fs" in arr.files else 128.0
    else:
        batch = generate_synthetic(
            SyntheticConfig(n_per_class=1, length=args.length, fs=128.0, seed=args.seed)
        )
        # Pick the requested class if provided.
        if args.class_name:
            mask = batch.class_name == args.class_name
            if not mask.any():
                print(f"no synthetic sample for class {args.class_name}", file=sys.stderr)
                return 2
            X = batch.X[mask][:1]
        else:
            X = batch.X[:1]
        fs = batch.fs
        print(f"demo input class≈{batch.class_name[0] if not args.class_name else args.class_name}")

    # Minimal preprocess: z-score only (matches train when bandpass already applied
    # at train time on the full pipeline; for demo we re-run the same helper).
    from motioncode.data.schema import RecordBatch

    tmp = RecordBatch(
        X=X.astype(np.float32),
        y=np.zeros(X.shape[0], dtype=np.int64),
        record_id=np.array([f"infer-{i}" for i in range(X.shape[0])], dtype="U32"),
        class_name=np.array(["unknown"] * X.shape[0], dtype="U16"),
        fs=fs,
        source="infer",
    )
    clean = preprocess_batch(tmp)

    if model_name == "cnn1d":
        model = _load_cnn(artifacts / "cnn1d.npz")
        proba = model.predict_proba(clean.X)[0]
        pred = int(proba.argmax())
    else:
        bundle = joblib.load(artifacts / f"{model_name}.joblib")
        pred = int(bundle.predict(clean.X)[0])
        proba = bundle.predict_proba(clean.X)[0]

    payload = {
        "model": model_name,
        "pred_class": ID_TO_CLASS[pred],
        "pred_id": pred,
        "proba": {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))},
        "data_note": "Synthetic sample; labels are morphology tags, not diagnoses.",
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Train the CI config then run one inference: the smoke path from the README."""
    args.config = args.config or "configs/ci.yaml"
    rc = cmd_train(args)
    if rc != 0:
        return rc
    args.model = args.model or "logistic"
    args.input = None
    args.class_name = args.class_name or "irregular"
    args.length = 256
    args.seed = 7
    return cmd_infer(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motioncode", description="MotionCode ECG-like ML toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Run preprocess → split → train → metrics")
    t.add_argument("--config", default="configs/synthetic.yaml")
    t.add_argument("--reports-dir", default=None)
    t.add_argument("--artifacts-dir", default=None)
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer", help="Score one window with a saved artifact")
    i.add_argument("--model", default="logistic", choices=["dummy", "logistic", "forest", "cnn1d"])
    i.add_argument("--artifacts-dir", default="artifacts")
    i.add_argument("--input", default=None, help="Optional .npz with X (and fs)")
    i.add_argument("--class-name", default=None, help="Synthetic demo class if no --input")
    i.add_argument("--length", type=int, default=512)
    i.add_argument("--seed", type=int, default=0)
    i.set_defaults(func=cmd_infer)

    d = sub.add_parser("demo", help="Train CI config and run one synthetic inference")
    d.add_argument("--config", default="configs/ci.yaml")
    d.add_argument("--reports-dir", default="reports")
    d.add_argument("--artifacts-dir", default="artifacts")
    d.add_argument("--model", default="logistic")
    d.add_argument("--class-name", default="irregular")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
