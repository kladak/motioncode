# MotionCode

**Scientific ML / ECG-like classification — reproducible pipelines, baselines, honest metrics.**

Portfolio project for [Karim Ladak](https://github.com/kladak) (Applied AI + scientific software). Spec: [`SPEC.md`](SPEC.md).

## Honesty (read first)

- **Educational research tooling.** Not a medical device. Not clinical decision support. Not FDA-facing software.
- Labels in the default path are **synthetic morphology tags** (`regular` / `irregular` / `wide` / `burst`). They are **not diagnoses**.
- There is **no prior public MotionCode repo** under `kladak/motioncode`. A circulating **45% → 69.3%** accuracy story is **not reconstructible** from any public commit, paper, or notebook tied to this project. This codebase does **not** invent or backfill that claim.
- Default CI uses **synthetic data only** (no network). Optional PhysioNet download is documented and **not** executed in CI. No PhysioNet-derived scores are shipped in v0.

## Quick start (offline)

```bash
python -m pip install -e ".[dev]"
make test
make train-ci          # configs/ci.yaml → reports/metrics.json
python -m motioncode.cli infer --model logistic --class-name irregular
# or:
make demo
```

Local (larger synthetic draw):

```bash
make train             # configs/synthetic.yaml
```

## What you get

```text
preprocess → grouped train/val/test (record_id) + leakage checks
  → dummy / logistic / forest (handcrafted features)
  → tiny numpy 1D CNN (raw window)
  → accuracy, macro-F1, AUROC (ovr), confusion plots
  → reports/metrics.json
```

| Piece | Path |
|-------|------|
| Spec | [`SPEC.md`](SPEC.md) |
| Experiment YAMLs | [`configs/`](configs/) |
| Generator + schema | [`motioncode/data/`](motioncode/data/) |
| Leakage-aware split | [`motioncode/data/splits.py`](motioncode/data/splits.py) |
| Features / baselines / CNN | [`motioncode/features.py`](motioncode/features.py), [`motioncode/models/`](motioncode/models/) |
| Train + metrics | [`motioncode/pipeline.py`](motioncode/pipeline.py) |
| CLI | `python -m motioncode.cli {train,infer,demo}` |
| Optional PhysioNet helper | [`scripts/download_physionet.py`](scripts/download_physionet.py) (not used by CI) |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

## Synthetic CI metrics (real run, seed=42)

Produced by `make train-ci` on this machine (2026-09-13). **n_per_class=24**, length=256, split 66/15/15, leakage_ok=true. These are **simulator** scores, not clinical performance.

| Model | Accuracy | Macro-F1 | AUROC (ovr) |
|-------|----------|----------|-------------|
| dummy (most_frequent) | 0.267 | 0.105 | 0.500 |
| logistic (features) | 0.733 | 0.698 | 0.869 |
| random forest (features) | 0.733 | 0.697 | 0.907 |
| cnn1d (25 epochs, numpy) | 0.467 | 0.357 | 0.723 |

Takeaways that are allowed on a resume:

- Feature baselines beat a majority dummy on a leakage-checked split.
- A tiny teaching CNN beats dummy after more epochs but still trails handcrafted features on this short synthetic draw — expected, and reported as such.
- Absolute numbers will move with seed / `n_per_class`; regenerate with `make train-ci` and trust `reports/metrics.json`.

Confusion matrices are written to `reports/confusion_*.png` (gitignored; regenerate locally).

## Dataset paths

1. **Synthetic (default / CI)** — in-repo generator, no network.
2. **PhysioNet (optional)** — `python scripts/download_physionet.py` after `pip install '.[physionet]'`. Label-map adapter into the four morphology tags is **intentionally not implemented** in v0 so we do not pretend MIT-BIH symbols equal simulator classes.

## Tests

```bash
make test
```

Covers split leakage (record id + waveform hash), tensor shapes, deterministic seed for synthetic + sklearn baselines, CNN forward shapes, and an end-to-end CI-config smoke (baselines).

## License

MIT — see [`LICENSE`](LICENSE).
