# MotionCode — SPEC

**Owner:** Karim Ladak (`kladak`)
**Repo:** https://github.com/kladak/motioncode
**Status:** Greenfield v0 (created 2026-09-13)
**Core question:** *Can we run an ECG / time-series classification experiment that is reproducible, leakage-aware, and honest about what the numbers mean?*

> **Honesty (read this first).** MotionCode is **educational research tooling**, not a medical device and not a clinical decision-support product. Nothing here is clinically validated. Synthetic morphology labels are **not diagnoses**. Do not use this software for patient care, triage, or any diagnostic purpose.
>
> There is **no prior public MotionCode repository** under `kladak/motioncode`. A circulating historical claim of a **45% → 69.3%** accuracy lift is **not reconstructible** from any public commit, paper, dataset card, or lab notebook associated with this project. This repo does **not** repeat, backfill, or invent that number. Metrics in `reports/` come only from runs of the code in this tree.

## 1. Problem

ECG classification papers and take-home projects often fail in the same ways: random row splits that leak the same patient into train and test, undocumented preprocessing, a single accuracy number with no baseline, and charts that cannot be regenerated. Interviewers (and anyone who has shipped scientific software) ask for the opposite: a pipeline you can re-run, a split you can defend, a dummy/linear baseline, and metrics that match the files on disk.

## 2. Goals (v0)

1. Generate a **public, offline-first dataset path**: a synthetic ECG-like generator that CI can run with no network.
2. Document an **optional PhysioNet download** that maps into the same record schema — not executed in CI, not used to invent published scores.
3. Preprocess → **grouped train/val/test split** with automated leakage checks.
4. Train **baselines first** (majority dummy, logistic, shallow forest) and a **tiny 1D CNN** on the raw window.
5. Report accuracy, macro-F1, and AUROC (ovr) plus a confusion matrix and a short error analysis.
6. Drive the run from a YAML experiment config; write metrics JSON that CI can produce without secrets.

## 3. Non-goals (v0)

- Claiming clinical performance on MIT-BIH, PTB-XL, or any hospital dataset.
- Reconstructing or citing a 45% → 69.3% historical MotionCode result.
- Shipping a diagnostic app, FHIR integration, or FDA-facing quality system.
- Large deep models, GPU training, or architecture search.
- A Streamlit / SaaS viewer. Plots are matplotlib files under `reports/`.

## 4. Dataset contract

Every record is a `RecordBatch`:

| Field | Meaning |
|-------|---------|
| `X` | `float32` array `(n, length)` — one window per record in v0 |
| `y` | integer class ids |
| `record_id` | stable string; **split unit** |
| `class_name` | morphology label (see below) |
| `fs` | sampling rate in Hz |
| `source` | `synthetic` or `physionet` |

### 4.1 Synthetic classes (v0)

These are **morphology tags for a simulator**, not rhythm diagnoses:

| Name | What the generator does |
|------|-------------------------|
| `regular` | Quasi-periodic QRS-like spikes, stable RR |
| `irregular` | Same spike shape, high RR variance |
| `wide` | Broader spike, stable RR |
| `burst` | Regular background plus occasional premature-like extras |

A model that scores well here has learned **simulator cues**. That is useful for testing the pipeline. It is not evidence of arrhythmia detection.

### 4.2 Optional PhysioNet path

`scripts/download_physionet.py` documents how to pull a public PhysioNet set (MIT-BIH arrhythmia as the default pointer) into the same schema. CI **must not** download it. This repo ships **no PhysioNet-derived metrics**.

## 5. Split and leakage

- Split is **stratified by class** and **grouped by `record_id`**.
- Default fractions: 70 / 15 / 15 train / val / test.
- Leakage checker fails the run if:
  - any `record_id` appears in more than one split
  - any exact waveform hash appears in more than one split
- Windowing the same record (later work) must keep all windows of a record in one split. v0 uses one window per record so this is automatic, but the API still groups on `record_id`.

## 6. Models

| Id | Input | Role |
|----|-------|------|
| `dummy` | class prior | Floor. If you cannot beat this, stop. |
| `logistic` | handcrafted features | Linear baseline |
| `forest` | same features | Shallow nonlinear baseline |
| `cnn1d` | z-scored raw window | Tiny two-layer 1D CNN (numpy) |

Feature list is versioned in code (`motioncode/features.py`). The CNN is a teaching model: small filters, global average pool, softmax. It is not a ResNet and is not compared to literature CNNs.

## 7. Metrics

On the **test** split, for every model:

- accuracy
- macro-F1
- AUROC (one-vs-rest, skipped only if a class is missing in y_true — should not happen with stratified generation)
- confusion matrix
- per-class precision / recall

Error analysis lists the most common off-diagonal pairs and a few example `record_id`s. No cherry-picked “best seed” hunting in v0: one seed in the YAML, one reported run.

## 8. Reproducibility

- Single `seed` in the experiment YAML feeds numpy, sklearn `random_state`, and the CNN RNG.
- CI uses `configs/ci.yaml` (small n, few CNN epochs) so the job stays offline and fast.
- Local default is `configs/synthetic.yaml`.
- `reports/metrics.json` is the machine-readable sink. Plots are extras.

## 9. Success criteria for this slice

- `make test` passes offline.
- `make train-ci` writes `reports/metrics.json` with the required keys.
- README / SPEC match the code: no device claim, no fabricated 45→69.3 history, no PhysioNet scores unless a real run is committed (none in v0).
- An independent reviewer can challenge the split and point at the leakage tests.

## 10. Interview talking points (must be true in code)

- “Here is the grouped split and the test that would fail if a record leaked.”
- “Here is dummy vs logistic vs forest vs tiny CNN on the same synthetic draw.”
- “These labels are simulator tags; I would not put this number on a resume as clinical AUROC.”
- “There is no reconstructible prior MotionCode benchmark. I started from a public generator and said so.”
