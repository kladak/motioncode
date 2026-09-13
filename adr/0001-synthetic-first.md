# ADR 0001 — Synthetic-first CI, optional PhysioNet

**Status:** Accepted (v0)
**Date:** 2026-09-13

## Context

ECG work wants a public dataset. PhysioNet sets (MIT-BIH, PTB-XL) are the honest public path, but they require a download, a license click-through, and extra Python deps (`wfdb`). CI that hits the network is fragile and makes a “clone and test” story fail.

There is also no prior MotionCode repo to replay. Inventing a historical 45% → 69.3% number to look like a continuation would be worse than starting clean.

## Decision

- Default experiment source is the in-repo **synthetic generator**.
- CI runs only `configs/ci.yaml` (synthetic, small n).
- PhysioNet is an **optional adapter** with the same `RecordBatch` schema. No PhysioNet metrics are committed until a real local run is checked in (none in v0).
- Documentation states that the old 45→69.3 claim is **unavailable**, not “pending.”

## Consequences

Synthetic accuracy can be high if the simulator cues are easy. That is acceptable if the README says so. Reviewers should grade the split, leakage tests, and baseline discipline — not the absolute synthetic score.
