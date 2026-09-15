# ADR 0001: Synthetic-first CI, optional PhysioNet

**Status:** Accepted (v0)
**Date:** 2026-09-13

## Context

ECG work usually starts from a public dataset. PhysioNet sets (MIT-BIH, PTB-XL) require a download, a license click-through and extra Python dependencies (`wfdb`). CI that hits the network is fragile and breaks a clone-and-test workflow.

## Decision

- Default experiment source is the in-repo **synthetic generator**.
- CI runs only `configs/ci.yaml` (synthetic, small n).
- PhysioNet is an optional adapter sharing the `RecordBatch` schema. Committed metrics come from the generator.

## Consequences

Accuracy on the simulator can be high when the cues are easy, so the grouped split, the leakage tests and the dummy baseline carry the signal.
