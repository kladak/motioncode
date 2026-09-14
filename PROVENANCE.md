# Provenance & historical notes

This file holds context that is useful for integrity, but is not needed on the front of the README for recruiters.

## Independent clean-room

There is no prior public MotionCode repository under `kladak/motioncode`. This tree is a clean-room educational implementation: synthetic morphology tags, leakage-checked splits, baselines + a tiny 1-D CNN, and metrics written by the included pipeline.

## Unreconstructible circulating claim

A circulating **45% → 69.3%** accuracy narrative is **not reconstructible** from any public commit, paper, or notebook tied to this project. This codebase does not invent or backfill that claim. Trust only numbers produced by `make train-ci` / `reports/metrics.json` on the synthetic benchmark.

## PhysioNet

Optional PhysioNet download (`scripts/download_physionet.py`) is documented for future adapters. Label-map into the four simulator morphology tags is intentionally not implemented in v0, so MIT-BIH symbols are not treated as equivalent to simulator classes. No PhysioNet-derived scores are shipped.
