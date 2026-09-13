#!/usr/bin/env python3
"""Optional PhysioNet download helper (NOT used by CI).

This script documents how to pull MIT-BIH arrhythmia records into a local
cache. It does not run in GitHub Actions. MotionCode v0 ships **no**
PhysioNet-derived metrics — do not invent them after downloading.

Usage (local, needs network + `pip install 'motioncode[physionet]'`):

    python scripts/download_physionet.py --out data/physionet --records 100 101

The adapter that maps WFDB annotations into MotionCode's RecordBatch schema
is intentionally stubbed: mapping MIT-BIH beat symbols onto the four
synthetic morphology tags would be misleading. Bring your own label map
when you extend this path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/physionet"))
    parser.add_argument(
        "--records",
        nargs="+",
        default=["100", "101", "102"],
        help="MIT-BIH record ids to download",
    )
    parser.add_argument(
        "--db",
        default="mitdb",
        help="PhysioNet database name (default: mitdb)",
    )
    args = parser.parse_args()

    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "wfdb is not installed. Run: pip install '.[physionet]'\n"
            "CI does not install this extra and must not call this script."
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading into",
        args.out,
        "— educational use only. Not a clinical dataset card.",
    )
    for rec in args.records:
        print(f"  {args.db}/{rec} ...")
        wfdb.dl_database(args.db, dl_dir=str(args.out), records=[rec])
    readme = args.out / "README_MOTIONCODE.txt"
    readme.write_text(
        "PhysioNet cache for MotionCode.\n"
        "Do not commit derived accuracy claims without a real evaluated run.\n"
        "The historical 45%→69.3% MotionCode number is unavailable / not reconstructible.\n"
    )
    print("done. Label-map adapter is not implemented in v0 (see SPEC.md).")


if __name__ == "__main__":
    main()
