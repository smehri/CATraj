"""Prepare the Zenodo data the notebooks read: descriptive names, noise removed.

Run it once after ``scripts/download_data.py``. It reads the archive in
``data/zenodo_1167595/`` and writes Parquet files to
``data/zenodo_1167595/preprocessed/`` (git ignores the whole folder):

=====================================  ==========================================
``AIS/ais.parquet``                    every valid AIS position, with vessel type
``Weather/weather.parquet``            hourly observations + station positions
``Ocean/ocean_<month>.parquet``        sea state, one file per ``oc_<month>.csv``
``Coastline/coastline.parquet``        land polygons in EPSG:32630 (GeoParquet)
``preprocessing_report.csv``/``.txt``  rows in and out of every step
=====================================  ==========================================

The work is done by :func:`catraj.preprocessing.preprocess_archive`; this is its
command line. Its module (``catraj/preprocessing/archive.py``) documents every
step and every range rule: names, "not available" codes to NaN, ranges of each
numeric field (MMSI by ITU-R M.585, position, speed, course, heading, rate of
turn, status, time; weather and ocean fields), duplicates and order, vessel
type, projection to UTM zone 30N (EPSG:32630) and the median filter.

The default window is 30 seconds, the package default. It changes AIS
positions, which arrive every few seconds, but leaves hourly weather and
three-hourly sea-state series unchanged.

Usage, from the repository root, in the ``catraj`` environment (it needs
``pyarrow``)::

    python scripts/prepare_zenodo.py                  # everything that was downloaded
    python scripts/prepare_zenodo.py --force          # rebuild files that exist
    python scripts/prepare_zenodo.py --window 60      # another window, in seconds
    python scripts/prepare_zenodo.py --n-jobs -1      # every core (1: serial, the default)
    python scripts/prepare_zenodo.py --data data/sample/raw --out data/sample/preprocessed

An output that already exists is skipped unless ``--force``; a dataset whose raw
files are absent is skipped with a message. The full AIS file (~19 million
rows) is filtered in batches of whole vessels to bound memory.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")   # see catraj/__init__.py

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_1167595"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the Zenodo archive: names, codes, ranges, noise.",
        epilog="Run scripts/download_data.py first. Full guide: docs/getting_the_data.md")
    parser.add_argument("--data", type=Path, default=DATA,
                        help="folder holding the raw archive (default: data/zenodo_1167595)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write (default: DATA/preprocessed)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="width of the median window, in seconds (default: 30)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild outputs that already exist")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="processes: 1 serial (default), -1 every core; weather, "
                             "each sea-state month and the coastline run side by "
                             "side, then the AIS filter per batch of vessels")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")
    if not args.window > 0:
        parser.error("--window must be positive")
    if importlib.util.find_spec("pyarrow") is None:
        sys.exit("pyarrow is needed to write Parquet: pip install pyarrow")

    from catraj.preprocessing import preprocess_archive

    out = args.out if args.out is not None else args.data / "preprocessed"
    print(f"{args.data} -> {out}  (median window {args.window:g} s, "
          f"n_jobs {args.n_jobs})")
    t0 = time.perf_counter()
    report = preprocess_archive(args.data, out, args.window, force=args.force,
                               n_jobs=args.n_jobs)
    if len(report):
        print(f"\n{report.to_text()}")
        print(f"\nreport: {out / 'preprocessing_report.csv'}")
    else:
        print("nothing to do: run scripts/download_data.py first, or pass --force")
    print(f"done in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
