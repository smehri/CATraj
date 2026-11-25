"""Prepare the US MarineCadastre AIS data the notebooks read.

Run it once on the daily files. It reads ``data/UsAIS/csv2017/ais-2017-MM-DD.csv.zst``
and the NOAA shoreline in ``data/UsAIS/shoreline/`` and writes Parquet files to
``data/UsAIS/preprocessed/`` (git ignores the folder):

=====================================  ==========================================
``AIS/ais_2017-11.parquet``            every valid AIS position of November in
``AIS/ais_2017-12.parquet``            the east-coast study area, December too
``Coastline/shoreline.parquet``        NOAA shoreline lines in EPSG:32618
``preprocessing_report.csv``/``.txt``  rows in and out of every step
=====================================  ==========================================

The work is done by :func:`catraj.preprocessing.preprocess_us`; this is its command
line. Its module (``catraj/preprocessing/marinecadastre.py``) documents the study
area, every rule, the vessel-type table, the projection (EPSG:32618) with grid
bearings, and the median filter.

Usage, from the repository root, in the ``catraj`` environment::

    python scripts/prepare_us.py --n-jobs -1                 # both months, every core
    python scripts/prepare_us.py --months 2017-11 --force    # one month, rebuild
    python scripts/prepare_us.py --raw data/UsAIS/sample/raw --out data/UsAIS/sample/preprocessed

A month is about 68 million positions (about 4-5 GB in memory); with ``--n-jobs -1``
the days are read, and then the vessels filtered, on every core.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")   # see catraj/__init__.py

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "UsAIS"


class PeakMemory:
    """Peak resident memory of this process and its children, sampled every 2 s."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                procs = [self._proc, *self._proc.children(recursive=True)]
                total = 0
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except Exception:
                        pass
                self.peak = max(self.peak, total)
            except Exception:
                pass
            self._stop.wait(2)

    def __enter__(self):
        if self._proc is not None:
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess the MarineCadastre AIS of Nov-Dec 2017: area, codes, ranges, "
                    "types, grid bearings, noise.")
    parser.add_argument("--raw", type=Path, default=DATA / "csv2017",
                        help="folder of ais-2017-MM-DD.csv.zst (default: data/UsAIS/csv2017)")
    parser.add_argument("--out", type=Path, default=DATA / "preprocessed",
                        help="where to write (default: data/UsAIS/preprocessed)")
    parser.add_argument("--shoreline", type=Path, default=None,
                        help="the shoreline .shp (default: RAW/shoreline or RAW/../shoreline)")
    parser.add_argument("--months", nargs="+", default=["2017-11", "2017-12"],
                        help="months to process (default: 2017-11 2017-12)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="width of the median window, in seconds (default: 30)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="processes: 1 serial (default), -1 every core")
    parser.add_argument("--force", action="store_true", help="rebuild outputs that exist")
    args = parser.parse_args(argv)
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")
    if not args.window > 0:
        parser.error("--window must be positive")

    from catraj.preprocessing import preprocess_us

    print(f"{args.raw} -> {args.out}  (months {' '.join(args.months)}, median window "
          f"{args.window:g} s, n_jobs {args.n_jobs})", flush=True)
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        report = preprocess_us(args.raw, args.out, tuple(args.months), args.n_jobs,
                               args.window, args.force, shoreline=args.shoreline,
                               log=lambda s: print(s, flush=True))
    if len(report):
        print(f"\n{report.to_text()}")
        print(f"\nreport: {args.out / 'preprocessing_report.csv'}")
    else:
        print("nothing to do: check --raw, or pass --force")
    print(f"done in {time.perf_counter() - t0:.1f} s; peak memory (process tree) "
          f"{mem.peak / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
