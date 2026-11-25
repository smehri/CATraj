"""Cut the committed multi-day sample from the raw Zenodo archive.

Reads ``data/zenodo_1167595/`` (``scripts/download_data.py``) and writes
``data/sample/raw/``, which mirrors the archive's folders, file names and CSV
formats, so ``catraj.preprocessing.preprocess_archive`` reads either root:

====================================  ===========================================
``P1_AIS_Data/nari_dynamic.csv``      the rows inside the window
``P1_AIS_Data/nari_static.csv``       the static rows of those vessels: the rows
                                      in the window and, for a vessel with none,
                                      its last static row before it
``P1_AIS_Codes/``                     the code tables, whole
``E2_Weather_Conditions/``            observations inside the window; stations
                                      and wind directions whole
``E1_ocean/oc_november.csv``          the window plus one 3-hour step on each
                                      side, in the AIS bounding box + a margin
``C2_European_Coastline/``            the coastline clipped to the same box
====================================  ===========================================

Every folder's ``README_parameters.txt`` is copied. The default window is
2015-11-09 00:00 to 2015-11-11 00:00 UTC (the start inclusive, the end
exclusive): 565 vessels and complete hourly weather at the 16 stations.

``oc_november.csv`` is extracted from ``E1_Ocean_Conditions.zip`` into
``data/zenodo_1167595/E1_ocean/`` if it is not there (the zip is kept).

The sample's bounding box is that of the window's AIS positions that lie in the
archive's stated area, plus ``--margin`` degrees. The coastline is clipped to
it in the coastline's own CRS (EPSG:3035) and written with the archive's name.

Usage, from the repository root, in the ``catraj`` environment::

    python scripts/make_sample.py
    python scripts/make_sample.py --start 2015-12-01 --end 2015-12-03 --out DIR
    python scripts/make_sample.py --margin 0.1 --ocean-month november

Prints the size of every file written and the total.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")   # see catraj/__init__.py

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_1167595"
OUT = ROOT / "data" / "sample" / "raw"
CHUNK = 2_000_000                 # CSV rows read at once
OCEAN_STEP = 3 * 3600             # the sea-state forecast's step, in seconds
AREA = (-10.0, 0.0, 45.0, 51.0)   # lon_min, lon_max, lat_min, lat_max: the archive's
COASTLINE = "Europe Coastline (Polygone)"
COASTLINE_CRS = 3035
BUDGET_TOTAL, BUDGET_FILE = 80e6, 45e6      # bytes


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _epoch(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def _chunks(path: Path):
    """The CSV in chunks, every field kept as the text the archive holds."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=CHUNK)


def _write_csv(df: pd.DataFrame, path: Path, first: bool) -> None:
    df.to_csv(path, mode="w" if first else "a", header=first, index=False,
              lineterminator="\n")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ---- AIS --------------------------------------------------------------------------

def cut_ais(data: Path, out: Path, t0: int, t1: int):
    """Window of ``nari_dynamic.csv`` and the matching static rows; the box."""
    folder = out / "P1_AIS_Data"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "nari_dynamic.csv"
    first, mmsis, n_rows = True, set(), 0
    lon_lo, lon_hi, lat_lo, lat_hi = 1e9, -1e9, 1e9, -1e9
    for chunk in _chunks(data / "P1_AIS_Data" / "nari_dynamic.csv"):
        t = chunk["t"].astype("int64")
        keep = chunk[(t >= t0) & (t < t1)]
        if keep.empty:
            continue
        _write_csv(keep, target, first)
        first = False
        n_rows += len(keep)
        mmsis.update(keep["sourcemmsi"].astype("int64"))
        lon, lat = keep["lon"].astype(float), keep["lat"].astype(float)
        ok = lon.between(AREA[0], AREA[1]) & lat.between(AREA[2], AREA[3])
        if ok.any():
            lon_lo, lon_hi = min(lon_lo, lon[ok].min()), max(lon_hi, lon[ok].max())
            lat_lo, lat_hi = min(lat_lo, lat[ok].min()), max(lat_hi, lat[ok].max())
    if first:
        raise SystemExit("no AIS rows in the window")

    static = pd.read_csv(data / "P1_AIS_Data" / "nari_static.csv", dtype=str,
                         keep_default_na=False)
    ts = static["t"].astype("int64")
    ids = static["sourcemmsi"].astype("int64")
    inside = ids.isin(mmsis) & (ts >= t0) & (ts < t1)
    before = static[ids.isin(mmsis - set(ids[inside])) & (ts < t0)]
    before = before.assign(_t=ts[before.index]).sort_values("_t", kind="stable")
    before = before.drop_duplicates("sourcemmsi", keep="last").drop(columns="_t")
    rows = pd.concat([static[inside], before]).sort_index()
    _write_csv(rows, folder / "nari_static.csv", True)
    _copy(data / "P1_AIS_Data" / "README_parameters.txt",
          folder / "README_parameters.txt")

    codes = data / "P1_AIS_Codes"
    for f in sorted(codes.glob("*")):
        _copy(f, out / "P1_AIS_Codes" / f.name)
    print(f"AIS: {n_rows:,} rows, {len(mmsis):,} vessels; static {len(rows):,} rows "
          f"({len(before):,} from before the window)")
    return (lon_lo, lon_hi, lat_lo, lat_hi)


# ---- weather ----------------------------------------------------------------------

def cut_weather(data: Path, out: Path, t0: int, t1: int) -> None:
    src, dst = data / "E2_Weather_Conditions", out / "E2_Weather_Conditions"
    dst.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(src / "table_wheatherObservation.csv", dtype=str,
                      keep_default_na=False)
    t = obs["local_time"].astype("int64")
    obs = obs[(t >= t0) & (t < t1)]
    _write_csv(obs, dst / "table_wheatherObservation.csv", True)
    for name in ("table_weatherStation.csv", "table_windDirection.csv",
                 "README_parameters.txt", "README_map_weatherStations.png"):
        if (src / name).exists():
            _copy(src / name, dst / name)
    per = obs.groupby("id_station")["local_time"].size()
    print(f"weather: {len(obs):,} observations at {len(per)} stations "
          f"({per.min()}-{per.max()} each)")


# ---- ocean ------------------------------------------------------------------------

def _ensure_month(data: Path, month: str) -> Path:
    csv = data / "E1_ocean" / f"oc_{month}.csv"
    if csv.exists():
        return csv
    zpath = data / "E1_Ocean_Conditions.zip"
    if not zpath.exists():
        raise SystemExit(f"{_rel(csv)} and {_rel(zpath)} are both absent")
    print(f"extracting oc_{month}.csv from {_rel(zpath)} ...")
    (data / "E1_ocean").mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extract(f"oc_{month}.csv", data / "E1_ocean")
    return csv


def cut_ocean(data: Path, out: Path, t0: int, t1: int, box, month: str) -> None:
    src = _ensure_month(data, month)
    dst = out / "E1_ocean"
    dst.mkdir(parents=True, exist_ok=True)
    lo, hi = t0 - OCEAN_STEP, t1 + OCEAN_STEP
    lon_lo, lon_hi, lat_lo, lat_hi = box
    first, n_rows = True, 0
    for chunk in _chunks(src):
        t = chunk["ts"].astype("int64")
        lon, lat = chunk["lon"].astype(float), chunk["lat"].astype(float)
        keep = chunk[(t >= lo) & (t <= hi) & lon.between(lon_lo, lon_hi)
                     & lat.between(lat_lo, lat_hi)]
        if not keep.empty:
            _write_csv(keep, dst / f"oc_{month}.csv", first)
            first = False
            n_rows += len(keep)
    if first:
        raise SystemExit("no ocean rows in the window and box")
    _copy(data / "E1_ocean" / "README_parameters.txt", dst / "README_parameters.txt")
    oc = pd.read_csv(dst / f"oc_{month}.csv", usecols=["lon", "lat", "ts"])
    print(f"ocean: {n_rows:,} rows, {len(oc[['lon', 'lat']].drop_duplicates()):,} "
          f"cells, {oc.ts.nunique()} time steps")


# ---- coastline --------------------------------------------------------------------

def cut_coastline(data: Path, out: Path, box) -> None:
    import geopandas as gpd
    from shapely.geometry import box as bbox

    src, dst = data / "C2_European_Coastline", out / "C2_European_Coastline"
    dst.mkdir(parents=True, exist_ok=True)
    clip = gpd.GeoSeries([bbox(box[0], box[2], box[1], box[3]).segmentize(0.01)],
                         crs=4326).to_crs(COASTLINE_CRS).iloc[0]
    coast = gpd.read_file(src / f"{COASTLINE}.shp", bbox=tuple(clip.bounds))
    coast = coast[coast.intersects(clip)].copy()
    coast["geometry"] = coast.geometry.intersection(clip)
    coast = coast[~coast.geometry.is_empty]
    coast.to_file(dst / f"{COASTLINE}.shp", encoding="utf-8")
    prj = src / f"{COASTLINE}.prj"
    if prj.exists():                      # keep the archive's own .prj text
        _copy(prj, dst / prj.name)
    _copy(src / "README_parameters.txt", dst / "README_parameters.txt")
    print(f"coastline: {len(coast):,} polygons clipped to the box")


# ---- driver -----------------------------------------------------------------------

def report_sizes(out: Path) -> bool:
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print("\nfile sizes")
    for p in files:
        print(f"  {p.stat().st_size / 1e6:9.3f} MB  {p.relative_to(out).as_posix()}")
    big = max((p.stat().st_size for p in files), default=0)
    print(f"  {total / 1e6:9.3f} MB  total of {_rel(out)} "
          f"(budget for data/sample/: {BUDGET_TOTAL / 1e6:.0f} MB, "
          f"one file {BUDGET_FILE / 1e6:.0f} MB)")
    return total <= BUDGET_TOTAL and big <= BUDGET_FILE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cut the multi-day sample from the raw Zenodo archive.",
        epilog="Needs scripts/download_data.py to have run. See the module docstring.")
    parser.add_argument("--data", type=Path, default=DATA,
                        help="the raw archive (default: data/zenodo_1167595)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="where to write (default: data/sample/raw)")
    parser.add_argument("--start", default="2015-11-09 00:00",
                        help="window start, UTC, inclusive (default: 2015-11-09 00:00)")
    parser.add_argument("--end", default="2015-11-11 00:00",
                        help="window end, UTC, exclusive (default: 2015-11-11 00:00)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="degrees added around the AIS bounding box (default: 0.2)")
    parser.add_argument("--ocean-month", default="november",
                        help="the oc_<month>.csv with the window (default: november)")
    args = parser.parse_args(argv)
    t0, t1 = _epoch(args.start), _epoch(args.end)
    if not t0 < t1:
        parser.error("--start must be before --end")
    if args.margin < 0:
        parser.error("--margin must not be negative")

    print(f"window {args.start} to {args.end} UTC -> {_rel(args.out)}/")
    box = cut_ais(args.data, args.out, t0, t1)
    box = (max(AREA[0], box[0] - args.margin), min(AREA[1], box[1] + args.margin),
           max(AREA[2], box[2] - args.margin), min(AREA[3], box[3] + args.margin))
    print(f"box (lon_min, lon_max, lat_min, lat_max): "
          f"{', '.join(f'{v:.4f}' for v in box)}")
    cut_weather(args.data, args.out, t0, t1)
    cut_ocean(args.data, args.out, t0, t1, box, args.ocean_month)
    cut_coastline(args.data, args.out, box)
    if not report_sizes(args.out):
        print("\nOVER BUDGET: see the sizes above; decide what to cut")
    return 0


if __name__ == "__main__":
    sys.exit(main())
