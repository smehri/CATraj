"""Cut the committed US sample from the MarineCadastre daily files and preprocess it.

Two consecutive days of November 2017 (default 2017-11-01 and 2017-11-02), the
positions inside one busy region of the study area, written in the daily files'
own format::

    data/UsAIS/sample/raw/ais-2017-11-DD.csv.zst          the region's rows, all 17 columns
    data/UsAIS/sample/raw/shoreline/us_medium_shoreline.*  NOAA shoreline, clipped to region + margin
    data/UsAIS/sample/preprocessed/                        what preprocess_us makes of them

Regions (``--region``): ``ny_nj`` New York / New Jersey harbour approaches,
``chesapeake`` Chesapeake Bay and Hampton Roads, ``delaware`` Delaware Bay.
``--survey`` prints, for each, the positions, vessels and mix of vessel groups of
the two days, which is how the default was chosen (see the docstring of
:data:`REGIONS`); nothing is written.

Usage, from the repository root, in the ``catraj`` environment::

    python scripts/make_us_sample.py --survey
    python scripts/make_us_sample.py                 # the default region
    python scripts/make_us_sample.py --region chesapeake --force

Needs ``data/UsAIS/csv2017`` and ``data/UsAIS/shoreline``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")   # see catraj/__init__.py

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
US = ROOT / "data" / "UsAIS"

#: name -> (lon_min, lon_max, lat_min, lat_max, description)
REGIONS = {
    "ny_nj": (-74.40, -73.60, 40.20, 40.95, "New York / New Jersey harbour approaches"),
    "chesapeake": (-76.60, -75.75, 36.75, 38.30, "Chesapeake Bay mouth and Hampton Roads"),
    "delaware": (-75.75, -74.60, 38.60, 39.75, "Delaware Bay and river"),
}
DEFAULT_REGION = "ny_nj"
SHORELINE_MARGIN = 0.25             # degrees around the region
DAYS = ("2017-11-01", "2017-11-02")


def read_region(path, box):
    """Rows of one daily file inside ``box``, all 17 columns as written, chunk by chunk."""
    import pandas as pd

    lo, hi, la0, la1 = box[:4]
    parts = []
    for chunk in pd.read_csv(path, compression="zstd", chunksize=2_000_000, dtype=str):
        lon, lat = chunk["longitude"].astype(float), chunk["latitude"].astype(float)
        parts.append(chunk[(lon >= lo) & (lon <= hi) & (lat >= la0) & (lat <= la1)])
    return pd.concat(parts, ignore_index=True)


def survey(days) -> None:
    import pandas as pd

    from catraj.preprocessing import marinecadastre as mc, mmsi_classes

    print(f"{'region':12s} {'rows':>9s} {'vessels':>8s}  vessels per group "
          "(most frequent code per vessel)")
    for name, box in REGIONS.items():
        frames = [read_region(US / "csv2017" / f"ais-{d}.csv.zst", box) for d in days]
        df = pd.concat(frames, ignore_index=True)
        df["code"] = pd.to_numeric(df["vessel_type"], errors="coerce")
        mmsi = df["mmsi"].astype("int64")
        ship = (mmsi_classes(mmsi) == "ship").to_numpy()
        df, mmsi = df[ship], mmsi[ship]
        mode = mc.most_frequent(mmsi.to_numpy(), df["code"].to_numpy())
        groups = (pd.Series([mc.vessel_group_of(mode.get(m)) for m in pd.unique(mmsi)])
                    .value_counts())
        mix = ", ".join(f"{g} {n}" for g, n in groups.items())
        print(f"{name:12s} {len(df):9,d} {mmsi.nunique():8,d}  {mix}")


def write_raw(days, box, out_raw: Path) -> None:
    import pandas as pd
    import zstandard

    out_raw.mkdir(parents=True, exist_ok=True)
    for d in days:
        df = read_region(US / "csv2017" / f"ais-{d}.csv.zst", box)
        text = df.to_csv(index=False).encode()
        # The file's own format: same columns and text as the daily file wrote them.
        (out_raw / f"ais-{d}.csv.zst").write_bytes(
            zstandard.ZstdCompressor(level=19).compress(text))
        print(f"  {d}: {len(df):,} rows, {df['mmsi'].nunique():,} MMSIs")


def write_shoreline(box, out_raw: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box as sbox

    src = US / "shoreline"
    dst = out_raw / "shoreline"
    dst.mkdir(parents=True, exist_ok=True)
    zone = sbox(box[0] - SHORELINE_MARGIN, box[2] - SHORELINE_MARGIN,
                box[1] + SHORELINE_MARGIN, box[3] + SHORELINE_MARGIN)
    lines = gpd.read_file(src / "us_medium_shoreline.shp", bbox=zone.bounds)
    lines = lines[lines.intersects(zone)].copy()
    lines["geometry"] = lines.geometry.intersection(zone)
    lines = lines[~lines.geometry.is_empty].explode(index_parts=False)
    lines = lines[lines.geometry.geom_type == "LineString"].reset_index(drop=True)
    lines.to_file(dst / "us_medium_shoreline.shp")
    for name in ("us_medium_shoreline.xml", "medium_shoreline_metadata_d.html", "us_medium_shoreline.prj"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"  shoreline: {len(lines):,} lines")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION)
    parser.add_argument("--days", nargs=2, default=list(DAYS), metavar="YYYY-MM-DD",
                        help="two consecutive days (default: 2017-11-01 2017-11-02)")
    parser.add_argument("--survey", action="store_true",
                        help="print positions, vessels and group mix per region and stop")
    parser.add_argument("--sample", type=Path, default=US / "sample",
                        help="where to write (default: data/UsAIS/sample)")
    parser.add_argument("--force", action="store_true", help="rebuild what exists")
    args = parser.parse_args(argv)

    if args.survey:
        survey(args.days)
        return 0

    box = REGIONS[args.region]
    raw, out = args.sample / "raw", args.sample / "preprocessed"
    t0 = time.perf_counter()
    print(f"{args.region}: {box[4]}, lon {box[0]}..{box[1]}, lat {box[2]}..{box[3]}")
    if args.force or not all((raw / f"ais-{d}.csv.zst").exists() for d in args.days):
        write_raw(args.days, box, raw)
    if args.force or not (raw / "shoreline" / "us_medium_shoreline.shp").exists():
        write_shoreline(box, raw)

    from catraj.preprocessing import preprocess_us

    months = tuple(sorted({d[:7] for d in args.days}))
    report = preprocess_us(raw, out, months, n_jobs=1, force=args.force)
    print(report.to_text())
    total = sum(p.stat().st_size for p in args.sample.rglob("*") if p.is_file())
    for sub in (raw, out):
        size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
        print(f"{sub.relative_to(ROOT) if sub.is_relative_to(ROOT) else sub}: "
              f"{size / 2**20:.1f} MB")
    print(f"done in {time.perf_counter() - t0:.0f} s; sample {total / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
