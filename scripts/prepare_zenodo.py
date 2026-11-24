"""Prepare the Zenodo data the notebooks read: descriptive names, noise removed.

Run it once after ``scripts/download_data.py``. It reads the archive's raw CSV
files in ``data/zenodo_1167595/`` and writes cleaned Parquet files to
``data/zenodo_1167595/clean/`` (git ignores the whole folder):

====================================  =========================================
``clean/ais_dynamic.parquet``         every AIS position (``nari_dynamic.csv``)
``clean/ocean_<month>.parquet``       sea state, one file per ``oc_<month>.csv``
``clean/weather.parquet``             hourly observations + station positions
====================================  =========================================

What it does to each dataset

* **Names.** The archive's coded column names become what the columns measure
  (``sourcemmsi`` -> ``mmsi``, ``hs`` -> ``wave_height``, ``Ff`` ->
  ``wind_speed``, ...): the same names the notebooks use.
* **Missing values.** The archives' "not available" codes become NaN: AIS speed
  over ground >= 102.3, course over ground >= 360, true heading >= 360 and rate
  of turn -128; ocean wave height, wave length and wave direction < 0; weather
  -65536. AIS rows whose MMSI is not a ship station's by ITU-R M.585 (nine
  digits, an allocated MID; :mod:`catraj.preprocessing.mmsi`; the classes
  found go to ``clean/mmsi_check.csv``) and AIS rows without a usable position
  are dropped, as are land cells of
  the ocean grid (``bottom_depth`` <= -16000).
* **Order.** AIS: duplicate (mmsi, unix_time) rows are dropped and the rows
  sorted by (mmsi, unix_time). Ocean and weather: sorted by (grid point or
  station, unix_time).
* **Vessel type.** ``vessel_type`` (and ``ship_type_code``) from
  ``nari_static.csv`` and ``Ship Types List.csv``, as notebook 1 derives them.
* **Noise.** A centred median over ``--window`` seconds (``catraj.preprocessing``,
  the package's own filter) of the positions, speed over ground, course over
  ground (a bearing, filtered on the circle) and true heading (likewise) per
  vessel; of sea level, wave height, wave length and wave direction (a
  bearing) per grid point; and of the numeric weather measurements per station.
  Codes are never filtered: ids, the compass-coded wind direction, the
  precipitation period, navigational status, ship type. The static
  ``bottom_depth`` is left as it is. After filtering, ``lon``/``lat`` of the
  AIS positions are recomputed from the filtered position, and ``x``/``y`` (UTM
  zone 30N, EPSG:32630, metres) are stored beside them.

The default window is 30 seconds, the package default. It changes AIS
positions, which arrive every few seconds, but leaves hourly weather and
three-hourly sea-state series unchanged (a 30 second window holds one
observation of them; see ``catraj/preprocessing/filtering.py``). The step is
there so that every column the notebooks use goes through the same filter.

Usage, from the repository root, in the ``catraj`` environment (it needs
``pyarrow``: ``pip install pyarrow``)::

    python scripts/prepare_zenodo.py                  # everything that was downloaded
    python scripts/prepare_zenodo.py --force          # rebuild files that exist
    python scripts/prepare_zenodo.py --window 60      # another window, in seconds
    python scripts/prepare_zenodo.py --data DIR       # read DIR, write DIR/clean

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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_1167595"
CRS = 32630                       # UTM zone 30N, metres
MONTHS = ("october", "november", "december", "january", "february", "march")
BATCH_ROWS = 1_500_000            # AIS rows filtered at once (whole vessels)
POINT_GAP = 1e10                  # seconds between two series in the stacked time axis

# The archives' coded column names -> what they measure (README_parameters.txt);
# the same dictionaries the notebooks use.
AIS_NAMES = {
    "sourcemmsi": "mmsi", "t": "unix_time", "navigationalstatus": "navigational_status",
    "rateofturn": "rate_of_turn", "speedoverground": "speed_over_ground",
    "courseoverground": "course_over_ground", "trueheading": "true_heading",
}
OCEAN_NAMES = {
    "dpt": "bottom_depth", "wlv": "sea_level", "hs": "wave_height",
    "lm": "wave_length", "dir": "wave_direction", "ts": "unix_time",
}
WEATHER_NAMES = {
    "id_station": "station_id", "local_time": "unix_time",
    "T": "air_temperature", "Tn": "min_air_temperature", "Tx": "max_air_temperature",
    "P": "air_pressure", "U": "relative_humidity", "Td": "dew_point",
    "id_windDirection": "wind_direction_code",
    "Ff": "wind_speed", "ff10": "max_gust_10min", "ff3": "max_gust", "VV": "visibility",
    "RRR": "precipitation", "tR": "precipitation_period",
}

# Filtered columns. Everything else (ids, codes) is passed through untouched.
AIS_FILTERED = ["speed_over_ground", "course_over_ground", "true_heading"]
AIS_CIRCULAR = ["course_over_ground", "true_heading"]
OCEAN_FILTERED = ["sea_level", "wave_height", "wave_length", "wave_direction"]
OCEAN_CIRCULAR = ["wave_direction"]
WEATHER_FILTERED = [
    "air_temperature", "min_air_temperature", "max_air_temperature", "air_pressure",
    "relative_humidity", "dew_point", "wind_speed", "max_gust_10min", "max_gust",
    "visibility", "precipitation",
]

AIS_USECOLS = list(AIS_NAMES) + ["lon", "lat"]
AIS_SPEED_NA, AIS_ANGLE_NA, AIS_TURN_NA = 102.3, 360.0, -128.0
LAND = -16000                      # bottom_depth at or below: land cell (undefined: -16384)
SEA_LEVEL_NA = -327.0              # sea_level at or below: undefined (-327.67)
WEATHER_NA = -65536


# ---- Parquet output ---------------------------------------------------------------

def _need_pyarrow() -> None:
    if importlib.util.find_spec("pyarrow") is None:
        sys.exit("pyarrow is needed to write Parquet: pip install pyarrow")


class _ParquetSink:
    """Write DataFrames to one Parquet file, batch by batch, via a temporary name."""

    def __init__(self, path: Path):
        self.path, self.tmp, self.writer = path, path.with_name(path.name + ".part"), None

    def write(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp, table.schema, compression="zstd")
        self.writer.write_table(table.cast(self.writer.schema))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            os.replace(self.tmp, self.path)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ---- shared steps -------------------------------------------------------------------

def _stack_time(codes, unix_time):
    """One time axis for many series: a series' times are 1e10 s apart from the next's,
    so a single ``rolling_filter`` call never mixes two series in a window."""
    return np.asarray(codes, dtype=float) * POINT_GAP + np.asarray(unix_time, dtype=float)


def _filter_series(df, codes, columns, circular, window):
    """Median-filter ``columns`` of ``df`` (sorted by series ``codes``, then time)."""
    from catraj.preprocessing import rolling_filter

    t = _stack_time(codes, df["unix_time"])
    values = df[columns].to_numpy(dtype=float, na_value=np.nan)
    flags = [c in circular for c in columns]
    done = rolling_filter(t, values, window, circular=flags)
    out = df.copy()
    for j, c in enumerate(columns):
        out[c] = done[:, j]
    return out


# ---- AIS ------------------------------------------------------------------------------

def _vessel_types(data: Path):
    """(mmsi -> ship type code, code -> type name), as notebook 1 derives them."""
    static = data / "P1_AIS_Data" / "nari_static.csv"
    codes = data / "P1_AIS_Codes" / "Ship Types List.csv"
    if not (static.exists() and codes.exists()):
        return None
    st = (pd.read_csv(static, usecols=["sourcemmsi", "shiptype"])
            .dropna().drop_duplicates("sourcemmsi", keep="last"))
    table = pd.read_csv(codes)
    names = {c: row.type_name.split(" - ")[0] for row in table.itertuples()
             for c in range(row.shiptype_min, row.shiptype_max + 1)}
    return st.set_index("sourcemmsi")["shiptype"], names


def _ais_batches(ais):
    start = 0
    while start < len(ais):
        stop = min(start + BATCH_ROWS, len(ais))
        if stop < len(ais):   # extend to the end of the vessel
            stop = int(np.searchsorted(ais["mmsi"].to_numpy(), ais["mmsi"].iat[stop - 1],
                                       side="right"))
        yield start, stop
        start = stop


def prepare_ais(data: Path, out: Path, window: float):
    import geopandas as gpd
    from pyproj import Transformer
    from catraj.preprocessing import denoise, mmsi_classes

    src = data / "P1_AIS_Data" / "nari_dynamic.csv"
    ais = pd.read_csv(
        src, usecols=AIS_USECOLS,
        dtype={"sourcemmsi": "int64", "t": "int64", "navigationalstatus": "float32",
               "rateofturn": "float32", "speedoverground": "float32",
               "courseoverground": "float32", "trueheading": "float32"},
    ).rename(columns=AIS_NAMES)
    n_in = len(ais)

    ais["speed_over_ground"] = ais.speed_over_ground.where(ais.speed_over_ground < AIS_SPEED_NA)
    ais["course_over_ground"] = ais.course_over_ground.where(ais.course_over_ground < AIS_ANGLE_NA)
    ais["true_heading"] = ais.true_heading.where(ais.true_heading < AIS_ANGLE_NA)
    ais["rate_of_turn"] = ais.rate_of_turn.where(ais.rate_of_turn != AIS_TURN_NA)
    if not ais.navigational_status.isna().any():     # a code: an integer when complete
        ais["navigational_status"] = ais.navigational_status.astype("int8")

    # MMSI check (ITU-R M.585, catraj.preprocessing.mmsi): keep ship stations,
    # nine digits with an allocated MID; set aside placeholders, coast stations,
    # aids to navigation, distress devices, ... and report every class.
    classes = mmsi_classes(ais.mmsi)
    check = (pd.DataFrame({"mmsi": ais.mmsi, "mmsi_class": classes})
               .groupby("mmsi_class").agg(MMSIs=("mmsi", "nunique"), positions=("mmsi", "size"))
               .sort_values("positions", ascending=False))
    check.reset_index().to_csv(out.with_name("mmsi_check.csv"), index=False)
    print("  MMSI check (ITU-R M.585), written to " + _rel(out.with_name("mmsi_check.csv")))
    print("    " + check.to_string().replace("\n", "\n    "))
    mmsi_ok = (classes == "ship").to_numpy()
    n_mmsi = int((~mmsi_ok).sum())
    n_mmsi_ids = int(ais.mmsi[~mmsi_ok].nunique())
    ais = ais[mmsi_ok]
    ok = ais.lon.between(-180, 180) & ais.lat.between(-90, 90)   # 181 / 91: "not available"
    n_bad = int((~ok).sum())
    ais = (ais[ok].drop_duplicates(["mmsi", "unix_time"])
              .sort_values(["mmsi", "unix_time"]).reset_index(drop=True))
    print(f"  {n_in:,} rows read, {n_mmsi:,} not from a ship station by ITU-R M.585 "
          f"({n_mmsi_ids:,} MMSIs), {n_bad:,} without a position, "
          f"{n_in - n_mmsi - n_bad - len(ais):,} duplicates dropped")

    types = _vessel_types(data)
    if types is None:
        print("  nari_static.csv / Ship Types List.csv absent: no vessel_type column")
    else:
        by_mmsi, names = types
        code = ais.mmsi.map(by_mmsi)
        ais["ship_type_code"] = code.astype("float32")    # NaN: no static message
        ais["vessel_type"] = code.map(names).fillna("Unknown")

    fwd = Transformer.from_crs(4326, CRS, always_xy=True)
    inv = Transformer.from_crs(CRS, 4326, always_xy=True)
    sink = _ParquetSink(out)
    n_out = 0
    try:
        for start, stop in _ais_batches(ais):
            b = ais.iloc[start:stop].copy()
            x, y = fwd.transform(b.lon.to_numpy(), b.lat.to_numpy())
            gdf = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(x, y), crs=CRS)
            gdf = denoise(gdf, window, by="mmsi", time_column="unix_time",
                          columns=AIS_FILTERED, circular=AIS_CIRCULAR)
            b["x"], b["y"] = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
            b["lon"], b["lat"] = inv.transform(b.x.to_numpy(), b.y.to_numpy())
            for c in AIS_FILTERED:
                v = gdf[c].to_numpy().astype("float32")
                if c in AIS_CIRCULAR:          # float32 rounds 359.99997 up to 360
                    v[v >= np.float32(360)] = 0
                b[c] = v
            sink.write(b)
            n_out += len(b)
    except BaseException:
        if sink.writer is not None:
            sink.writer.close()
        raise
    sink.close()
    return n_in, n_out


# ---- ocean ------------------------------------------------------------------------------

def prepare_ocean(src: Path, out: Path, window: float):
    oc = pd.read_csv(src).rename(columns=OCEAN_NAMES)
    n_in = len(oc)
    oc = oc[oc.bottom_depth > LAND].copy()                       # sea cells only
    for c in ("wave_height", "wave_length", "wave_direction"):   # negative = no value
        oc[c] = oc[c].where(oc[c] >= 0)
    # Sea level may be negative; its "undefined" is -327.67 (README_parameters.txt).
    oc["sea_level"] = oc.sea_level.where(oc.sea_level > SEA_LEVEL_NA)
    codes = oc.groupby(["lon", "lat"], sort=True).ngroup()
    oc = (oc.assign(_point=codes).sort_values(["_point", "unix_time"], kind="stable")
            .reset_index(drop=True))
    oc = _filter_series(oc, oc["_point"], OCEAN_FILTERED, OCEAN_CIRCULAR, window)
    oc = oc.drop(columns="_point")
    sink = _ParquetSink(out)
    sink.write(oc)
    sink.close()
    return n_in, len(oc)


# ---- weather ----------------------------------------------------------------------------

def prepare_weather(data: Path, out: Path, window: float):
    from pyproj import Transformer

    folder = data / "E2_Weather_Conditions"
    obs = pd.read_csv(folder / "table_wheatherObservation.csv")
    n_in = len(obs)
    obs = obs.replace(WEATHER_NA, np.nan)                       # the archive's "missing"
    sites = pd.read_csv(folder / "table_weatherStation.csv")
    obs = obs.merge(sites[["id_station", "latitude", "longitude"]], on="id_station")
    obs = obs.rename(columns=WEATHER_NAMES)
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(
        obs.longitude.to_numpy(), obs.latitude.to_numpy())
    obs["x"], obs["y"] = x, y
    obs = obs.sort_values(["station_id", "unix_time"], kind="stable").reset_index(drop=True)
    columns = [c for c in WEATHER_FILTERED if c in obs.columns]
    codes = pd.factorize(obs["station_id"])[0]                  # sorted, so non-decreasing
    obs = _filter_series(obs, codes, columns, [], window)
    sink = _ParquetSink(out)
    sink.write(obs)
    sink.close()
    return n_in, len(obs)


# ---- driver -------------------------------------------------------------------------------

def jobs(data: Path, clean: Path):
    """(label, raw files needed, output, function) for every dataset."""
    ais = data / "P1_AIS_Data" / "nari_dynamic.csv"
    yield "AIS", [ais], clean / "ais_dynamic.parquet", lambda w: prepare_ais(data, clean / "ais_dynamic.parquet", w)
    for month in MONTHS:
        raw = data / "E1_ocean" / f"oc_{month}.csv"
        target = clean / f"ocean_{month}.parquet"
        yield (f"ocean {month}", [raw], target,
               lambda w, raw=raw, target=target: prepare_ocean(raw, target, w))
    folder = data / "E2_Weather_Conditions"
    yield ("weather", [folder / "table_wheatherObservation.csv", folder / "table_weatherStation.csv"],
           clean / "weather.parquet",
           lambda w: prepare_weather(data, clean / "weather.parquet", w))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rename the Zenodo columns and median-filter them; write data/zenodo_1167595/clean/.",
        epilog="Run scripts/download_data.py first. Full guide: docs/getting_the_data.md")
    parser.add_argument("--data", type=Path, default=DATA,
                        help="folder holding the raw archive (default: data/zenodo_1167595); "
                             "output goes to DATA/clean")
    parser.add_argument("--window", type=float, default=30.0,
                        help="width of the median window, in seconds (default: 30)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild outputs that already exist")
    args = parser.parse_args(argv)
    if not args.window > 0:
        parser.error("--window must be positive")

    _need_pyarrow()
    data = args.data
    clean = data / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    print(f"{_rel(data)}/ -> {_rel(clean)}/  (median window {args.window:g} s)")

    done = skipped = 0
    for label, raw, target, run in jobs(data, clean):
        missing = [p for p in raw if not p.exists()]
        if missing:
            print(f"skip {label}: not downloaded ({_rel(missing[0])})")
            skipped += 1
            continue
        if target.exists() and not args.force:
            print(f"skip {label}: {_rel(target)} exists (--force to rebuild)")
            skipped += 1
            continue
        print(f"{label} ...")
        t0 = time.perf_counter()
        n_in, n_out = run(args.window)
        print(f"  {label}: {n_in:,} rows in, {n_out:,} out -> {_rel(target)} "
              f"({time.perf_counter() - t0:.1f} s)")
        done += 1
    print(f"\ndone: {done} written, {skipped} skipped")
    if not done and skipped:
        print("nothing to do: run scripts/download_data.py first, or pass --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
