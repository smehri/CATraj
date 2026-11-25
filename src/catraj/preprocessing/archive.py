"""Preprocessing of the Zenodo archive, one step per function.

The archive (Ray et al., 2019, doi:10.5281/zenodo.1167595) holds AIS messages,
coastal weather, a sea-state forecast and the European coastline as the
providers wrote them: coded column names, "not available" codes inside the
numbers, unchecked ranges. This module turns each into a clean table. Every
step is a public function that takes a table and returns one, so a notebook can
run them one per cell and look at what each does; :func:`preprocess_archive`
runs them all and writes the files the notebooks read.

Every function takes an optional ``report`` (a :class:`PreprocessingReport`)
and, when given one, records the rows in and out and the values it set to NaN,
per rule. ``preprocess_archive`` writes that record as
``preprocessing_report.csv`` and ``.txt``.

Names, units and "not available" codes come from each folder's
``README_parameters.txt``; the rules below say where each limit comes from.
A value outside its range becomes NaN; a row whose identifier, time or position
is invalid is dropped, because nothing can be done with it.

**January 2016 is left out of every dataset** (decided with the author,
2026-11-25). The archive has no sea state for it: its ``oc_january.csv`` is a
byte-for-byte copy of ``oc_november.csv`` (times 2015-11-01 to 2015-11-30; the
zip's md5 matches the one Zenodo publishes, so the copy is in the source).
So that every notebook works on the same period, :func:`drop_excluded_periods`
drops the AIS, weather and sea-state rows of January (:data:`EXCLUDED_PERIODS`)
and :func:`ocean_month_matches` refuses to write the copy as January. The data
then covers October to December 2015 and February to March 2016. Pass
``exclude=()`` to :func:`preprocess_archive` to keep January's AIS and weather.

AIS (``P1_AIS_Data/nari_dynamic.csv``)
--------------------------------------

====================  ======================================  ==================
rule                  keep                                    source
====================  ======================================  ==================
``mmsi``              a ship station: nine digits, an         ITU-R M.585,
                      allocated MID                           :mod:`.mmsi`
``unix_time``         2015-10-01 00:00:00 to 2016-03-31       README, "Period"
                      23:59:59 UTC
``lon``, ``lat``      -180..180 and -90..90 (181 / 91 mean    AIS standard
                      "not available"), then inside the
                      archive's area, -10..0 and 45..51       README, "Coverage"
``speed_over_ground`` 0 <= v < 102.3 knots (102.3: n/a)       README, ITU-R M.1371
``course_over_ground``0 <= v < 360 degrees (360: n/a)         README
``true_heading``      0 <= v < 360 degrees (511: n/a)         README
``rate_of_turn``      -127..127, the coded value; -128 is n/a README, ITU-R M.1371
``navigational_status`` the integers 0..15                    ``Navigational
                                                              Status.csv``
====================  ======================================  ==================

A row without a ship MMSI, time or position is dropped; the other values become
NaN. ``rate_of_turn`` is the AIS code, not degrees per minute: the README's
"0 to 720 degrees per minute" is the range the code ``4.733 * sqrt(rate)``
covers, 0 to 127 (127 * 127 / 4.733**2 = 720). -127 and 127 mean "turning at
more than 5 degrees per 30 s, no turn indicator" and are kept, as before. Then
duplicate (mmsi, unix_time) rows are dropped and the rows sorted by (mmsi,
unix_time), isolated position jumps are dropped (below), and ``vessel_type``
and ``ship_type_code`` are attached from
``nari_static.csv`` and ``Ship Types List.csv`` (the last static message of the
vessel; "Unknown" without one).

**Position jumps.** :func:`drop_position_jumps` drops an isolated spike: a
position whose speed from the vessel's previous position *and* to its next
position both exceed ``max_speed_knots`` (50 by default: faster than any ship
in the archive but a few wing-in-ground and military craft, and far below what
a decoding error or a wrong fix implies), while the previous and the next
position, taken directly, do not (a vessel that really moves faster, with every
position consistent, is kept). The first and last position of a vessel have one
neighbour: they are dropped when the speed to it exceeds the limit and the
neighbour's own next segment does not. Great-circle distance over time; a zero
time gap with a different position is infinite speed, with the same position
zero. One vectorised pass over all vessels, on the positions as given, so a
run of several consecutive spikes (each close to the next) is not removed;
isolated ones, the usual decoding error, are. ``max_speed_knots=None``
disables the rule.

Weather (``E2_Weather_Conditions``)
-----------------------------------

The README gives units but no limits, so the limits are physical ones, wide
enough to hold every real observation (the world records, in the README's
units): air, minimum, maximum and dew point temperature -90..60 degrees C;
pressure 640..820 mm of mercury (870..1085 hPa is the record range); relative
humidity 0..100 %; mean wind speed and the two gusts 0..120 m/s; visibility
0..100 km; precipitation 0..1000 mm; its period 0..24 hours. The wind direction
code must be in ``table_windDirection.csv``. -65536 is the archive's "missing"
in every column. A row whose station is not in ``table_weatherStation.csv`` or
whose time is outside the archive's period is dropped.

Ocean (``E1_ocean/oc_<month>.csv``)
-----------------------------------

Undefined values, from the README: depth -16384, sea level -327.67, wave height
-65.534, wave length -32767, wave direction -3276.7. A cell whose depth is
-16000 or less is land and its rows are dropped (the other fields of such rows
are all undefined); a negative wave height, length or direction, and a sea
level of -327 or less, is "undefined" and becomes NaN. Ranges: depth at most
11000 m (the deepest ocean), sea level -20..20 m, wave height 0..30 m, wave
length 0..2000 m, wave direction 0 <= v < 360 degrees. Rows outside the
archive's area or period are dropped.

Coastline (``C2_European_Coastline``)
-------------------------------------

The EEA polygons (EPSG:3035) inside the archive's area are read, reprojected to
EPSG:32630 (UTM zone 30N, metres, the CRS of the positions) and clipped to the
area's outline there.

Projection and noise
--------------------

AIS positions and weather stations get ``x`` and ``y`` in EPSG:32630. Then
a centred median over ``window`` seconds (:func:`catraj.preprocessing.denoise`;
30 s by default) is taken of the AIS positions, speed, course (on the circle)
and heading (on the circle) per vessel; of sea level, wave height, wave length
and wave direction (on the circle) per grid point; of the numeric measurements
per weather station. Codes are never filtered. AIS ``lon`` and ``lat`` are
recomputed from the filtered position. The window changes AIS, which arrives
every few seconds, and leaves hourly weather and three-hourly sea state as they
are (one observation fits in the window).

Output of :func:`preprocess_archive`
------------------------------------

``AIS/ais.parquet`` (+ ``AIS/mmsi_check.csv``, the MMSI classes found),
``Weather/weather.parquet``, ``Ocean/ocean_<month>.parquet``,
``Coastline/coastline.parquet`` (GeoParquet), ``preprocessing_report.csv`` and
``.txt``. A dataset whose raw files are absent is skipped. ``n_jobs`` spreads
the independent work over processes (weather, each month of sea state and the
coastline; then the AIS filter per batch of whole vessels) and changes nothing
in the files or the report.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .._parallel import resolve_n_jobs, run_parallel
from .filtering import denoise, rolling_filter
from .mmsi import mmsi_classes

__all__ = [
    "AREA", "ARCHIVE_PERIOD", "CRS", "EXCLUDED_PERIODS", "MONTHS",
    "PreprocessingReport", "ocean_month_matches", "drop_excluded_periods",
    "read_ais_raw", "mmsi_check", "drop_invalid_mmsi", "drop_invalid_time",
    "drop_invalid_position", "unavailable_to_nan", "check_ranges",
    "drop_duplicates_and_sort", "drop_position_jumps", "vessel_types",
    "attach_vessel_types", "project",
    "iter_denoised_ais", "denoise_ais",
    "read_weather_raw", "weather_unavailable_to_nan", "merge_weather_stations",
    "check_weather_ranges", "drop_invalid_weather", "denoise_weather",
    "read_ocean_raw", "drop_invalid_ocean", "drop_land_cells",
    "ocean_unavailable_to_nan", "check_ocean_ranges", "sort_ocean", "denoise_ocean",
    "read_coastline", "project_coastline", "clip_coastline", "preprocess_archive",
]

CRS = 32630                         # UTM zone 30N, metres
AREA = (-10.0, 0.0, 45.0, 51.0)     # lon_min, lon_max, lat_min, lat_max (README)
ARCHIVE_PERIOD = (                  # unix seconds, both ends included (README)
    int(pd.Timestamp("2015-10-01 00:00:00", tz="UTC").timestamp()),
    int(pd.Timestamp("2016-03-31 23:59:59", tz="UTC").timestamp()),
)
MONTHS = ("october", "november", "december", "january", "february", "march")
#: Periods left out of every dataset: (first second, last second, reason).
EXCLUDED_PERIODS = (
    (1451606400, 1454284799,             # 2016-01-01 00:00 to 2016-01-31 23:59:59
     "January 2016: no sea state in the archive (oc_january.csv repeats "
     "November)"),
)
BATCH_ROWS = 1_500_000              # AIS rows filtered at once (whole vessels)
POINT_GAP = 1e10                    # seconds between two series in a stacked time axis
COASTLINE_CRS = 3035                # what the EEA file is in

# The archive's coded column names -> what they measure (README_parameters.txt).
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
AIS_TURN_MAX = 127.0
STATUS_CODES = tuple(range(16))
LAND = -16000                       # bottom_depth at or below: land (undefined: -16384)
SEA_LEVEL_NA = -327.0               # sea_level at or below: undefined (-327.67)
WEATHER_NA = -65536

# (column, low, high, high is included) of the weather and ocean range rules.
WEATHER_RANGES = [
    ("air_temperature", -90, 60, True), ("min_air_temperature", -90, 60, True),
    ("max_air_temperature", -90, 60, True), ("dew_point", -90, 60, True),
    ("air_pressure", 640, 820, True), ("relative_humidity", 0, 100, True),
    ("wind_speed", 0, 120, True), ("max_gust_10min", 0, 120, True),
    ("max_gust", 0, 120, True), ("visibility", 0, 100, True),
    ("precipitation", 0, 1000, True), ("precipitation_period", 0, 24, True),
]
OCEAN_RANGES = [
    ("sea_level", -20, 20, True), ("wave_height", 0, 30, True),
    ("wave_length", 0, 2000, True), ("wave_direction", 0, 360, False),
]
OCEAN_DEPTH_MAX = 11_000.0


# ---- the record of what each step did -------------------------------------------------

class PreprocessingReport:
    """What each preprocessing step did: rows in and out, values set to NaN.

    One line per rule: ``dataset``, ``step`` (the function), ``rule``,
    ``rows_in``, ``rows_out``, ``rows_dropped`` and ``values_to_nan``. A rule
    that sets values to NaN keeps every row.
    """

    COLUMNS = ["dataset", "step", "rule", "rows_in", "rows_out", "rows_dropped",
               "values_to_nan"]

    def __init__(self):
        self._rows: list[dict] = []

    def add(self, dataset, step, rule, rows_in, rows_out, values_to_nan=0):
        self._rows.append(dict(
            dataset=dataset, step=step, rule=rule, rows_in=int(rows_in),
            rows_out=int(rows_out), rows_dropped=int(rows_in) - int(rows_out),
            values_to_nan=int(values_to_nan)))

    def extend(self, other: "PreprocessingReport") -> None:
        self._rows.extend(other._rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=self.COLUMNS)

    def to_text(self) -> str:
        return self.to_frame().to_string(index=False)

    def __len__(self):
        return len(self._rows)

    def __repr__(self):
        return self.to_text()


def _drop(df, keep, report, dataset, step, rule):
    """``df[keep]``, the rows dropped recorded under ``rule``."""
    keep = np.asarray(keep, dtype=bool)
    out = df[keep]
    if report is not None:
        report.add(dataset, step, rule, len(df), len(out))
    return out


def _to_nan(df, column, bad, report, dataset, step, rule):
    """Set ``column`` to NaN where ``bad``; count the values that were present."""
    bad = np.asarray(bad, dtype=bool) & df[column].notna().to_numpy()
    n = int(bad.sum())
    if n:
        df[column] = df[column].mask(bad)
    if report is not None:
        report.add(dataset, step, rule, len(df), len(df), n)
    return n


def _label(low, high, high_included=True):
    return f"{low}..{high}" if high_included else f"{low}..<{high}"


def _in_range(s, low, high, high_included=True):
    return (s >= low) & ((s <= high) if high_included else (s < high))


# ---- AIS --------------------------------------------------------------------------------

def read_ais_raw(path, report=None) -> pd.DataFrame:
    """Read ``nari_dynamic.csv`` with the descriptive column names.

    Columns: ``mmsi``, ``navigational_status``, ``rate_of_turn``,
    ``speed_over_ground``, ``course_over_ground``, ``true_heading``, ``lon``,
    ``lat``, ``unix_time``. The measurements are float32, the rest as stored.
    """
    ais = pd.read_csv(
        path, usecols=AIS_USECOLS,
        dtype={"sourcemmsi": "int64", "t": "int64", "navigationalstatus": "float32",
               "rateofturn": "float32", "speedoverground": "float32",
               "courseoverground": "float32", "trueheading": "float32"},
    ).rename(columns=AIS_NAMES)
    if report is not None:
        report.add("AIS", "read_ais_raw", "rows read", len(ais), len(ais))
    return ais


def mmsi_check(ais) -> pd.DataFrame:
    """The ITU-R M.585 classes found: MMSIs and positions per class."""
    classes = mmsi_classes(ais["mmsi"])
    return (pd.DataFrame({"mmsi": ais["mmsi"], "mmsi_class": classes})
              .groupby("mmsi_class")
              .agg(MMSIs=("mmsi", "nunique"), positions=("mmsi", "size"))
              .sort_values("positions", ascending=False).reset_index())


def drop_invalid_mmsi(ais, report=None) -> pd.DataFrame:
    """Keep the ship stations: nine digits, an allocated MID (ITU-R M.585).

    Placeholders, coast stations, aids to navigation, distress devices and
    the like are dropped; the report has one line per class dropped.
    """
    classes = mmsi_classes(ais["mmsi"])
    ok = (classes == "ship").to_numpy()
    if report is not None:
        for name, n in classes[~ok].value_counts().items():
            report.add("AIS", "drop_invalid_mmsi", f"mmsi: {name}", n, 0)
        report.add("AIS", "drop_invalid_mmsi", "mmsi: ship station kept",
                   len(ais), int(ok.sum()))
    return ais[ok]


def drop_invalid_time(df, report=None, dataset="AIS", column="unix_time",
                      period=ARCHIVE_PERIOD) -> pd.DataFrame:
    """Drop rows whose time is missing or outside the archive's period."""
    ok = _in_range(df[column], period[0], period[1])
    return _drop(df, ok, report, dataset, "drop_invalid_time",
                 f"{column} inside the archive period")


def drop_excluded_periods(df, periods=EXCLUDED_PERIODS, report=None,
                          dataset="AIS", column="unix_time") -> pd.DataFrame:
    """Drop the rows inside any of ``periods`` (by default, January 2016).

    ``periods`` holds ``(first, last, reason)`` tuples, unix seconds, both ends
    included; ``None`` or ``()`` keeps everything.
    """
    for first, last, reason in periods or ():
        t = df[column]
        df = _drop(df, ~((t >= first) & (t <= last)), report, dataset,
                   "drop_excluded_periods", reason)
    return df


def drop_invalid_position(ais, area=AREA, report=None, dataset="AIS",
                          lon="lon", lat="lat") -> pd.DataFrame:
    """Drop rows without a position: -180..180 / -90..90, then inside ``area``.

    181 and 91 are the AIS codes for "not available". ``area`` is
    ``(lon_min, lon_max, lat_min, lat_max)``, by default the archive's own;
    ``None`` skips that second rule.
    """
    step = "drop_invalid_position"
    ais = _drop(ais, ais[lon].between(-180, 180) & ais[lat].between(-90, 90),
                report, dataset, step, "lon, lat in valid range")
    if area is not None:
        inside = ais[lon].between(area[0], area[1]) & ais[lat].between(area[2], area[3])
        ais = _drop(ais, inside, report, dataset, step, "position inside archive area")
    return ais


def unavailable_to_nan(ais, report=None) -> pd.DataFrame:
    """Set the AIS "not available" codes to NaN.

    Speed over ground of 102.3 knots or more, course over ground and true
    heading of 360 degrees or more (heading 511), rate of turn -128.
    """
    ais = ais.copy()
    step = "unavailable_to_nan"
    _to_nan(ais, "speed_over_ground", ais.speed_over_ground >= AIS_SPEED_NA,
            report, "AIS", step, "speed_over_ground >= 102.3 (n/a)")
    _to_nan(ais, "course_over_ground", ais.course_over_ground >= AIS_ANGLE_NA,
            report, "AIS", step, "course_over_ground >= 360 (n/a)")
    _to_nan(ais, "true_heading", ais.true_heading >= AIS_ANGLE_NA,
            report, "AIS", step, "true_heading >= 360 (511 n/a)")
    _to_nan(ais, "rate_of_turn", ais.rate_of_turn == AIS_TURN_NA,
            report, "AIS", step, "rate_of_turn == -128 (n/a)")
    return ais


def check_ranges(ais, report=None) -> pd.DataFrame:
    """Set to NaN every AIS value outside its range (see the module docstring).

    Speed below 0, course or heading outside 0..360, rate of turn beyond
    +-127, a navigational status that is not an integer 0..15. The "not
    available" codes are :func:`unavailable_to_nan`'s; the upper limits are
    checked here as well, so the function is complete on its own. The status
    becomes an ``int8`` column when no value is missing.
    """
    ais = ais.copy()
    step = "check_ranges"
    for col, high in (("speed_over_ground", AIS_SPEED_NA),
                      ("course_over_ground", AIS_ANGLE_NA),
                      ("true_heading", AIS_ANGLE_NA)):
        _to_nan(ais, col, ~_in_range(ais[col], 0, high, False), report, "AIS", step,
                f"{col} outside 0..<{high:g}")
    turn = f"-{AIS_TURN_MAX:g}..{AIS_TURN_MAX:g}"
    _to_nan(ais, "rate_of_turn", ~ais.rate_of_turn.abs().le(AIS_TURN_MAX),
            report, "AIS", step, f"rate_of_turn outside {turn}")
    status = ais.navigational_status
    _to_nan(ais, "navigational_status", ~status.isin(STATUS_CODES),
            report, "AIS", step, "navigational_status not a code 0..15")
    if not ais.navigational_status.isna().any():     # a code: an integer when complete
        ais["navigational_status"] = ais.navigational_status.astype("int8")
    return ais


def drop_duplicates_and_sort(ais, report=None) -> pd.DataFrame:
    """Drop repeated (mmsi, unix_time) rows and sort by (mmsi, unix_time)."""
    out = ais.drop_duplicates(["mmsi", "unix_time"]).sort_values(["mmsi", "unix_time"])
    if report is not None:
        report.add("AIS", "drop_duplicates_and_sort", "duplicate (mmsi, unix_time)",
                   len(ais), len(out))
    return out.reset_index(drop=True)


def _knots(lat0, lon0, lat1, lon1, dt):
    """Great-circle speed between consecutive positions, in knots (inf: dt <= 0)."""
    p0, p1 = np.radians(lat0), np.radians(lat1)
    a = (np.sin((p1 - p0) / 2) ** 2
         + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lon1 - lon0) / 2) ** 2)
    metres = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dt = np.asarray(dt, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = metres / dt * 3600 / 1852
    return np.where(dt > 0, v, np.where(metres > 0, np.inf, 0.0))


def drop_position_jumps(ais, max_speed_knots=50, report=None):
    """Drop isolated position spikes (see "Position jumps" in the module docstring).

    ``ais`` must be sorted by ``mmsi`` and ``unix_time``. A position goes when
    the speeds from its previous and to its next position both exceed
    ``max_speed_knots`` but the speed from the previous to the next does not;
    a vessel's first or last position when its one speed exceeds it and the
    neighbour's next segment does not. ``None`` returns ``ais`` unchanged.
    """
    if max_speed_knots is None:
        return ais
    n = len(ais)
    m = ais["mmsi"].to_numpy()
    t = ais["unix_time"].to_numpy(dtype=float)
    lon = ais["lon"].to_numpy(dtype=float)
    lat = ais["lat"].to_numpy(dtype=float)
    out = np.zeros(n, dtype=bool)
    if n >= 3:
        same = m[1:] == m[:-1]
        fast = same & (_knots(lat[:-1], lon[:-1], lat[1:], lon[1:], t[1:] - t[:-1])
                       > max_speed_knots)
        skip = (m[:-2] == m[2:]) & (
            _knots(lat[:-2], lon[:-2], lat[2:], lon[2:], t[2:] - t[:-2])
            > max_speed_knots)
        has_prev, has_next = np.r_[False, same], np.r_[same, False]
        fast_next, fast_prev = np.r_[fast, False], np.r_[False, fast]
        skip = np.r_[False, skip, False]
        inner = has_prev & has_next & fast_prev & fast_next & ~skip
        has_next2 = has_next & np.r_[has_next[1:], False]
        first = (~has_prev & fast_next & has_next2 & ~np.r_[fast_next[1:], False])
        has_prev2 = has_prev & np.r_[False, has_prev[:-1]]
        last = (~has_next & fast_prev & has_prev2 & ~np.r_[False, fast_prev[:-1]])
        out = inner | first | last
    result = ais[~out]
    if report is not None:
        report.add("AIS", "drop_position_jumps",
                   f"isolated jump above {max_speed_knots:g} knots", n, len(result))
    return result.reset_index(drop=True)


def vessel_types(static_csv, ship_types_csv):
    """``(mmsi -> ship type code, code -> type name)`` from the static messages.

    The last row of ``nari_static.csv`` that has a type, per vessel; the names
    from ``Ship Types List.csv`` (the part before `` - ``).
    """
    st = (pd.read_csv(static_csv, usecols=["sourcemmsi", "shiptype"])
            .dropna().drop_duplicates("sourcemmsi", keep="last"))
    table = pd.read_csv(ship_types_csv)
    names = {c: row.type_name.split(" - ")[0] for row in table.itertuples()
             for c in range(row.shiptype_min, row.shiptype_max + 1)}
    return st.set_index("sourcemmsi")["shiptype"], names


def attach_vessel_types(ais, static_csv, ship_types_csv, report=None) -> pd.DataFrame:
    """Add ``ship_type_code`` (NaN: no static message) and ``vessel_type``.

    ``vessel_type`` is ``"Unknown"`` when the vessel sent no static message or
    its code is not in the type table (0, "not available", for one).  The
    report counts both, as two rules.
    """
    by_mmsi, names = vessel_types(static_csv, ship_types_csv)
    ais = ais.copy()
    code = ais["mmsi"].map(by_mmsi)
    ais["ship_type_code"] = code.astype("float32")
    named = code.map(names)
    ais["vessel_type"] = named.fillna("Unknown")
    if report is not None:
        no_static = int(code.isna().sum())
        report.add("AIS", "attach_vessel_types", "Unknown: no static message",
                   len(ais), len(ais), no_static)
        report.add("AIS", "attach_vessel_types", "Unknown: code not in type table",
                   len(ais), len(ais), int(named.isna().sum()) - no_static)
    return ais


def project(df, lon="lon", lat="lat", crs=CRS) -> pd.DataFrame:
    """Add ``x`` and ``y``, ``lon`` and ``lat`` projected to ``crs`` (metres)."""
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(
        df[lon].to_numpy(dtype=float), df[lat].to_numpy(dtype=float))
    out = df.copy()
    out["x"], out["y"] = x, y
    return out


def _vessel_batches(mmsi, batch_rows):
    start = 0
    while start < len(mmsi):
        stop = min(start + batch_rows, len(mmsi))
        if stop < len(mmsi):     # extend to the end of the vessel
            stop = int(np.searchsorted(mmsi, mmsi[stop - 1], side="right"))
        yield start, stop
        start = stop


def _denoise_batch(b, window):
    """One batch of whole vessels, median-filtered (what a worker process runs)."""
    import geopandas as gpd
    from pyproj import Transformer

    inv = Transformer.from_crs(CRS, 4326, always_xy=True)
    if "x" not in b.columns:
        b = project(b)
    else:
        b = b.copy()
    gdf = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.x, b.y), crs=CRS)
    gdf = denoise(gdf, window, by="mmsi", time_column="unix_time",
                  columns=AIS_FILTERED, circular=AIS_CIRCULAR)
    b["x"], b["y"] = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    b["lon"], b["lat"] = inv.transform(b.x.to_numpy(), b.y.to_numpy())
    for c in AIS_FILTERED:
        v = gdf[c].to_numpy().astype("float32")
        if c in AIS_CIRCULAR:          # float32 rounds 359.99997 up to 360
            v[v >= np.float32(360)] = 0
        b[c] = v
    return b


def iter_denoised_ais(ais, window=30.0, batch_rows=BATCH_ROWS, n_jobs=1):
    """:func:`denoise_ais` in batches of whole vessels, to bound memory.

    ``ais`` must be sorted by ``mmsi``. Yields one frame per batch, in order.
    Vessels are independent, so with ``n_jobs`` other than 1 the batches are
    filtered on that many processes (``-1``: every core), ``n_jobs`` batches at
    a time; the frames are the same as a serial run's. Memory: each worker
    holds one batch (``batch_rows`` rows) and its filtered copy.
    """
    batches = list(_vessel_batches(ais["mmsi"].to_numpy(), batch_rows))
    workers = resolve_n_jobs(n_jobs, len(batches))
    for first in range(0, len(batches), workers):
        wave = batches[first:first + workers]
        tasks = [(ais.iloc[start:stop], window) for start, stop in wave]
        yield from run_parallel(_denoise_batch, tasks, n_jobs=workers)


def denoise_ais(ais, window=30.0, report=None, n_jobs=1) -> pd.DataFrame:
    """Median-filter positions, speed, course and heading per vessel.

    ``x`` and ``y`` (projected, see :func:`project`) are filtered, ``lon`` and
    ``lat`` recomputed from them. ``ais`` must be sorted by ``mmsi`` and time.
    ``n_jobs`` as in :func:`iter_denoised_ais`.
    """
    parts = list(iter_denoised_ais(ais, window, n_jobs=n_jobs))
    out = pd.concat(parts, ignore_index=True) if parts else ais.iloc[:0].copy()
    if report is not None:
        report.add("AIS", "denoise_ais", f"median filter, {window:g} s window",
                   len(ais), len(out))
    return out


# ---- shared: filtering many series at once -------------------------------------------

def _filter_series(df, codes, columns, circular, window):
    """Median-filter ``columns`` of ``df`` (sorted by series ``codes``, then time).

    The series' times are stacked 1e10 s apart, so that one call of
    ``rolling_filter`` never mixes two series in a window.
    """
    t = np.asarray(codes, dtype=float) * POINT_GAP + np.asarray(
        df["unix_time"], dtype=float)
    values = df[columns].to_numpy(dtype=float, na_value=np.nan)
    flags = [c in circular for c in columns]
    done = rolling_filter(t, values, window, circular=flags)
    out = df.copy()
    for j, c in enumerate(columns):
        out[c] = done[:, j]
    return out


# ---- weather -------------------------------------------------------------------------------

def read_weather_raw(folder, report=None):
    """``(observations, stations, wind_directions)`` of ``E2_Weather_Conditions``.

    The observations keep the archive's column names until
    :func:`merge_weather_stations`; ``wind_directions`` is ``None`` when
    ``table_windDirection.csv`` is absent.
    """
    folder = Path(folder)
    obs = pd.read_csv(folder / "table_wheatherObservation.csv")
    stations = pd.read_csv(folder / "table_weatherStation.csv")
    wind = folder / "table_windDirection.csv"
    wind = pd.read_csv(wind) if wind.exists() else None
    if report is not None:
        report.add("weather", "read_weather_raw", "rows read", len(obs), len(obs))
    return obs, stations, wind


def weather_unavailable_to_nan(obs, report=None) -> pd.DataFrame:
    """Set the archive's "missing" code, -65536, to NaN in every column."""
    obs = obs.copy()
    for col in obs.columns:
        if col in ("id_station", "local_time"):
            continue
        name = WEATHER_NAMES.get(col, col)
        _to_nan(obs, col, obs[col] == WEATHER_NA, report, "weather",
                "weather_unavailable_to_nan", f"{name} == -65536")
    return obs


def merge_weather_stations(obs, stations, report=None) -> pd.DataFrame:
    """Rename the columns and add each station's ``latitude`` and ``longitude``.

    Observations of a station that is not in the station table are dropped.
    """
    out = obs.merge(stations[["id_station", "latitude", "longitude"]], on="id_station")
    if report is not None:
        report.add("weather", "merge_weather_stations", "station in station table",
                   len(obs), len(out))
    return out.rename(columns=WEATHER_NAMES)


def drop_invalid_weather(obs, report=None) -> pd.DataFrame:
    """Drop observations whose time is outside the archive's period."""
    return drop_invalid_time(obs, report, dataset="weather")


def check_weather_ranges(obs, wind_directions=None, report=None) -> pd.DataFrame:
    """Set to NaN every weather value outside its range (module docstring).

    ``wind_directions`` is ``table_windDirection.csv``: a code not in it
    becomes NaN too.
    """
    obs = obs.copy()
    step = "check_weather_ranges"
    for col, low, high, closed in WEATHER_RANGES:
        if col in obs.columns:
            _to_nan(obs, col, ~_in_range(obs[col], low, high, closed), report,
                    "weather", step, f"{col} outside {_label(low, high, closed)}")
    if wind_directions is not None and "wind_direction_code" in obs.columns:
        _to_nan(obs, "wind_direction_code",
                ~obs.wind_direction_code.isin(wind_directions["id_windDirection"]),
                report, "weather", step, "wind_direction_code not in table")
    return obs


def denoise_weather(obs, window=30.0, report=None) -> pd.DataFrame:
    """Project the stations, sort by (station, time) and median-filter each series.

    Adds ``x`` and ``y`` (EPSG:32630) if absent.
    """
    if "x" not in obs.columns:
        obs = project(obs, "longitude", "latitude")
    obs = obs.sort_values(["station_id", "unix_time"], kind="stable")
    obs = obs.reset_index(drop=True)
    columns = [c for c in WEATHER_FILTERED if c in obs.columns]
    codes = pd.factorize(obs["station_id"])[0]          # sorted, so non-decreasing
    out = _filter_series(obs, codes, columns, [], window)
    if report is not None:
        report.add("weather", "denoise_weather", f"median filter, {window:g} s window",
                   len(obs), len(out))
    return out


# ---- ocean --------------------------------------------------------------------------------

def read_ocean_raw(path, report=None) -> pd.DataFrame:
    """Read ``oc_<month>.csv`` with the descriptive column names.

    Columns: ``lon``, ``lat``, ``bottom_depth``, ``sea_level``, ``wave_height``,
    ``wave_length``, ``wave_direction``, ``unix_time``.
    """
    oc = pd.read_csv(path).rename(columns=OCEAN_NAMES)
    if report is not None:
        report.add("ocean", "read_ocean_raw", "rows read", len(oc), len(oc))
    return oc


def drop_invalid_ocean(oc, area=AREA, report=None) -> pd.DataFrame:
    """Drop rows outside the archive's period or (unless ``area`` is None) area."""
    oc = drop_invalid_time(oc, report, dataset="ocean")
    if area is not None:
        inside = oc.lon.between(area[0], area[1]) & oc.lat.between(area[2], area[3])
        oc = _drop(oc, inside, report, "ocean", "drop_invalid_ocean",
                   "cell inside archive area")
    return oc


def drop_land_cells(oc, report=None) -> pd.DataFrame:
    """Keep the sea cells: ``bottom_depth`` above -16000 (undefined: -16384)."""
    return _drop(oc, oc.bottom_depth > LAND, report, "ocean", "drop_land_cells",
                 "bottom_depth > -16000 (sea cell)")


def ocean_unavailable_to_nan(oc, report=None) -> pd.DataFrame:
    """Set the undefined codes to NaN: negative wave fields, sea level <= -327.

    Sea level may be negative (a tide), so only its code, -327.67, is undefined.
    """
    oc = oc.copy()
    step = "ocean_unavailable_to_nan"
    for col in ("wave_height", "wave_length", "wave_direction"):
        _to_nan(oc, col, oc[col] < 0, report, "ocean", step, f"{col} < 0 (undefined)")
    _to_nan(oc, "sea_level", oc.sea_level <= SEA_LEVEL_NA, report, "ocean", step,
            "sea_level <= -327 (undefined)")
    return oc


def check_ocean_ranges(oc, report=None) -> pd.DataFrame:
    """Set to NaN every ocean value outside its range (module docstring)."""
    oc = oc.copy()
    step = "check_ocean_ranges"
    _to_nan(oc, "bottom_depth", oc.bottom_depth > OCEAN_DEPTH_MAX, report, "ocean",
            step, f"bottom_depth > {OCEAN_DEPTH_MAX:g}")
    for col, low, high, closed in OCEAN_RANGES:
        _to_nan(oc, col, ~_in_range(oc[col], low, high, closed), report, "ocean", step,
                f"{col} outside {_label(low, high, closed)}")
    return oc


def sort_ocean(oc, report=None) -> pd.DataFrame:
    """Sort by grid point (lon, lat), then time."""
    out = oc.sort_values(["lon", "lat", "unix_time"], kind="stable")
    if report is not None:
        report.add("ocean", "sort_ocean", "sorted by (lon, lat, unix_time)",
                   len(oc), len(out))
    return out.reset_index(drop=True)


def denoise_ocean(oc, window=30.0, report=None) -> pd.DataFrame:
    """Median-filter each grid point's series (``oc`` sorted by :func:`sort_ocean`)."""
    codes = oc.groupby(["lon", "lat"], sort=True).ngroup()
    out = _filter_series(oc, codes, OCEAN_FILTERED, OCEAN_CIRCULAR, window)
    if report is not None:
        report.add("ocean", "denoise_ocean", f"median filter, {window:g} s window",
                   len(oc), len(out))
    return out


# ---- coastline ------------------------------------------------------------------------------

def _outline(area, crs):
    """The area's rectangle (densified, so its edges curve) in ``crs``."""
    import geopandas as gpd
    from shapely.geometry import box

    rect = box(area[0], area[2], area[1], area[3]).segmentize(0.01)
    return gpd.GeoSeries([rect], crs=4326).to_crs(crs).iloc[0]


def read_coastline(path, area=AREA, report=None):
    """The coastline polygons that touch ``area`` (``None``: all), in the file's CRS."""
    import geopandas as gpd
    from pyogrio import read_info

    bbox = None
    if area is not None:
        bbox = tuple(_outline(area, read_info(path)["crs"]).bounds)
    coast = gpd.read_file(path, bbox=bbox)
    if report is not None:
        report.add("coastline", "read_coastline", "polygons touching the area",
                   len(coast), len(coast))
    return coast


def project_coastline(coast, crs=CRS, report=None):
    """Reproject the polygons to ``crs`` (EPSG:32630, metres)."""
    out = coast.to_crs(crs)
    if report is not None:
        report.add("coastline", "project_coastline", f"to EPSG:{crs}",
                   len(coast), len(out))
    return out


def clip_coastline(coast, area=AREA, report=None):
    """Clip the polygons (in a projected CRS) to ``area``'s outline; drop empties."""
    outline = _outline(area, coast.crs)
    coast = coast[coast.intersects(outline)].copy()
    n = len(coast)
    coast["geometry"] = coast.geometry.make_valid().intersection(outline)
    out = coast[~coast.geometry.is_empty & coast.geometry.notna()]
    if report is not None:
        report.add("coastline", "clip_coastline", "clipped to the area", n, len(out))
    return out.reset_index(drop=True)


# ---- the whole archive ------------------------------------------------------------------------

class _ParquetSink:
    """Write DataFrames to one Parquet file, batch by batch, via a temporary name."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp = path.with_name(path.name + ".part")
        self.writer = None

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

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _to_parquet(df, path: Path) -> None:
    sink = _ParquetSink(path)
    sink.write(df)
    sink.close()


def _run_ais(raw: Path, out: Path, window, area, report, batch_rows, max_speed,
             exclude=EXCLUDED_PERIODS, n_jobs=1):
    folder = raw / "P1_AIS_Data"
    ais = read_ais_raw(folder / "nari_dynamic.csv", report)
    mmsi_check(ais).to_csv(out.with_name("mmsi_check.csv"), index=False)
    ais = drop_invalid_mmsi(ais, report)
    ais = drop_invalid_time(ais, report)
    ais = drop_excluded_periods(ais, exclude, report)
    ais = drop_invalid_position(ais, area, report)
    ais = unavailable_to_nan(ais, report)
    ais = check_ranges(ais, report)
    ais = drop_duplicates_and_sort(ais, report)
    ais = drop_position_jumps(ais, max_speed, report)
    static = folder / "nari_static.csv"
    types = raw / "P1_AIS_Codes" / "Ship Types List.csv"
    if static.exists() and types.exists():
        ais = attach_vessel_types(ais, static, types, report)
    sink, n_out = _ParquetSink(out), 0
    try:
        for part in iter_denoised_ais(ais, window, batch_rows, n_jobs):
            sink.write(part)
            n_out += len(part)
    except BaseException:
        sink.abort()
        raise
    sink.close()
    report.add("AIS", "denoise_ais", f"median filter, {window:g} s window; x, y",
               len(ais), n_out)
    return n_out


def _run_weather(raw: Path, out: Path, window, report, exclude=EXCLUDED_PERIODS):
    obs, stations, wind = read_weather_raw(raw / "E2_Weather_Conditions", report)
    n_in = len(obs)
    obs = weather_unavailable_to_nan(obs, report)
    obs = merge_weather_stations(obs, stations, report)
    obs = drop_invalid_weather(obs, report)
    obs = drop_excluded_periods(obs, exclude, report, dataset="weather")
    obs = check_weather_ranges(obs, wind, report)
    obs = denoise_weather(obs, window, report)
    _to_parquet(obs, out)
    return n_in, len(obs)


def ocean_month_matches(oc, month) -> bool:
    """True when most of ``oc``'s times fall in ``month`` (a name of :data:`MONTHS`).

    The archive's ``oc_january.csv`` is a byte-for-byte copy of
    ``oc_november.csv`` (times 2015-11-01 to 2015-11-30): there is no January
    sea state.  A file whose times are not in its month is skipped, not written
    under the wrong name.
    """
    if len(oc) == 0:
        return False
    stamp = pd.to_datetime(oc["unix_time"].median(), unit="s")
    return stamp.strftime("%B").lower() == month


def _run_ocean(src: Path, out: Path, window, area, report, month=None,
               exclude=EXCLUDED_PERIODS):
    oc = read_ocean_raw(src, report)
    n_in = len(oc)
    if month is not None and not ocean_month_matches(oc, month):
        if report is not None:
            report.add("ocean", "ocean_month_matches",
                       f"oc_{month}.csv: times not in {month}, file skipped",
                       n_in, 0)
        return None
    oc = drop_invalid_ocean(oc, area, report)
    oc = drop_excluded_periods(oc, exclude, report, dataset="ocean")
    oc = drop_land_cells(oc, report)
    oc = ocean_unavailable_to_nan(oc, report)
    oc = check_ocean_ranges(oc, report)
    oc = sort_ocean(oc, report)
    oc = denoise_ocean(oc, window, report)
    _to_parquet(oc, out)
    return n_in, len(oc)


def _run_coastline(src: Path, out: Path, area, report):
    coast = read_coastline(src, area, report)
    n_in = len(coast)
    coast = project_coastline(coast, CRS, report)
    if area is not None:
        coast = clip_coastline(coast, area, report)
    coast.to_parquet(out, compression="zstd")
    return n_in, len(coast)


def preprocess_archive(raw_root, out_root, window=30.0, *, force=False, area=AREA,
                       months=MONTHS, batch_rows=BATCH_ROWS, max_speed_knots=50,
                       exclude=EXCLUDED_PERIODS, n_jobs=1, log=print):
    """Preprocess every dataset found under ``raw_root``; write to ``out_root``.

    ``raw_root`` is the archive (``data/zenodo_1167595``) or a sample cut from it
    (``data/sample/raw``). Writes ``AIS/ais.parquet`` (+ ``AIS/mmsi_check.csv``),
    ``Weather/weather.parquet``, ``Ocean/ocean_<month>.parquet``,
    ``Coastline/coastline.parquet`` and ``preprocessing_report.csv`` / ``.txt``.
    A dataset whose raw files are absent is skipped, and so is one whose output
    exists, unless ``force``.

    Parameters
    ----------
    window : float
        Width of the median filter, in seconds (30, the package default).
    force : bool
        Rebuild outputs that exist.
    area : tuple or None
        ``(lon_min, lon_max, lat_min, lat_max)`` the data must lie in, and the
        coastline is clipped to; ``None`` keeps everything.
    months : sequence of str
        The ``oc_<month>.csv`` files to look for.
    batch_rows : int
        AIS rows filtered at once (whole vessels), to bound memory.
    max_speed_knots : float or None
        Isolated AIS position jumps above this speed are dropped (default 50;
        ``None`` disables the rule).
    exclude : sequence of (first, last, reason) or None
        Periods dropped from every dataset (default :data:`EXCLUDED_PERIODS`,
        January 2016); ``()`` keeps everything.
    n_jobs : int or None
        Processes to use (scikit-learn convention: ``1`` or ``None`` serial in
        this process, ``-1`` every core). Weather, each month of sea state and
        the coastline are independent and run side by side; then the AIS
        median filter runs per batch of whole vessels on that many processes.
        The files and the report are identical to a serial run's. Memory: each
        sea-state month in flight holds its table (about 1-2 GB on the full
        archive) and each AIS worker one batch (``batch_rows`` rows).
    log : callable or None
        Called with each progress line; ``None`` is silent.

    Returns
    -------
    PreprocessingReport
        Rows in and out of every step; also written beside the outputs
        (nothing is written if every dataset was skipped).
    """
    import time

    raw_root, out_root = Path(raw_root), Path(out_root)
    if not window > 0:
        raise ValueError("window must be positive")
    say = log if log is not None else (lambda *_: None)
    ais = raw_root / "P1_AIS_Data" / "nari_dynamic.csv"
    wdir = raw_root / "E2_Weather_Conditions"
    shp = raw_root / "C2_European_Coastline" / "Europe Coastline (Polygone).shp"
    # (label, raw files needed, target, kind, keyword arguments of the run)
    jobs = [("AIS", [ais], out_root / "AIS" / "ais.parquet", "ais",
             dict(raw=raw_root, window=window, area=area, batch_rows=batch_rows,
                  max_speed=max_speed_knots, exclude=exclude, n_jobs=n_jobs)),
            ("weather", [wdir / "table_wheatherObservation.csv",
                         wdir / "table_weatherStation.csv"],
             out_root / "Weather" / "weather.parquet", "weather",
             dict(raw=raw_root, window=window, exclude=exclude))]
    for month in months:
        src = raw_root / "E1_ocean" / f"oc_{month}.csv"
        jobs.append((f"ocean {month}", [src],
                     out_root / "Ocean" / f"ocean_{month}.parquet", "ocean",
                     dict(src=src, window=window, area=area, month=month,
                          exclude=exclude)))
    jobs.append(("coastline", [shp], out_root / "Coastline" / "coastline.parquet",
                 "coastline", dict(src=shp, area=area)))

    todo = []
    for job in jobs:
        label, needs, target = job[:3]
        missing = [p for p in needs if not p.exists()]
        if missing:
            say(f"skip {label}: raw file absent ({missing[0]})")
        elif target.exists() and not force:
            say(f"skip {label}: {target} exists (force=True to rebuild)")
        else:
            todo.append(job)

    def finish(job, outcome):
        label, target = job[0], job[2]
        reports[label] = outcome[2]
        result, seconds = outcome[:2]
        if result is None:
            say(f"  {label}: skipped, its times are not in its month "
                "(the archive's oc_january.csv repeats November)")
            return 0
        n = result if isinstance(result, int) else result[1]
        say(f"  {label}: {n:,} rows -> {target} ({seconds:.1f} s)")
        return 1

    reports, done = {}, 0
    if resolve_n_jobs(n_jobs, len(todo)) == 1:
        for job in todo:
            say(f"{job[0]} ...")
            done += finish(job, _execute(job[3], job[2], job[4]))
    else:
        # Phase 1: the small datasets side by side. Phase 2: the AIS, whose
        # batches use the processes themselves (no pool inside a pool).
        small = [j for j in todo if j[3] != "ais"]
        for job in small:
            say(f"{job[0]} ... (in parallel)")
        outcomes = run_parallel(_execute, [(j[3], j[2], j[4]) for j in small],
                                n_jobs=n_jobs)
        for job, outcome in zip(small, outcomes):
            done += finish(job, outcome)
        for job in todo:
            if job[3] == "ais":
                say(f"{job[0]} ... ({resolve_n_jobs(n_jobs)} processes)")
                done += finish(job, _execute(job[3], job[2], job[4]))

    report = PreprocessingReport()
    for job in todo:                     # in the serial order, whatever ran first
        report.extend(reports[job[0]])
    if done:
        out_root.mkdir(parents=True, exist_ok=True)
        frame = report.to_frame()
        frame.to_csv(out_root / "preprocessing_report.csv", index=False)
        (out_root / "preprocessing_report.txt").write_text(
            report.to_text() + "\n", encoding="utf-8")
    return report


def _execute(kind, target, kw):
    """Run one dataset's preprocessing: ``(result, seconds, its PreprocessingReport)``.

    Module-level, with plain arguments, so that a worker process can run it.
    """
    import time

    report = PreprocessingReport()
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if kind == "ais":
        result = _run_ais(kw["raw"], target, kw["window"], kw["area"], report,
                          kw["batch_rows"], kw["max_speed"], kw["exclude"],
                          kw["n_jobs"])
    elif kind == "weather":
        result = _run_weather(kw["raw"], target, kw["window"], report, kw["exclude"])
    elif kind == "ocean":
        result = _run_ocean(kw["src"], target, kw["window"], kw["area"], report,
                            kw["month"], kw["exclude"])
    else:
        result = _run_coastline(kw["src"], target, kw["area"], report)
    return result, time.perf_counter() - t0, report
