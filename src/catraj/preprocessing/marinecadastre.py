"""Preprocessing of the US MarineCadastre AIS data (2017), one step per function.

The 2021 paper ([Mehri2021] section III-B) used the AIS broadcasts of the US
Coast Guard that MarineCadastre.gov publishes, one CSV per day, for the "eastern
coast" of the United States in November and December 2017, together with the
NOAA Medium Resolution Shoreline as the barrier of the compression.  This module
turns those files into the tables the notebooks read.  Like :mod:`.archive`, it
is a list of public steps, each taking a table and returning one, each recording
rows in and out and the values set to NaN into a :class:`PreprocessingReport`;
:func:`preprocess_us` runs them all and writes the files.

Raw data (``ais-2017-MM-DD.csv.zst``)
-------------------------------------

Columns used: ``mmsi``, ``base_date_time`` ("2017-11-01 00:00:00", UTC),
``longitude``, ``latitude``, ``sog``, ``cog``, ``heading``, ``status``,
``vessel_type``, ``length``, ``width``.  Names, call signs and IMO numbers are
not read.  The broadcasts are already thinned to about one per minute (median
interval 70 s), as the paper says.

Study area (the author's decision, 2026-11-25)
----------------------------------------------

The east coast: the box lon -82..-60, lat 23..46, **minus** the Gulf side of
Florida (lon -82..-80.9, lat 25.2..31) **minus** the Great Lakes (lon -82..-75,
lat 41.3..46) -- exactly that, nothing more (:data:`AREA_BOXES`,
:func:`study_area_polygon`).  The clip is vectorised bounding-box tests
(:func:`in_study_area`), not a polygon test per point; the tests check it against
the polygon.  The part of the area north of 44.5 N and west of 73 W (the St.
Lawrence river and Lake Champlain) is **kept**; its positions are counted in the
report ("informational" line) and nothing more.  Against the paper's counts (31.0
M positions in November, 27.5 M in December for its "eastern coast") this area
holds about twice as many: the paper's own polygon is not given, and the author
knows.  The counts per month are in the report.

Rules (per daily file, then per month)
--------------------------------------

====================  ==========================================  =============
rule                  keep                                        source
====================  ==========================================  =============
area                  inside the study area above                 the author
``mmsi``              a ship station: nine digits, an allocated   ITU-R M.585,
                      MID                                         :mod:`.mmsi`
``unix_time``         2017-11-01 00:00:00 to 2017-12-31 23:59:59  paper, III-B
                      UTC
``lon``, ``lat``     -180..180 and -90..90                        AIS standard
``speed_over_ground`` 0 <= v < 102.3 knots (102.3: n/a)           ITU-R M.1371
``course_over_ground``0 <= v < 360 degrees (360: n/a)             ITU-R M.1371
``true_heading``      0 <= v < 360 degrees (511: n/a)             ITU-R M.1371
``navigational_status`` the integers 0..15                        ITU-R M.1371
``length``, ``width`` 0 < length <= 1022 m, 0 < width <= 126 m    AIS fields
====================  ==========================================  =============

The paper removed the MMSIs that are not nine digits; the rule here is stricter
(the MID must also be allocated, and coast stations, aids to navigation,
placeholders, SAR aircraft and the like go), the same rule as for the Zenodo
archive.  A row without a ship MMSI, time or position is dropped; an
out-of-range value becomes NaN.  In the 2017 files the "not available" codes
were already NaN (the maxima found are 96.6 knots, 359.9 and 359 degrees, status
14); the rules stay as a guard, and the report counts what they find.  Then
duplicate (mmsi, unix_time) rows are dropped and the rows sorted by (mmsi,
unix_time), and isolated position jumps are dropped
(:func:`~catraj.preprocessing.archive.drop_position_jumps`, 50 knots).

**Months.**  The days of one month are read in parallel (``n_jobs``) and then
concatenated, so that duplicates, jumps and the median filter see every vessel's
whole month, across midnight.  The months are processed one after the other
(memory: a month is about 68 million positions); a trip that spans the night of
November 30 to December 1 is cut there, because the two months are separate
files and separate filters, and a vessel's type is chosen per month.

Vessel type and group
---------------------

``vessel_type`` is the AIS ship-type code (0..99, ITU-R M.1371) or one of the
AVIS service codes 1001..1025 that MarineCadastre adds for vessels that
broadcast no type; both are in the official table "AIS Vessel Type and Group
Codes used by the Marine Cadastre Project" (2018-05-23,
coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf; the 2017 files
use the same codes: 1001..1024 are present in them).  Per vessel the most
frequent non-null code of the month is taken (the smallest on a tie).  The
names (:data:`VESSEL_TYPE_NAMES`) follow the Zenodo names where there is one
(Cargo, Tanker, Passenger, Fishing, Tug, Sailing Vessel, Pleasure Craft, SAR,
Pilot Vessel, ...); the AVIS services keep their own names (Freight Barge,
Industrial Vessel, ...).  ``vessel_group`` is the paper's grouping of Fig. 2:
Tanker, Tug, Cargo, Sailing, Passenger, Fishing, Military, Other, taken from
the official table's group column -- Pleasure Craft/Sailing is "Sailing" and Tug
Tow is "Tug", "Not Available" and every unlisted code are "Other" -- and a
vessel without a code is "Other" (decision C12 of :mod:`catraj.prediction.clstm`).
The table has two oddities that are followed as published: the WIG codes 21
and 22 and the tank barge 1023 are in group Tug Tow.  ``length`` and ``width``
are likewise the most frequent non-null value per vessel.  Both name columns are
``category`` columns with the same categories in every file.

Projection and grid north
-------------------------

One CRS for the whole area: EPSG:32618, UTM zone 18N (metres).  The area spans
zones 17 to 19, so a position far from the central meridian (75 W) is in a
distorted but still metric system, and **grid north is not true north**: it
differs by the meridian convergence, up to about 10 degrees at the eastern edge.
The curvilinear model of the CLSTM (decision C8 of
:mod:`catraj.prediction.clstm`) takes grid north as north, so the course the
network is given must be the course on the map.  :func:`grid_bearing` converts a
true bearing to the grid bearing (``true - convergence``, the convergence from
pyproj); the output keeps ``course_over_ground`` and ``true_heading`` (true
north, as broadcast) and adds ``course_over_ground_grid`` and
``true_heading_grid``, computed from the filtered course at the filtered
position.

Noise
-----

A centred median over ``window`` seconds (30 by default,
:func:`catraj.preprocessing.rolling_filter`, the engine of ``denoise``) per
vessel of the projected position, speed, course (on the circle) and heading (on
the circle); ``lon`` and ``lat`` are recomputed from the filtered position, as
for the archive.  The filter runs on batches of whole vessels, so ``n_jobs``
changes nothing in the files.

Shoreline
---------

NOAA Medium Resolution Shoreline (``us_medium_shoreline.shp``, EPSG:4269,
LineStrings).  The regions N (New England), M (Mid-Atlantic), S (South
Atlantic), GS and SG (Florida Keys) and MN and NM (border) are kept, clipped to
the study area buffered by 0.5 degrees, projected to EPSG:32618 and written as
GeoParquet; the lines are valid barriers for :class:`catraj.compression.Barriers`.

Output of :func:`preprocess_us`
-------------------------------

``AIS/ais_2017-11.parquet`` and ``AIS/ais_2017-12.parquet`` with the columns
``mmsi, unix_time, x, y, lon, lat, speed_over_ground, course_over_ground,
true_heading, course_over_ground_grid, true_heading_grid, navigational_status,
ship_type_code, vessel_type, vessel_group, length, width``;
``Coastline/shoreline.parquet``; ``preprocessing_report.csv`` and ``.txt``.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .._parallel import resolve_n_jobs, run_parallel
from . import archive
from .archive import PreprocessingReport, _ParquetSink, _to_nan, _drop, _vessel_batches
from .filtering import rolling_filter

__all__ = [
    "US_CRS", "CRS", "US_PERIOD", "PERIOD", "AREA_BOXES", "AREA", "StudyArea",
    "SHORELINE_REGIONS", "VESSEL_TYPE_NAMES", "VESSEL_GROUPS", "study_area_polygon",
    "in_study_area", "grid_bearing", "vessel_type_name", "vessel_group_of",
    "map_vessel_types", "most_frequent",
    # the same step names as catraj.preprocessing.archive
    "read_ais_raw", "drop_invalid_mmsi", "drop_invalid_time", "drop_invalid_position",
    "unavailable_to_nan", "check_ranges", "drop_duplicates_and_sort",
    "drop_position_jumps", "attach_vessel_types", "project", "iter_denoised_ais",
    "denoise_ais", "read_coastline", "project_coastline", "clip_coastline",
    "mmsi_check", "preprocess_us",
]

US_CRS = CRS = 32618                # UTM zone 18N, metres
US_PERIOD = (                       # unix seconds, both ends included
    int(pd.Timestamp("2017-11-01 00:00:00", tz="UTC").timestamp()),
    int(pd.Timestamp("2017-12-31 23:59:59", tz="UTC").timestamp()),
)
PERIOD = US_PERIOD
#: ``(lon_min, lon_max, lat_min, lat_max)``: the box, and the two boxes cut out of it.
AREA_BOXES = {"outer": (-82.0, -60.0, 23.0, 46.0),
              "gulf_florida": (-82.0, -80.9, 25.2, 31.0),
              "great_lakes": (-82.0, -75.0, 41.3, 46.0)}
SHORELINE_REGIONS = ("N", "M", "S", "GS", "SG", "MN", "NM")
SHORELINE_MARGIN = 0.5              # degrees
ST_LAWRENCE = (44.5, -73.0)         # lat above and lon below which positions are counted
BATCH_ROWS = 1_500_000              # rows filtered at once (whole vessels)
CHUNK_ROWS = 2_000_000              # rows of a daily file parsed at once

SPEED_NA, ANGLE_NA = 102.3, 360.0
STATUS_CODES = tuple(range(16))
LENGTH_MAX, WIDTH_MAX = 1022.0, 126.0
FILTERED = ["speed_over_ground", "course_over_ground", "true_heading"]
CIRCULAR = ["course_over_ground", "true_heading"]
POINT_GAP = 1e10

RAW_COLUMNS = {"mmsi": "mmsi", "base_date_time": "unix_time", "longitude": "lon",
               "latitude": "lat", "sog": "speed_over_ground", "cog": "course_over_ground",
               "heading": "true_heading", "status": "navigational_status",
               "vessel_type": "ship_type_code", "length": "length", "width": "width"}
OUTPUT_COLUMNS = [
    "mmsi", "unix_time", "x", "y", "lon", "lat", "speed_over_ground",
    "course_over_ground", "true_heading", "course_over_ground_grid",
    "true_heading_grid", "navigational_status", "ship_type_code", "vessel_type",
    "vessel_group", "length", "width"]


# ---- vessel types (VesselTypeCodes2018.pdf of MarineCadastre) ------------------------------

def _build_types():
    names, groups = {0: "Unknown"}, {0: "Other"}
    for c in range(1, 20):
        names[c], groups[c] = "Other", "Other"
    for c in range(20, 30):
        names[c], groups[c] = "Wing in Ground", ("Tug" if c in (21, 22) else "Other")
    single = {30: ("Fishing", "Fishing"), 31: ("Towing", "Tug"), 32: ("Towing", "Tug"),
              33: ("Dredger", "Other"), 34: ("Diving Ops", "Other"),
              35: ("Military Ops", "Military"), 36: ("Sailing Vessel", "Sailing"),
              37: ("Pleasure Craft", "Sailing"), 38: ("Other", "Other"),
              39: ("Other", "Other"), 50: ("Pilot Vessel", "Other"), 51: ("SAR", "Other"),
              52: ("Tug", "Tug"), 53: ("Port Tender", "Other"),
              54: ("Anti-Pollution", "Other"), 55: ("Law Enforcement", "Other"),
              56: ("Other", "Other"), 57: ("Other", "Other"),
              58: ("Medical Transport", "Other"), 59: ("Other", "Other")}
    for c, (n, g) in single.items():
        names[c], groups[c] = n, g
    for c in range(40, 50):
        names[c], groups[c] = "High-Speed Craft", "Other"
    for lo, n, g in ((60, "Passenger", "Passenger"), (70, "Cargo", "Cargo"),
                     (80, "Tanker", "Tanker"), (90, "Other", "Other")):
        for c in range(lo, lo + 10):
            names[c], groups[c] = n, g
    avis = {1001: ("Fishing", "Fishing"), 1002: ("Fish Processing", "Fishing"),
            1003: ("Freight Barge", "Cargo"), 1004: ("Cargo", "Cargo"),
            1005: ("Industrial Vessel", "Other"), 1006: ("Miscellaneous Vessel", "Other"),
            1007: ("Mobile Offshore Drilling Unit", "Other"),
            1008: ("Non-vessel", "Other"), 1009: ("Non-vessel", "Other"),
            1010: ("Offshore Supply Vessel", "Other"), 1011: ("Oil Recovery", "Other"),
            1012: ("Passenger", "Passenger"),
            1013: ("Passenger (Uninspected)", "Passenger"),
            1014: ("Passenger Barge", "Passenger"),
            1015: ("Passenger Barge (Uninspected)", "Passenger"),
            1016: ("Public Freight", "Cargo"), 1017: ("Public Tanker", "Tanker"),
            1018: ("Public Vessel", "Other"), 1019: ("Pleasure Craft", "Sailing"),
            1020: ("Research Vessel", "Other"), 1021: ("SAR Aircraft", "Military"),
            1022: ("School Ship", "Other"), 1023: ("Tank Barge", "Tug"),
            1024: ("Tanker", "Tanker"), 1025: ("Tug", "Tug")}
    for c, (n, g) in avis.items():
        names[c], groups[c] = n, g
    return names, groups


#: ship-type code -> vessel_type name, and -> vessel_group; codes not listed
#: (100..999, 1000, above 1025) are "Other" / "Other"; no code is "Unknown" / "Other".
VESSEL_TYPE_NAMES, _GROUP_OF_CODE = _build_types()
VESSEL_GROUPS = ("Tanker", "Tug", "Cargo", "Sailing", "Passenger", "Fishing",
                 "Military", "Other")
_TYPE_CATEGORIES = sorted(set(VESSEL_TYPE_NAMES.values()) | {"Other", "Unknown"})


def vessel_type_name(code) -> str:
    """The ``vessel_type`` name of one ship-type code (``None`` or NaN: "Unknown")."""
    if code is None or code != code:
        return "Unknown"
    return VESSEL_TYPE_NAMES.get(int(code), "Other")


def vessel_group_of(code) -> str:
    """The paper's group of one ship-type code; no code or an unlisted one: "Other"."""
    if code is None or code != code:
        return "Other"
    return _GROUP_OF_CODE.get(int(code), "Other")


def map_vessel_types(code, report=None, dataset="AIS") -> pd.DataFrame:
    """``vessel_type`` and ``vessel_group`` (categories) of an array of codes.

    Returns a frame aligned with ``code`` (a null code gives "Unknown" and
    "Other"); both columns have the same categories whatever the codes, so
    frames can be written to one Parquet file batch by batch.
    """
    code = pd.Series(code).reset_index(drop=True)
    values, inverse = np.unique(code.to_numpy(dtype=float), return_inverse=True)
    names = [vessel_type_name(v) for v in values]
    groups = [vessel_group_of(v) for v in values]
    type_cat = pd.CategoricalDtype(_TYPE_CATEGORIES)
    group_cat = pd.CategoricalDtype(list(VESSEL_GROUPS))
    t_codes = np.array([type_cat.categories.get_loc(n) for n in names], dtype=np.int8)
    g_codes = np.array([group_cat.categories.get_loc(g) for g in groups], dtype=np.int8)
    out = pd.DataFrame({
        "vessel_type": pd.Categorical.from_codes(t_codes[inverse], dtype=type_cat),
        "vessel_group": pd.Categorical.from_codes(g_codes[inverse], dtype=group_cat)})
    if report is not None:
        report.add(dataset, "map_vessel_types", "Unknown / Other: no ship-type code",
                   len(code), len(code), int(code.isna().sum()))
    return out


def most_frequent(keys, values) -> pd.Series:
    """The most frequent non-null ``value`` per key (the smallest on a tie).

    Returns a Series indexed by the keys that have at least one value.
    """
    df = pd.DataFrame({"k": np.asarray(keys), "v": np.asarray(values, dtype=float)}).dropna()
    counts = df.groupby(["k", "v"]).size().rename("n").reset_index()
    return _mode_of_counts(counts)


def _mode_of_counts(counts) -> pd.Series:
    """``most_frequent`` from a (k, v, n) table, which may hold a pair several times."""
    counts = counts.groupby(["k", "v"], as_index=False)["n"].sum()
    counts = counts.sort_values(["k", "n", "v"], ascending=[True, False, True])
    return counts.drop_duplicates("k").set_index("k")["v"]


# ---- study area ----------------------------------------------------------------------------

def study_area_polygon():
    """The study area as a shapely polygon in lon/lat (box minus the two boxes)."""
    from shapely.geometry import box

    def b(name):
        lo, hi, la0, la1 = AREA_BOXES[name]
        return box(lo, la0, hi, la1)

    return b("outer").difference(b("gulf_florida")).difference(b("great_lakes"))


class StudyArea:
    """The study area: a callable ``area(lon, lat) -> bool array`` and its polygon.

    Vectorised box tests, no shapely per point.  A position exactly on the
    area's boundary (an edge of the outer box or of a box cut out of it) is
    outside, as for ``shapely``'s ``polygon.contains``; that is how the author's
    count of 2,264,758 positions on 2017-11-01 was made.  ``polygon`` is
    :func:`study_area_polygon`, ``bounds`` its ``(lon_min, lon_max, lat_min,
    lat_max)`` bounding box.
    """

    bounds = AREA_BOXES["outer"]

    @property
    def polygon(self):
        return study_area_polygon()

    def __call__(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        lo, hi, la0, la1 = AREA_BOXES["outer"]
        inside = (lon > lo) & (lon < hi) & (lat > la0) & (lat < la1)
        _, g_hi, g0, g1 = AREA_BOXES["gulf_florida"]
        inside &= ~((lon <= g_hi) & (lat >= g0) & (lat <= g1))
        _, l_hi, l0, _ = AREA_BOXES["great_lakes"]
        inside &= ~((lon <= l_hi) & (lat >= l0))
        return inside

    def __repr__(self):
        return "StudyArea(US east coast: box -82..-60, 23..46 minus Gulf Florida, Great Lakes)"


AREA = StudyArea()


def in_study_area(lon, lat) -> np.ndarray:
    """Whether each position is inside the study area (:class:`StudyArea`)."""
    return AREA(lon, lat)


# ---- grid north ----------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _proj(crs):
    from pyproj import Proj

    return Proj(crs)


def grid_bearing(lon, lat, bearing_deg, crs=US_CRS):
    """The bearing on the map (grid north) of a true bearing at ``lon``, ``lat``.

    ``true - convergence``, in ``[0, 360)``, where the meridian convergence is
    the angle between grid north and true north at the position (pyproj,
    positive east of the central meridian in the northern hemisphere).  NaN
    stays NaN.  Arrays (any broadcastable shapes) or scalars; the CRS is
    anything pyproj reads (default EPSG:32618).
    """
    lon, lat, bearing = np.broadcast_arrays(*(np.asarray(v, dtype=float)
                                              for v in (lon, lat, bearing_deg)))
    gamma = np.asarray(_proj(crs).get_factors(lon, lat).meridian_convergence, dtype=float)
    out = np.mod(bearing - gamma, 360.0)
    out = np.where(out >= 360.0, 0.0, out)
    return out if out.ndim else float(out)


# ---- reading and cleaning one daily file --------------------------------------------------

def _unix_seconds(text) -> np.ndarray:
    """Unix seconds of "YYYY-MM-DD HH:MM:SS" strings; int64 min where unparseable."""
    stamp = pd.to_datetime(pd.Series(text), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return stamp.to_numpy().astype("datetime64[s]").astype("int64")   # NaT: int64 min


def _read_one(path, area, chunk_rows):
    """``(frame, rows read, rows in the St. Lawrence part)`` of one daily file."""
    path = Path(path)
    f32 = "float32"
    reader = pd.read_csv(
        path, usecols=list(RAW_COLUMNS),
        dtype={"longitude": "float64", "latitude": "float64", "sog": f32, "cog": f32,
               "heading": f32, "status": f32, "vessel_type": f32, "length": f32,
               "width": f32},
        compression="zstd" if path.name.endswith(".zst") else "infer",
        chunksize=chunk_rows)
    parts, n_read, n_st = [], 0, 0
    for chunk in reader:
        n_read += len(chunk)
        sub = chunk if area is None else chunk[area(chunk["longitude"], chunk["latitude"])]
        n_st += int(((sub["latitude"] > ST_LAWRENCE[0])
                     & (sub["longitude"] < ST_LAWRENCE[1])).sum())
        sub = sub.rename(columns=RAW_COLUMNS)
        sub["unix_time"] = _unix_seconds(sub["unix_time"])
        if sub["mmsi"].dtype.kind != "i":
            sub = sub[sub["mmsi"].notna()]
        sub["mmsi"] = sub["mmsi"].astype("int64")
        parts.append(sub)
    cols = list(RAW_COLUMNS.values())
    df = (pd.concat(parts, ignore_index=True) if parts else
          pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}))[cols]
    df.insert(df.columns.get_loc("true_heading") + 1, "rate_of_turn", np.float32(np.nan))
    return df, n_read, n_st


def read_ais_raw(path, report=None, area=AREA, chunk_rows=CHUNK_ROWS) -> pd.DataFrame:
    """Read a daily file (or every ``ais-*.csv*`` of a folder) with the Zenodo names.

    Columns: ``mmsi``, ``unix_time`` (int64 seconds), ``lon``, ``lat`` (float64),
    ``speed_over_ground``, ``course_over_ground``, ``true_heading``,
    ``rate_of_turn`` (all NaN: MarineCadastre has none), ``navigational_status``,
    ``ship_type_code``, ``length``, ``width`` (float32).

    The file is parsed ``chunk_rows`` rows at a time and **clipped to** ``area``
    (default the study area) at once, because a day holds ~8 million positions of
    all US waters; ``area=None`` keeps them all.  The report gets the rows read,
    the rows inside, and, for information only (nothing is dropped), how many of
    those lie in the St. Lawrence / Lake Champlain part (lat > 44.5, lon < -73).
    """
    path = Path(path)
    files = sorted(path.glob("ais-*.csv*")) if path.is_dir() else [path]
    parts = [_read_one(f, area, chunk_rows) for f in files]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    n_read, n_st = sum(p[1] for p in parts), sum(p[2] for p in parts)
    if report is not None:
        report.add("AIS", "read_ais_raw", "rows read", n_read, n_read)
        if area is not None:
            report.add("AIS", "read_ais_raw", "position inside the study area",
                       n_read, len(df))
        report.add("AIS", "read_ais_raw",
                   "information only, kept: of these, lat > 44.5 and lon < -73 "
                   "(St. Lawrence, Lake Champlain)", n_st, n_st)
    return df


def mmsi_check(ais) -> pd.DataFrame:
    """The ITU-R M.585 classes found (as for the archive)."""
    return archive.mmsi_check(ais)


#: Same rule as for the archive: a ship station, nine digits, an allocated MID
#: (stricter than the paper's "nine digits").
drop_invalid_mmsi = archive.drop_invalid_mmsi


def drop_invalid_time(df, report=None, dataset="AIS", column="unix_time",
                      period=US_PERIOD) -> pd.DataFrame:
    """Drop rows whose time is missing or outside 2017-11-01..2017-12-31 (UTC).

    :func:`archive.drop_invalid_time` with the study's period as the default.
    """
    return archive.drop_invalid_time(df, report, dataset, column, period)


def drop_invalid_position(ais, area=AREA, report=None, dataset="AIS",
                          lon="lon", lat="lat") -> pd.DataFrame:
    """Drop rows without a position: -180..180 / -90..90, then inside ``area``.

    ``area`` is a :class:`StudyArea` (or any ``area(lon, lat) -> mask`` callable),
    or the archive's ``(lon_min, lon_max, lat_min, lat_max)`` tuple; ``None``
    skips the second rule.
    """
    step = "drop_invalid_position"
    ais = _drop(ais, ais[lon].between(-180, 180) & ais[lat].between(-90, 90),
                report, dataset, step, "lon, lat in valid range")
    if area is not None:
        if callable(area):
            inside = area(ais[lon].to_numpy(), ais[lat].to_numpy())
        else:
            inside = (ais[lon].between(area[0], area[1])
                      & ais[lat].between(area[2], area[3])).to_numpy()
        ais = _drop(ais, inside, report, dataset, step, "position inside study area")
    return ais


#: The archive's rule: speed >= 102.3, course and heading >= 360 become NaN
#: (rate_of_turn is all NaN here).  The 2017 files hold none; a guard.
unavailable_to_nan = archive.unavailable_to_nan


def check_ranges(ais, report=None) -> pd.DataFrame:
    """The archive's range rules, plus ``length`` and ``width``.

    Negative speed, course or heading outside 0..360, a status that is not an
    integer 0..15 (:func:`archive.check_ranges`), and a length or width that is
    not positive or exceeds the AIS field (1022 m, 126 m): NaN.
    """
    ais = archive.check_ranges(ais, report)
    step = "check_ranges"
    for col, high in (("length", LENGTH_MAX), ("width", WIDTH_MAX)):
        if col in ais.columns:
            _to_nan(ais, col, ~((ais[col] > 0) & (ais[col] <= high)), report, "AIS",
                    step, f"{col} outside 0..{high:g}")
    return ais


#: Repeated (mmsi, unix_time) rows dropped, rows sorted; and isolated position
#: jumps dropped: the archive's functions, as they are.
drop_duplicates_and_sort = archive.drop_duplicates_and_sort
drop_position_jumps = archive.drop_position_jumps


def project(df, lon="lon", lat="lat", crs=US_CRS) -> pd.DataFrame:
    """Add ``x`` and ``y`` in EPSG:32618 (:func:`archive.project`)."""
    return archive.project(df, lon, lat, crs)


def _counts(ais, column):
    """(k, v, n) counts of a vessel attribute, to be merged over days."""
    d = ais[["mmsi", column]].dropna()
    c = d.groupby(["mmsi", column]).size().rename("n").reset_index()
    return c.rename(columns={"mmsi": "k", column: "v"})


_VESSEL_ATTRIBUTES = ("ship_type_code", "length", "width")


def _attach(ais, counts, report=None, dataset="AIS"):
    """Per-vessel modes of the code, length and width, and the type and group names."""
    ais = ais.copy()
    for c in _VESSEL_ATTRIBUTES:
        ais[c] = _vessel_values(ais, _mode_of_counts(counts[c]))
    ais = pd.concat([ais, map_vessel_types(ais["ship_type_code"])], axis=1)
    if report is not None:
        report.add(dataset, "attach_vessel_types",
                   "Unknown / Other: vessel without a ship-type code",
                   len(ais), len(ais), int(ais["ship_type_code"].isna().sum()))
    return ais


def attach_vessel_types(ais, report=None) -> pd.DataFrame:
    """Per vessel, the most frequent ship-type code, length and width; add the names.

    ``ship_type_code``, ``length`` and ``width`` become constant per vessel (NaN
    when it never sent one); ``vessel_type`` and ``vessel_group`` are added
    (categories, see the module docstring).  The codes are in the data itself, so
    unlike the archive's function there are no static-message files.
    """
    return _attach(ais, {c: _counts(ais, c) for c in _VESSEL_ATTRIBUTES}, report)


def _day_task(path, chunk_rows=CHUNK_ROWS):
    """One daily file, cleaned: ``(frame, vessel attribute counts, report)``."""
    report = PreprocessingReport()
    ais = read_ais_raw(path, report, AREA, chunk_rows)
    ais = drop_invalid_mmsi(ais, report)
    ais = drop_invalid_time(ais, report)
    ais = drop_invalid_position(ais, None, report)      # clipped by the reader
    ais = unavailable_to_nan(ais, report)
    ais = check_ranges(ais, report)
    counts = {c: _counts(ais, c) for c in _VESSEL_ATTRIBUTES}
    return ais.drop(columns="rate_of_turn"), counts, report


# ---- the month ---------------------------------------------------------------------------

def _sum_reports(reports, dataset) -> PreprocessingReport:
    """One report from many (daily, or per batch): equal (step, rule) lines are summed."""
    frames = [r.to_frame() for r in reports if len(r)]
    out = PreprocessingReport()
    if not frames:
        return out
    f = pd.concat(frames, ignore_index=True)
    f["_order"] = pd.factorize(f["step"])[0]            # steps in order of first appearance
    f = f.groupby(["_order", "step", "rule"], sort=False)[
        ["rows_in", "rows_out", "values_to_nan"]].sum().reset_index()
    f = f.sort_values("_order", kind="stable")
    for r in f.itertuples():
        out.add(dataset, r.step, r.rule, r.rows_in, r.rows_out, r.values_to_nan)
    return out


def _relabel(report, dataset):
    out = PreprocessingReport()
    for r in report.to_frame().itertuples():
        out.add(dataset, r.step, r.rule, r.rows_in, r.rows_out, r.values_to_nan)
    return out


def _denoise_batch(b, window=30.0, crs=US_CRS):
    """One batch of whole vessels: project, median-filter, grid bearings.

    What a worker process runs.  ``x`` and ``y`` are added if absent; the filtered
    ``x``, ``y``, speed, course and heading replace the raw ones, ``lon`` and
    ``lat`` are recomputed from them, and ``course_over_ground_grid`` and
    ``true_heading_grid`` follow from the filtered course and position.
    """
    from pyproj import Transformer

    b = project(b, crs=crs) if "x" not in b.columns else b.copy()
    codes = pd.factorize(b["mmsi"].to_numpy())[0]         # sorted: non-decreasing
    t = codes.astype(float) * POINT_GAP + b["unix_time"].to_numpy(dtype=float)
    values = np.column_stack([b["x"].to_numpy(), b["y"].to_numpy()]
                             + [b[c].to_numpy(dtype=float, na_value=np.nan)
                                for c in FILTERED])
    flags = [False, False] + [c in CIRCULAR for c in FILTERED]
    done = rolling_filter(t, values, window, circular=flags)
    b["x"], b["y"] = done[:, 0], done[:, 1]
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    b["lon"], b["lat"] = inv.transform(b["x"].to_numpy(), b["y"].to_numpy())
    for k, c in enumerate(FILTERED, start=2):
        v = done[:, k].astype("float32")
        if c in CIRCULAR:                        # float32 rounds 359.99997 up to 360
            v[v >= np.float32(360)] = 0
        b[c] = v
    lon, lat = b["lon"].to_numpy(), b["lat"].to_numpy()
    for c in CIRCULAR:
        b[f"{c}_grid"] = grid_bearing(lon, lat, b[c].to_numpy(dtype=float),
                                      crs).astype("float32")
    return b


def _jumps_and_denoise(b, window, max_speed):
    """Jumps, then :func:`_denoise_batch`: ``(frame, report)``; a month's worker."""
    report = PreprocessingReport()
    b = drop_position_jumps(b, max_speed, report)
    b = _denoise_batch(b, window)
    report.add("AIS", "denoise_ais", f"median filter, {window:g} s window; x, y; grid "
               "bearings", len(b), len(b))
    return b[OUTPUT_COLUMNS], report


def iter_denoised_ais(ais, window=30.0, batch_rows=BATCH_ROWS, n_jobs=1):
    """:func:`denoise_ais` in batches of whole vessels (``ais`` sorted by ``mmsi``).

    Yields one frame per batch, in order; ``n_jobs`` as for the archive.
    """
    batches = list(_vessel_batches(ais["mmsi"].to_numpy(), batch_rows))
    workers = resolve_n_jobs(n_jobs, len(batches))
    for first in range(0, len(batches), workers):
        wave = batches[first:first + workers]
        tasks = [(ais.iloc[a:b], window) for a, b in wave]
        yield from run_parallel(_denoise_batch, tasks, n_jobs=workers)


def denoise_ais(ais, window=30.0, report=None, n_jobs=1) -> pd.DataFrame:
    """Median-filter positions, speed, course and heading per vessel; grid bearings.

    As :func:`archive.denoise_ais`, projecting to EPSG:32618, and adding
    ``course_over_ground_grid`` and ``true_heading_grid``.
    """
    parts = list(iter_denoised_ais(ais, window, n_jobs=n_jobs))
    out = pd.concat(parts, ignore_index=True) if parts else ais.iloc[:0].copy()
    if report is not None:
        report.add("AIS", "denoise_ais", f"median filter, {window:g} s window",
                   len(ais), len(out))
    return out


def _vessel_values(ais, table) -> np.ndarray:
    """float32 value per row of the per-vessel Series ``table`` (NaN: none)."""
    idx = table.index.get_indexer(ais["mmsi"].to_numpy())
    vals = table.to_numpy(dtype="float32")
    out = np.full(len(ais), np.nan, dtype="float32")
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def _run_month(files, target, window, max_speed, batch_rows, n_jobs, say):
    """One month: ``(rows out, PreprocessingReport)``; writes ``target``."""
    label = f"AIS {Path(target).stem.removeprefix('ais_')}"
    t0 = time.perf_counter()
    results = run_parallel(_day_task, [(f,) for f in files], n_jobs=n_jobs)
    say(f"  {label}: {len(files)} days read in {time.perf_counter() - t0:.0f} s")
    reports = [r[2] for r in results]
    counts = {c: pd.concat([r[1][c] for r in results], ignore_index=True)
              for c in _VESSEL_ATTRIBUTES}
    ais = pd.concat([r[0] for r in results], ignore_index=True)
    del results
    report = _sum_reports(reports, label)

    local = PreprocessingReport()
    ais = drop_duplicates_and_sort(ais, local)
    ais = _attach(ais, counts, local)
    report.extend(_relabel(local, label))

    sink, n_out, batch_reports = _ParquetSink(Path(target)), 0, []
    batches = list(_vessel_batches(ais["mmsi"].to_numpy(), batch_rows))
    workers = resolve_n_jobs(n_jobs, len(batches))
    try:
        for first in range(0, len(batches), workers):
            wave = batches[first:first + workers]
            tasks = [(ais.iloc[a:b], window, max_speed) for a, b in wave]
            for part, rep in run_parallel(_jumps_and_denoise, tasks, n_jobs=workers):
                sink.write(part)
                n_out += len(part)
                batch_reports.append(rep)
            say(f"  {label}: {min(first + workers, len(batches))}/{len(batches)} "
                "batches filtered")
    except BaseException:
        sink.abort()
        raise
    sink.close()
    report.extend(_sum_reports(batch_reports, label))
    return n_out, report, int(ais["mmsi"].nunique())


# ---- shoreline --------------------------------------------------------------------------

def read_coastline(path, area=AREA, report=None, regions=SHORELINE_REGIONS,
                   margin=SHORELINE_MARGIN):
    """The shoreline lines of ``regions`` that touch ``area`` + ``margin`` degrees.

    In the file's CRS (EPSG:4269).  ``area`` is a :class:`StudyArea` (or an
    archive-style ``(lon_min, lon_max, lat_min, lat_max)`` tuple); ``None`` reads
    every line of the regions.
    """
    import geopandas as gpd

    bbox = None
    if area is not None:
        bbox = tuple(_zone(area, margin).bounds)
    coast = gpd.read_file(path, bbox=bbox)
    n_read = len(coast)
    coast = coast[coast["REGIONS"].isin(regions)].reset_index(drop=True)
    if report is not None:
        report.add("coastline", "read_coastline", "lines touching the area", n_read, n_read)
        report.add("coastline", "read_coastline", f"regions {','.join(regions)}", n_read,
                   len(coast))
    return coast


def _zone(area, margin):
    """The area's polygon buffered by ``margin`` degrees (lon/lat)."""
    from shapely.geometry import box

    poly = area.polygon if hasattr(area, "polygon") else box(area[0], area[2], area[1], area[3])
    return poly.buffer(margin)


def project_coastline(coast, crs=US_CRS, report=None):
    """Reproject the lines to ``crs`` (EPSG:32618, metres)."""
    out = coast.to_crs(crs)
    if report is not None:
        report.add("coastline", "project_coastline", f"to EPSG:{crs}", len(coast), len(out))
    return out


def clip_coastline(coast, area=AREA, report=None, margin=SHORELINE_MARGIN):
    """Clip the lines (in a projected CRS) to ``area`` + ``margin`` degrees.

    The zone is the area's polygon buffered in lon/lat, densified and projected
    to the lines' CRS; pieces that are not lines (points where a line only touches
    the edge) and empty results are dropped.
    """
    import geopandas as gpd

    zone = gpd.GeoSeries([_zone(area, margin).segmentize(0.01)], crs=4326).to_crs(
        coast.crs).iloc[0]
    coast = coast[coast.intersects(zone)].copy()
    n = len(coast)
    coast["geometry"] = coast.geometry.intersection(zone)
    out = coast[~coast.geometry.is_empty & coast.geometry.notna()].explode(index_parts=False)
    out = out[out.geometry.geom_type == "LineString"]
    if report is not None:
        report.add("coastline", "clip_coastline",
                   f"clipped to the area + {margin:g} degrees", n, len(out))
    return out[["REGIONS", "geometry"]].reset_index(drop=True)


def _find_shoreline(raw_dir: Path):
    for folder in (raw_dir / "shoreline", raw_dir.parent / "shoreline"):
        hits = sorted(folder.glob("*.shp")) if folder.is_dir() else []
        if hits:
            return hits[0]
    return None


# ---- the whole run ----------------------------------------------------------------------

def preprocess_us(raw_dir, out_root, months=("2017-11", "2017-12"), n_jobs=1, window=30,
                  force=False, *, shoreline=None, max_speed_knots=50,
                  batch_rows=BATCH_ROWS, log=print):
    """Preprocess the MarineCadastre daily files of ``months``; write to ``out_root``.

    Writes ``AIS/ais_<month>.parquet`` (the files ``ais-<month>-DD.csv.zst`` of
    ``raw_dir``, every day found), ``Coastline/shoreline.parquet`` and
    ``preprocessing_report.csv`` / ``.txt``.  A month without files is skipped,
    and so is an output that exists, unless ``force``.

    Parameters
    ----------
    raw_dir : path
        Folder of the daily files (``data/UsAIS/csv2017``, or a sample).  The
        shoreline is read from ``raw_dir/shoreline/*.shp`` or the sibling
        ``shoreline`` folder, unless ``shoreline`` names the file.
    months : sequence of "YYYY-MM"
    n_jobs : int or None
        Processes (``1``/``None`` serial, ``-1`` every core): the days of a month
        are read side by side, then the median filter runs per batch of whole
        vessels.  Files and report do not depend on it.  Memory: a month is about
        4 GB of frame, and each worker holds one daily file or one batch.
    window : float
        Median filter width in seconds.
    max_speed_knots : float or None
        Isolated position jumps above this speed are dropped.

    Returns
    -------
    PreprocessingReport
    """
    raw_dir, out_root = Path(raw_dir), Path(out_root)
    if not window > 0:
        raise ValueError("window must be positive")
    say = log if log is not None else (lambda *_: None)
    report, done = PreprocessingReport(), 0
    for month in months:
        files = sorted(raw_dir.glob(f"ais-{month}-??.csv*"))
        target = out_root / "AIS" / f"ais_{month}.parquet"
        if not files:
            say(f"skip {month}: no ais-{month}-DD.csv.zst in {raw_dir}")
            continue
        if target.exists() and not force:
            say(f"skip {month}: {target} exists (force=True to rebuild)")
            continue
        say(f"{month}: {len(files)} daily files ...")
        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        n, rep, n_vessels = _run_month(files, target, window, max_speed_knots,
                                       batch_rows, n_jobs, say)
        report.extend(rep)
        done += 1
        say(f"  {month}: {n:,} rows, {n_vessels:,} vessels -> {target} "
            f"({time.perf_counter() - t0:.0f} s)")

    shp = Path(shoreline) if shoreline is not None else _find_shoreline(raw_dir)
    target = out_root / "Coastline" / "shoreline.parquet"
    if shp is None or not shp.exists():
        say(f"skip shoreline: no shapefile in {raw_dir / 'shoreline'}")
    elif target.exists() and not force:
        say(f"skip shoreline: {target} exists (force=True to rebuild)")
    else:
        rep = PreprocessingReport()
        coast = read_coastline(shp, AREA, rep)
        coast = project_coastline(coast, US_CRS, rep)
        coast = clip_coastline(coast, AREA, rep)
        target.parent.mkdir(parents=True, exist_ok=True)
        coast.to_parquet(target, compression="zstd")
        report.extend(rep)
        done += 1
        say(f"  shoreline: {len(coast):,} lines -> {target}")
    if done:
        out_root.mkdir(parents=True, exist_ok=True)
        report.to_frame().to_csv(out_root / "preprocessing_report.csv", index=False)
        (out_root / "preprocessing_report.txt").write_text(report.to_text() + "\n",
                                                           encoding="utf-8")
    return report
