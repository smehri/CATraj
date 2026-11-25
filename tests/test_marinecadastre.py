"""The US MarineCadastre preprocessing: area, codes, types, grid north, a whole run."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import shapely

from catraj.compression import Barriers, cpls_select, segment_crossings
from catraj.preprocessing import get_dataset, marinecadastre as mc
from catraj.preprocessing.archive import PreprocessingReport

T0 = int(pd.Timestamp("2017-11-01", tz="UTC").timestamp())


# ---- study area ----------------------------------------------------------------------

def test_area_matches_the_polygon_on_random_points():
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-90, -55, 20000), rng.uniform(20, 50, 20000)
    poly = mc.study_area_polygon()
    expect = shapely.contains(poly, shapely.points(lon, lat))
    assert (mc.in_study_area(lon, lat) == expect).all()
    assert 0 < expect.sum() < len(lon)


@pytest.mark.parametrize("lon, lat, inside", [
    (-70, 35, True),             # open Atlantic
    (-81.5, 28, False),          # Gulf side of Florida (cut-out)
    (-80.5, 28, True),           # Florida's Atlantic coast, east of the cut-out
    (-81.0, 28, False),          # lon -81 < -80.9
    (-85, 28, False),            # west of the box
    (-78, 43, False),            # Great Lakes (cut-out)
    (-74, 43, True),             # Hudson valley, east of -75
    (-70, 44.9, True),           # Gulf of Maine
    (-71, 46.5, False),          # north of the box
    (-65, 22, False),            # south of the box
    (-59, 40, False),            # east of the box
    (-81.266, 31.0, False),      # exactly on a cut-out's edge: outside, as shapely's contains
])
def test_area_points(lon, lat, inside):
    assert bool(mc.in_study_area(lon, lat)) is inside


# ---- steps on small frames --------------------------------------------------------------

def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = dict(mmsi=np.full(n, 366000001), unix_time=T0 + 60 * np.arange(n),
                lon=np.full(n, -74.0), lat=np.full(n, 40.5),
                speed_over_ground=np.float32(5), course_over_ground=np.float32(90),
                true_heading=np.float32(90), rate_of_turn=np.float32("nan"),
                navigational_status=np.float32(0), ship_type_code=np.float32(70),
                length=np.float32(100), width=np.float32(20))
    base.update(cols)
    return pd.DataFrame(base)


def test_mmsi_keeps_ship_stations_only():
    df = _frame(mmsi=[366000001, 12345, 2275000, 111111111, 99366000, 538001234])
    rep = PreprocessingReport()
    out = mc.drop_invalid_mmsi(df, rep)
    assert list(out.mmsi) == [366000001, 538001234]
    assert len(rep) > 2


def test_time_is_limited_to_november_and_december_2017():
    t = [T0 - 1, T0, T0 + 3600, int(pd.Timestamp("2017-12-31 23:59:59", tz="UTC").timestamp()),
         int(pd.Timestamp("2018-01-01", tz="UTC").timestamp()), np.iinfo("int64").min]
    out = mc.drop_invalid_time(_frame(unix_time=t))
    assert list(out.unix_time) == t[1:4]


def test_codes_become_nan_and_ranges_are_checked():
    df = _frame(speed_over_ground=np.float32([102.3, 10, -1, 5]),
                course_over_ground=np.float32([360, 10, -5, 359.9]),
                true_heading=np.float32([511, 10, 400, 0]),
                navigational_status=np.float32([0, 15, 16, np.nan]),
                length=np.float32([0, 100, 2000, np.nan]),
                width=np.float32([10, 0, 127, 20]))
    rep = PreprocessingReport()
    out = mc.check_ranges(mc.unavailable_to_nan(df, rep), rep)
    assert out.speed_over_ground.isna().tolist() == [True, False, True, False]
    assert out.course_over_ground.isna().tolist() == [True, False, True, False]
    assert out.true_heading.isna().tolist() == [True, False, True, False]
    assert out.navigational_status.isna().tolist() == [False, False, True, True]
    assert out.length.isna().tolist() == [True, False, True, True]
    assert out.width.isna().tolist() == [False, True, True, False]
    assert len(out) == 4 and out.rate_of_turn.isna().all()


def test_position_is_checked_and_the_area_applied():
    df = _frame(lon=[-74, -181, -81.5, -74], lat=[40, 40, 28, 91])
    out = mc.drop_invalid_position(df, mc.AREA)
    assert list(out.lon) == [-74]
    both = mc.drop_invalid_position(df, None)           # only the valid range
    assert len(both) == 2


def test_duplicates_are_dropped_and_rows_sorted():
    df = _frame(mmsi=[2, 1, 1, 1], unix_time=[T0, T0 + 60, T0, T0 + 60],
                lon=[-74, -74.1, -74.2, -74.3])
    out = mc.drop_duplicates_and_sort(df)
    assert list(zip(out.mmsi, out.unix_time)) == [(1, T0), (1, T0 + 60), (2, T0)]


# ---- vessel types -----------------------------------------------------------------------

@pytest.mark.parametrize("code, name, group", [
    (30, "Fishing", "Fishing"), (31, "Towing", "Tug"), (35, "Military Ops", "Military"),
    (36, "Sailing Vessel", "Sailing"), (37, "Pleasure Craft", "Sailing"),
    (52, "Tug", "Tug"), (60, "Passenger", "Passenger"), (69, "Passenger", "Passenger"),
    (70, "Cargo", "Cargo"), (79, "Cargo", "Cargo"), (80, "Tanker", "Tanker"),
    (89, "Tanker", "Tanker"), (90, "Other", "Other"), (0, "Unknown", "Other"),
    (1001, "Fishing", "Fishing"), (1003, "Freight Barge", "Cargo"),
    (1004, "Cargo", "Cargo"), (1012, "Passenger", "Passenger"),
    (1019, "Pleasure Craft", "Sailing"), (1021, "SAR Aircraft", "Military"),
    (1023, "Tank Barge", "Tug"), (1024, "Tanker", "Tanker"), (1025, "Tug", "Tug"),
    (1010, "Offshore Supply Vessel", "Other"), (500, "Other", "Other"),
    (1099, "Other", "Other"),
])
def test_vessel_type_codes(code, name, group):
    assert mc.vessel_type_name(code) == name
    assert mc.vessel_group_of(code) == group


def test_a_null_type_is_other_and_every_code_is_covered():
    assert mc.vessel_type_name(None) == "Unknown" and mc.vessel_type_name(np.nan) == "Unknown"
    assert mc.vessel_group_of(np.nan) == "Other" and mc.vessel_group_of(None) == "Other"
    assert {mc.vessel_group_of(c) for c in range(0, 1026)} <= set(mc.VESSEL_GROUPS)
    for c in list(range(0, 100)) + list(range(1001, 1026)):
        assert c in mc.VESSEL_TYPE_NAMES


def test_map_vessel_types_gives_the_same_categories_whatever_the_codes():
    a = mc.map_vessel_types(pd.Series([70.0, np.nan, 1004.0]))
    b = mc.map_vessel_types(pd.Series([80.0]))
    assert list(a.vessel_type) == ["Cargo", "Unknown", "Cargo"]
    assert list(a.vessel_group) == ["Cargo", "Other", "Cargo"]
    assert a.vessel_type.dtype == b.vessel_type.dtype
    assert a.vessel_group.dtype == b.vessel_group.dtype


def test_attach_vessel_types_takes_the_most_frequent_code_per_vessel():
    df = _frame(mmsi=[1] * 5 + [2] * 3,
                ship_type_code=np.float32([70, 70, 80, np.nan, 80, np.nan, np.nan, np.nan]),
                length=np.float32([100, 100, 90, np.nan, 90, np.nan, np.nan, np.nan]))
    out = mc.attach_vessel_types(df)
    one, two = out[out.mmsi == 1], out[out.mmsi == 2]
    assert set(one.ship_type_code) == {np.float32(70)}      # 70: two, 80: two -> smallest
    assert set(one.vessel_type) == {"Cargo"}
    assert set(one.length) == {np.float32(90)} or set(one.length) == {np.float32(100)}
    assert two.ship_type_code.isna().all()
    assert set(two.vessel_type) == {"Unknown"} and set(two.vessel_group) == {"Other"}
    assert list(mc.most_frequent([1, 1, 1, 2], [5, 6, 6, np.nan]).items()) == [(1, 6.0)]


# ---- grid north -------------------------------------------------------------------------

@pytest.mark.parametrize("lon, lat", [(-81.0, 25.0), (-80.0, 32.0), (-75.0, 35.0),
                                      (-73.9, 40.7), (-70.0, 42.0), (-66.0, 44.0),
                                      (-61.0, 45.5), (-76.4, 37.0)])
def test_grid_bearing_matches_the_projected_direction(lon, lat):
    geod = pyproj.Geod(ellps="WGS84")
    to_utm = pyproj.Transformer.from_crs(4326, mc.US_CRS, always_xy=True)
    for bearing in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 10.0):
        lon1, lat1, _ = geod.fwd(lon, lat, bearing, 10_000.0)
        x0, y0 = to_utm.transform(lon, lat)
        x1, y1 = to_utm.transform(lon1, lat1)
        on_map = np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360
        diff = (mc.grid_bearing(lon, lat, bearing) - on_map + 180) % 360 - 180
        assert abs(diff) < 0.05, (lon, lat, bearing, diff)


def test_grid_bearing_arrays_nan_and_the_central_meridian():
    out = mc.grid_bearing([-75.0, -66.0], [40.0, 44.0], [350.0, np.nan])
    assert out[0] == pytest.approx(350.0, abs=1e-6)          # -75 is zone 18's centre
    assert np.isnan(out[1])
    assert 0 <= mc.grid_bearing(-61, 45, 5.0) < 360          # wraps: 5 - 10 -> 355
    assert mc.grid_bearing(-61, 45, 5.0) == pytest.approx(355.0, abs=0.05)


# ---- shoreline lines as CPLS barriers ------------------------------------------------------

def test_linestring_shoreline_barriers_in_cpls():
    # A shoreline of LineStrings (with a REGIONS column) in a projected CRS, as
    # shoreline.parquet holds it, with a headland the track must go round.
    coast = gpd.GeoDataFrame(
        {"REGIONS": ["M", "M"]},
        geometry=[shapely.LineString([(5000, -50), (5000, 50)]),
                  shapely.LineString([(20000, 1000), (21000, 1200)])], crs=mc.US_CRS)
    barriers = Barriers(coast)
    assert len(barriers) == 2 and barriers.crs == coast.crs
    x = np.array([0, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000,
                  7000, 8000, 9000, 10000], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 80, 100, 80, 0, 0, 0, 0, 0], dtype=float)
    sel = cpls_select(x, y, 500, barriers)
    assert segment_crossings(x, y, barriers, keep=sel.dp).any()      # plain DP cuts the line
    assert not segment_crossings(x, y, barriers, keep=sel.mask).any()
    assert sel.mask.sum() > sel.dp.sum()
    assert barriers.hits([4000], [0], [6000], [0]).all()
    assert not barriers.hits([0], [0], [1000], [0]).any()


# ---- the whole run on a tiny raw folder ----------------------------------------------------

def _write_day(path, rows):
    import io

    import zstandard

    cols = ["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "heading",
            "vessel_name", "imo", "call_sign", "vessel_type", "status", "length", "width",
            "draft", "cargo", "transceiver"]
    df = pd.DataFrame(rows, columns=cols)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    path.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue().encode()))


def _track(mmsi, day, start_hour, n, lon0, lat0, code, **kw):
    rows = []
    for i in range(n):
        t = pd.Timestamp(f"2017-11-{day:02d} {start_hour:02d}:00:00") + pd.Timedelta(seconds=60 * i)
        rows.append(dict(
            mmsi=mmsi, base_date_time=t.strftime("%Y-%m-%d %H:%M:%S"),
            longitude=lon0 + 0.001 * i, latitude=lat0, sog=8.0, cog=90.0, heading=90.0,
            vessel_name="X", imo=np.nan, call_sign=np.nan, vessel_type=code, status=0.0,
            length=100.0, width=20.0, draft=np.nan, cargo=np.nan, transceiver=np.nan))
        rows[-1].update(kw)
    return rows


def _tiny_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    day1 = (_track(366000001, 1, 22, 120, -74.0, 40.5, 70)             # crosses midnight
            + _track(366000002, 1, 3, 30, -70.0, 35.0, 1019)           # pleasure craft
            + _track(366000003, 1, 3, 30, -90.0, 28.0, 80)             # Gulf: outside
            + _track(366000004, 1, 3, 30, -78.0, 43.0, 80)             # Great Lakes: outside
            + _track(12345, 1, 3, 30, -72.0, 38.0, 70)                 # bad MMSI
            + _track(366000005, 1, 5, 20, -72.0, 38.0, np.nan,
                     cog=-1.0, sog=102.3, status=20.0)                 # no type, bad codes
            + _track(366000006, 1, 6, 30, -74.0, 45.0, 1012))          # St. Lawrence part
    day1 += day1[:5]                                                   # duplicates
    day2 = (_track(366000001, 2, 0, 60, -73.88, 40.5, 70)
            + _track(366000007, 2, 1, 30, -66.0, 44.0, 1024))
    _write_day(raw / "ais-2017-11-01.csv.zst", day1)
    _write_day(raw / "ais-2017-11-02.csv.zst", day2)
    shore = gpd.GeoDataFrame(
        {"REGIONS": ["M", "G", "N"]},
        geometry=[shapely.LineString([(-74.5, 39.0), (-73.5, 41.5)]),
                  shapely.LineString([(-90, 28), (-89, 29)]),          # Gulf region
                  shapely.LineString([(-71, 42), (-70, 43)])], crs=4269)
    (raw / "shoreline").mkdir()
    shore.to_file(raw / "shoreline" / "us_medium_shoreline.shp")
    return raw


def test_preprocess_us_end_to_end(tmp_path):
    raw, out = _tiny_raw(tmp_path), tmp_path / "out"
    logs = []
    rep = mc.preprocess_us(raw, out, months=("2017-11", "2017-12"), n_jobs=1,
                           log=logs.append)
    assert any("skip 2017-12" in line for line in logs)
    ais = pd.read_parquet(out / "AIS" / "ais_2017-11.parquet")
    assert not (out / "AIS" / "ais_2017-12.parquet").exists()
    assert list(ais.columns) == mc.OUTPUT_COLUMNS
    assert set(ais.mmsi) == {366000001, 366000002, 366000005, 366000006, 366000007}
    one = ais[ais.mmsi == 366000001]
    assert len(one) == 180                                  # both days, one vessel
    assert (one.unix_time.diff().dropna() > 0).all()
    assert one.vessel_type.iloc[0] == "Cargo" and one.vessel_group.iloc[0] == "Cargo"
    assert set(ais[ais.mmsi == 366000002].vessel_group) == {"Sailing"}
    five = ais[ais.mmsi == 366000005]
    assert set(five.vessel_type) == {"Unknown"} and set(five.vessel_group) == {"Other"}
    assert five.course_over_ground.isna().all() and five.speed_over_ground.isna().all()
    assert five.navigational_status.isna().all()
    assert set(ais[ais.mmsi == 366000006].vessel_group) == {"Passenger"}
    # grid north: course 90 at -74 is a little off 90 on the map; the filter keeps it
    lonlat_one = one.iloc[0]
    expect = mc.grid_bearing(lonlat_one.lon, lonlat_one.lat, lonlat_one.course_over_ground)
    assert lonlat_one.course_over_ground_grid == pytest.approx(expect, abs=1e-3)
    assert one.x.between(0, 1e6).all() and one.y.between(4.4e6, 4.6e6).all()
    # report
    frame = rep.to_frame()
    def n(step, rule_part, col="rows_dropped"):
        sel = frame[(frame.step == step) & frame.rule.str.contains(rule_part, regex=False)]
        return int(sel[col].sum())
    assert n("read_ais_raw", "inside the study area") == 60       # Gulf + Great Lakes
    assert n("drop_duplicates_and_sort", "duplicate") == 5
    assert n("drop_invalid_mmsi", "kept") == 30
    assert n("read_ais_raw", "St. Lawrence", "rows_in") == 30
    assert (out / "preprocessing_report.csv").exists()
    # shoreline: Gulf region dropped, lines projected
    coast = gpd.read_parquet(out / "Coastline" / "shoreline.parquet")
    assert set(coast.REGIONS) <= {"M", "N"} and len(coast) == 2
    assert coast.crs.to_epsg() == mc.US_CRS
    # a second run skips what exists
    logs.clear()
    mc.preprocess_us(raw, out, months=("2017-11",), log=logs.append)
    assert any("exists" in line for line in logs)


def test_the_filter_does_not_depend_on_the_number_of_processes(tmp_path):
    raw = _tiny_raw(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    mc.preprocess_us(raw, a, months=("2017-11",), n_jobs=1, batch_rows=200)
    mc.preprocess_us(raw, b, months=("2017-11",), n_jobs=2, batch_rows=200)
    pd.testing.assert_frame_equal(pd.read_parquet(a / "AIS" / "ais_2017-11.parquet"),
                                  pd.read_parquet(b / "AIS" / "ais_2017-11.parquet"))


def test_denoise_equals_the_generic_filter():
    from catraj.preprocessing import denoise

    rng = np.random.default_rng(3)
    n = 60
    df = _frame(mmsi=np.repeat([1, 2], n // 2), unix_time=np.tile(T0 + 20 * np.arange(n // 2), 2),
                lon=-74 + 0.01 * np.arange(n) % 0.3 + rng.normal(0, 1e-4, n),
                lat=40 + rng.normal(0, 1e-4, n),
                course_over_ground=np.float32(rng.uniform(350, 360, n)),
                speed_over_ground=np.float32(rng.uniform(5, 6, n)))
    got = mc.denoise_ais(df.drop(columns=["rate_of_turn"]), 30)
    proj = mc.project(df)
    gdf = gpd.GeoDataFrame(proj, geometry=gpd.points_from_xy(proj.x, proj.y), crs=mc.US_CRS)
    ref = denoise(gdf, 30, by="mmsi", time_column="unix_time",
                  columns=["speed_over_ground", "course_over_ground", "true_heading"],
                  circular=["course_over_ground", "true_heading"])
    assert np.allclose(got.x, ref.geometry.x) and np.allclose(got.y, ref.geometry.y)
    assert np.allclose(got.speed_over_ground, ref.speed_over_ground, atol=1e-4)
    d = (got.course_over_ground - ref.course_over_ground + 180) % 360 - 180
    assert np.abs(d).max() < 1e-3


def test_the_dataset_registry_describes_both_datasets():
    us, zen = get_dataset("us"), get_dataset("zenodo")
    assert us.crs == 32618 and zen.crs == 32630
    assert us.has == ("ais", "coastline") and "weather" in zen.has and "ocean" in zen.has
    assert us.steps.read_ais_raw and zen.steps.read_ais_raw
    for name in ("read_ais_raw", "drop_invalid_mmsi", "drop_invalid_time",
                 "drop_invalid_position", "unavailable_to_nan", "check_ranges",
                 "drop_duplicates_and_sort", "drop_position_jumps", "project",
                 "denoise_ais", "read_coastline", "project_coastline", "clip_coastline"):
        assert callable(getattr(us.steps, name)) and callable(getattr(zen.steps, name))
    assert us.sample_preprocessed.parts[-3:] == ("UsAIS", "sample", "preprocessed")
    with pytest.raises(ValueError):
        get_dataset("mars")


# ---- the committed sample ---------------------------------------------------------------------

SAMPLE_US = get_dataset("us").sample_preprocessed


@pytest.mark.data
@pytest.mark.skipif(not (SAMPLE_US / "AIS").exists(), reason="US sample not present")
def test_us_sample_preprocessed():
    files = sorted((SAMPLE_US / "AIS").glob("ais_*.parquet"))
    assert files
    ais = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert list(ais.columns) == mc.OUTPUT_COLUMNS
    assert mc.in_study_area(ais.lon, ais.lat).mean() > 0.999
    assert ais.mmsi.nunique() > 100
    assert ais.vessel_group.nunique() >= 4
    assert (ais.groupby("mmsi").unix_time.diff().dropna() > 0).all()
    g = ais.dropna(subset=["course_over_ground"])
    diff = (g.course_over_ground_grid - mc.grid_bearing(
        g.lon.to_numpy(), g.lat.to_numpy(), g.course_over_ground.to_numpy()) + 180) % 360 - 180
    assert np.abs(diff).max() < 1e-3
    coast = gpd.read_parquet(SAMPLE_US / "Coastline" / "shoreline.parquet")
    assert len(coast) > 0 and coast.crs.to_epsg() == 32618
    assert set(coast.geometry.geom_type) == {"LineString"}
    assert (SAMPLE_US / "preprocessing_report.csv").exists()
