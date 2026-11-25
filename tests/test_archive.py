"""Preprocessing of the Zenodo archive: every rule, on tiny synthetic tables."""

import numpy as np
import pandas as pd
import pytest

from catraj.preprocessing import (
    ARCHIVE_PERIOD, PreprocessingReport, attach_vessel_types, check_ocean_ranges,
    check_ranges, check_weather_ranges, clip_coastline, denoise_ais, drop_invalid_mmsi,
    drop_invalid_ocean, drop_invalid_position, drop_invalid_time, drop_land_cells,
    drop_duplicates_and_sort, drop_position_jumps, mmsi_check,
    ocean_unavailable_to_nan, preprocess_archive,
    project, sort_ocean, unavailable_to_nan, weather_unavailable_to_nan,
    merge_weather_stations,
)
from helpers import ROOT

T0 = ARCHIVE_PERIOD[0] + 86_400 * 40          # a time inside the archive's period
SHIP = 227705102                              # France, MID 227


def ais_frame(**columns):
    """One valid AIS row per entry of the longest column; others take defaults."""
    n = max((len(v) for v in columns.values()), default=1)
    base = dict(
        mmsi=[SHIP] * n, navigational_status=[0.0] * n, rate_of_turn=[0.0] * n,
        speed_over_ground=[5.0] * n, course_over_ground=[90.0] * n,
        true_heading=[90.0] * n, lon=[-4.5] * n, lat=[48.4] * n,
        unix_time=[T0 + 10 * i for i in range(n)])
    base.update(columns)
    df = pd.DataFrame(base)
    for c in ("navigational_status", "rate_of_turn", "speed_over_ground",
              "course_over_ground", "true_heading"):
        df[c] = df[c].astype("float32")
    return df


def bad(report, step, rule_part):
    """The one report line of ``step`` whose rule starts with ``rule_part``."""
    f = report.to_frame()
    named = (f.rule == rule_part) | f.rule.str.startswith(rule_part + " ")
    row = f[(f.step == step) & named]
    assert len(row) == 1, (step, rule_part, f.rule.tolist())
    return row.iloc[0]


# ---- AIS --------------------------------------------------------------------------------

def test_mmsi_keeps_ship_stations_and_reports_each_class():
    ais = ais_frame(mmsi=[SHIP, 123456789, 2275000, 992271234, 1000000000, SHIP])
    r = PreprocessingReport()
    out = drop_invalid_mmsi(ais, r)
    assert out.mmsi.tolist() == [SHIP, SHIP]
    assert bad(r, "drop_invalid_mmsi", "mmsi: placeholder").rows_dropped == 1
    assert bad(r, "drop_invalid_mmsi", "mmsi: coast station").rows_dropped == 1
    assert bad(r, "drop_invalid_mmsi", "mmsi: aid to navigation").rows_dropped == 1
    assert bad(r, "drop_invalid_mmsi", "mmsi: not nine digits").rows_dropped == 1
    assert bad(r, "drop_invalid_mmsi", "mmsi: ship station kept").rows_out == 2
    check = mmsi_check(ais).set_index("mmsi_class")
    assert check.loc["ship", "positions"] == 2 and check.loc["ship", "MMSIs"] == 1


def test_time_outside_the_archive_period_is_dropped():
    lo, hi = ARCHIVE_PERIOD
    ais = ais_frame(unix_time=[lo - 1, lo, T0, hi, hi + 1])
    r = PreprocessingReport()
    out = drop_invalid_time(ais, r)
    assert out.unix_time.tolist() == [lo, T0, hi]
    assert r.to_frame().rows_dropped.iloc[0] == 2


def test_position_rules_valid_range_then_archive_area():
    ais = ais_frame(lon=[-4.5, 181.0, -4.5, -10.1, 0.1, -10.0, 0.0],
                    lat=[48.4, 48.4, 91.0, 48.4, 48.4, 45.0, 51.0])
    r = PreprocessingReport()
    out = drop_invalid_position(ais, report=r)
    assert out.lon.tolist() == [-4.5, -10.0, 0.0]            # the edges are inside
    assert bad(r, "drop_invalid_position", "lon, lat in valid range").rows_dropped == 2
    area = bad(r, "drop_invalid_position", "position inside archive area")
    assert area.rows_dropped == 2
    assert len(drop_invalid_position(ais, area=None)) == 5


def test_unavailable_codes_become_nan():
    ais = ais_frame(speed_over_ground=[5, 102.3, 102.2],
                    course_over_ground=[10, 360, 359.9],
                    true_heading=[10, 511, 359], rate_of_turn=[0, -128, 5])
    r = PreprocessingReport()
    out = unavailable_to_nan(ais, r)
    assert out.speed_over_ground.isna().tolist() == [False, True, False]
    assert out.course_over_ground.isna().tolist() == [False, True, False]
    assert out.true_heading.isna().tolist() == [False, True, False]
    assert out.rate_of_turn.isna().tolist() == [False, True, False]
    assert not ais.speed_over_ground.isna().any()             # the input is untouched
    for rule in ("speed_over_ground", "course_over_ground", "true_heading",
                 "rate_of_turn"):
        assert bad(r, "unavailable_to_nan", rule).values_to_nan == 1


@pytest.mark.parametrize("column, values, valid", [
    ("speed_over_ground", [0, 102.2, -0.1, 102.3, 500], [1, 1, 0, 0, 0]),
    ("course_over_ground", [0, 359.9, -1, 360, 400], [1, 1, 0, 0, 0]),
    ("true_heading", [0, 359, -1, 360, 511], [1, 1, 0, 0, 0]),
    ("rate_of_turn", [0, 127, -127, 128, -300], [1, 1, 1, 0, 0]),
])
def test_check_ranges_numeric_fields(column, values, valid):
    ais = ais_frame(**{column: values})
    r = PreprocessingReport()
    out = check_ranges(ais, r)
    assert out[column].notna().tolist() == [bool(v) for v in valid]
    assert bad(r, "check_ranges", column).values_to_nan == valid.count(0)


def test_check_ranges_navigational_status():
    ais = ais_frame(navigational_status=[0, 15, 16, -1, 3.5])
    r = PreprocessingReport()
    out = check_ranges(ais, r)
    assert out.navigational_status.isna().tolist() == [False, False, True, True, True]
    assert bad(r, "check_ranges", "navigational_status").values_to_nan == 3
    complete = check_ranges(ais_frame(navigational_status=[0, 1, 15]))
    assert complete.navigational_status.dtype == np.int8      # a code, once complete


def test_range_checks_count_only_values_that_were_present():
    ais = ais_frame(speed_over_ground=[np.nan, 200.0])
    r = PreprocessingReport()
    check_ranges(ais, r)
    assert bad(r, "check_ranges", "speed_over_ground").values_to_nan == 1


def test_duplicates_and_sort():
    ais = ais_frame(mmsi=[SHIP + 1, SHIP, SHIP, SHIP + 1],
                    unix_time=[T0, T0 + 5, T0 + 5, T0 - 5])
    r = PreprocessingReport()
    out = drop_duplicates_and_sort(ais, r)
    assert out[["mmsi", "unix_time"]].values.tolist() == [
        [SHIP, T0 + 5], [SHIP + 1, T0 - 5], [SHIP + 1, T0]]
    assert list(out.index) == [0, 1, 2]
    assert r.to_frame().rows_dropped.iloc[0] == 1


def track(lat, mmsi=SHIP, dt=60, lon=-4.5):
    n = len(lat)
    return ais_frame(mmsi=[mmsi] * n, lat=list(lat), lon=[lon] * n,
                     unix_time=[T0 + dt * i for i in range(n)])


def test_position_jump_spike_is_removed():
    lat = [48.4, 48.4001, 48.4002, 49.4, 48.4004, 48.4005]   # 1 degree in a minute
    r = PreprocessingReport()
    out = drop_position_jumps(track(lat), 50, r)
    assert out.lat.max() < 48.5 and len(out) == 5
    jump = bad(r, "drop_position_jumps", "isolated jump above 50 knots")
    assert jump.rows_dropped == 1


def test_fast_but_consistent_track_is_kept():
    # 0.01 degree of latitude (1.1 km) a minute is 36 knots; 0.02 is 72 knots
    fast = track([48.4 + 0.02 * i for i in range(10)])
    assert len(drop_position_jumps(fast, 50)) == 10           # no isolated point
    assert len(drop_position_jumps(track([48.4 + 0.01 * i for i in range(10)]))) == 10


def test_position_jump_at_the_ends_needs_one_fast_neighbour():
    out = drop_position_jumps(track([49.4, 48.4, 48.4001, 48.4002]))
    assert out.lat.iloc[0] == 48.4 and len(out) == 3          # the first point
    out = drop_position_jumps(track([48.4, 48.4001, 48.4002, 49.4]))
    assert out.lat.iloc[-1] == 48.4002 and len(out) == 3      # the last point
    assert len(drop_position_jumps(track([48.4]))) == 1       # no neighbour at all


def test_position_jump_is_per_vessel_and_handles_zero_time_gaps():
    a, b = track([48.4, 48.4001], SHIP), track([40.0, 40.0001], SHIP + 1)
    out = drop_position_jumps(pd.concat([a, b], ignore_index=True))
    assert len(out) == 4                                       # not across vessels
    zero = track([48.4, 48.4, 48.4, 48.4], dt=0)               # same time, same place
    assert len(drop_position_jumps(zero)) == 4
    moved = track([48.4, 48.4, 49.4, 48.4], dt=0)              # same time, far apart
    assert len(drop_position_jumps(moved)) < 4                 # infinite speed
    assert len(drop_position_jumps(track([48.4, 49.4, 48.4]), None)) == 3


def test_vessel_types(tmp_path):
    static = tmp_path / "nari_static.csv"
    static.write_text("sourcemmsi,shiptype\n"
                      f"{SHIP},30\n{SHIP},70\n{SHIP + 1},\n{SHIP + 2},31\n"
                      f"{SHIP + 4},0\n")
    types = tmp_path / "Ship Types List.csv"
    types.write_text("id_shiptype,shiptype_min,shiptype_max,type_name,"
                     "ais_type_summary\n"
                     "4,30,30,Fishing,Fishing\n5,31,32,Towing - Tug,Towing\n"
                     "6,70,79,Cargo - all,Cargo\n")
    ais = ais_frame(mmsi=[SHIP, SHIP + 1, SHIP + 2, SHIP + 3, SHIP + 4])
    r = PreprocessingReport()
    out = attach_vessel_types(ais, static, types, r)
    assert out.vessel_type.tolist() == ["Cargo", "Unknown", "Towing", "Unknown",
                                        "Unknown"]
    # (the last static message of a vessel wins; code 0 is not in the table)
    assert out.ship_type_code.isna().tolist() == [False, True, False, True, False]
    counts = r.to_frame().values_to_nan.tolist()
    assert counts == [2, 1]          # no static message; code not in the table


def test_project_and_denoise_ais_recompute_lon_lat():
    n = 40
    ais = ais_frame(lon=list(-4.5 + 1e-4 * np.arange(n)), lat=[48.4] * n,
                    unix_time=[T0 + 10 * i for i in range(n)])
    ais.loc[20, "lat"] = 48.5                                  # a position spike
    p = project(ais)
    assert {"x", "y"} <= set(p.columns) and p.x.between(1e5, 9e5).all()
    out = denoise_ais(p, 30.0)
    assert len(out) == n
    assert out.lat.sub(48.4).abs().max() < 1e-3                # the spike is gone
    assert out.course_over_ground.max() < 360


# ---- weather ------------------------------------------------------------------------------

def weather_frame(**columns):
    n = max((len(v) for v in columns.values()), default=1)
    base = dict(id_station=[1] * n, local_time=[T0 + 3600 * i for i in range(n)],
                T=[10.0] * n, Tn=[5.0] * n, Tx=[12.0] * n, P=[760.0] * n, U=[80.0] * n,
                id_windDirection=[3] * n, Ff=[5.0] * n, ff10=[8.0] * n, ff3=[9.0] * n,
                VV=[10.0] * n, Td=[8.0] * n, RRR=[0.5] * n, tR=[6.0] * n)
    base.update(columns)
    return pd.DataFrame(base)


def test_weather_missing_code_becomes_nan_in_every_column():
    obs = weather_frame(T=[1.0, -65536, 2.0], Ff=[-65536, -65536, 3.0])
    r = PreprocessingReport()
    out = weather_unavailable_to_nan(obs, r)
    assert out["T"].isna().tolist() == [False, True, False]
    assert out["Ff"].isna().sum() == 2
    assert bad(r, "weather_unavailable_to_nan", "air_temperature").values_to_nan == 1
    assert bad(r, "weather_unavailable_to_nan", "wind_speed").values_to_nan == 2
    assert (out["id_station"] == 1).all()                      # ids are not touched


def test_weather_station_merge_renames_and_drops_unknown_stations():
    obs = weather_frame(id_station=[1, 2, 99])
    stations = pd.DataFrame({"id_station": [1, 2], "station_name": ["a", "b"],
                             "latitude": [48.0, 49.0], "longitude": [-4.0, -5.0],
                             "elevation": [1, 2]})
    r = PreprocessingReport()
    out = merge_weather_stations(obs, stations, r)
    assert out.station_id.tolist() == [1, 2]
    assert {"air_temperature", "wind_speed", "latitude", "longitude"} <= set(
        out.columns)
    assert r.to_frame().rows_dropped.iloc[0] == 1


@pytest.mark.parametrize("column, bad_values", [
    ("air_temperature", [-91, 61]), ("min_air_temperature", [-91, 61]),
    ("max_air_temperature", [-91, 61]), ("dew_point", [-91, 61]),
    ("air_pressure", [639, 821]), ("relative_humidity", [-1, 101]),
    ("wind_speed", [-1, 121]), ("max_gust_10min", [-1, 121]), ("max_gust", [-1, 121]),
    ("visibility", [-1, 101]), ("precipitation", [-1, 1001]),
    ("precipitation_period", [-1, 25]),
])
def test_weather_range_rules(column, bad_values):
    obs = merge_weather_stations(
        weather_frame(id_station=[1] * 4),
        pd.DataFrame({"id_station": [1], "latitude": [48.0], "longitude": [-4.0]}))
    obs[column] = [obs[column].iloc[0], bad_values[0], bad_values[1], np.nan]
    r = PreprocessingReport()
    out = check_weather_ranges(obs, report=r)
    assert out[column].isna().tolist() == [False, True, True, True]
    assert bad(r, "check_weather_ranges", column).values_to_nan == 2


def test_wind_direction_code_must_be_in_the_table():
    obs = merge_weather_stations(
        weather_frame(id_windDirection=[1, 19, 25]),
        pd.DataFrame({"id_station": [1], "latitude": [48.0], "longitude": [-4.0]}))
    table = pd.DataFrame({"id_windDirection": range(1, 20)})
    out = check_weather_ranges(obs, table)
    assert out.wind_direction_code.isna().tolist() == [False, False, True]


# ---- ocean ----------------------------------------------------------------------------------

def ocean_frame(**columns):
    n = max((len(v) for v in columns.values()), default=1)
    base = dict(lon=[-5.0] * n, lat=[48.0] * n, bottom_depth=[100.0] * n,
                sea_level=[0.5] * n, wave_height=[1.5] * n, wave_length=[60.0] * n,
                wave_direction=[200.0] * n,
                unix_time=[T0 + 10_800 * i for i in range(n)])
    base.update(columns)
    return pd.DataFrame(base)


def test_ocean_land_cells_and_area_and_time():
    oc = ocean_frame(bottom_depth=[100, -16384, -16000, -15999, 10],
                     lon=[-5, -5, -5, -5, -11])
    r = PreprocessingReport()
    oc = drop_invalid_ocean(oc, report=r)                     # the cell at lon -11
    assert len(oc) == 4
    land = drop_land_cells(oc, r)
    assert land.bottom_depth.tolist() == [100, -15999]
    lo = ARCHIVE_PERIOD[0]
    assert len(drop_invalid_ocean(ocean_frame(unix_time=[lo - 1, lo]))) == 1


def test_ocean_undefined_codes_become_nan():
    oc = ocean_frame(sea_level=[-327.67, -1.2, 0.3], wave_height=[-65.534, 0.0, 2.0],
                     wave_length=[-32767, 0, 50], wave_direction=[-3276.7, 0, 359.9])
    r = PreprocessingReport()
    out = ocean_unavailable_to_nan(oc, r)
    assert out.sea_level.isna().tolist() == [True, False, False]   # -1.2 is a tide
    assert out.wave_height.isna().tolist() == [True, False, False]
    assert out.wave_length.isna().tolist() == [True, False, False]
    assert out.wave_direction.isna().tolist() == [True, False, False]
    assert bad(r, "ocean_unavailable_to_nan", "sea_level").values_to_nan == 1


@pytest.mark.parametrize("column, bad_values", [
    ("bottom_depth", [11_001]), ("sea_level", [-21, 21]), ("wave_height", [31]),
    ("wave_length", [2001]), ("wave_direction", [360, 400]),
])
def test_ocean_range_rules(column, bad_values):
    oc = ocean_frame(**{column: [ocean_frame()[column].iloc[0], *bad_values]})
    r = PreprocessingReport()
    out = check_ocean_ranges(oc, r)
    assert out[column].isna().tolist() == [False] + [True] * len(bad_values)
    assert bad(r, "check_ocean_ranges", column).values_to_nan == len(bad_values)


def test_ocean_sorted_by_cell_then_time():
    oc = ocean_frame(lon=[-5, -6, -5, -6], lat=[48, 48, 48, 48],
                     unix_time=[T0 + 10_800, T0, T0, T0 + 10_800])
    out = sort_ocean(oc)
    assert out.lon.tolist() == [-6, -6, -5, -5]
    assert out.unix_time.tolist() == [T0, T0 + 10_800, T0, T0 + 10_800]


# ---- coastline --------------------------------------------------------------------------------

def test_clip_coastline_to_the_area():
    import geopandas as gpd
    from shapely.geometry import box

    # one polygon crossing the area's west edge, one far away, in UTM 30N metres
    inside = gpd.GeoSeries([box(-5, 47, -3, 48)], crs=4326).to_crs(32630).iloc[0]
    crossing = gpd.GeoSeries([box(-12, 47, -9, 48)], crs=4326).to_crs(32630).iloc[0]
    outside = gpd.GeoSeries([box(5, 40, 6, 41)], crs=4326).to_crs(32630).iloc[0]
    coast = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=[inside, crossing, outside],
                             crs=32630)
    r = PreprocessingReport()
    out = clip_coastline(coast, report=r)
    assert out.id.tolist() == [1, 2]
    assert out.geometry.iloc[1].area < crossing.area
    west = gpd.GeoSeries(out.geometry.iloc[1], crs=32630).to_crs(4326).total_bounds
    assert west[0] >= -10.001
    assert r.to_frame().rows_out.iloc[0] == 2


# ---- the whole archive ---------------------------------------------------------------------------

def write_raw_root(root):
    """A tiny archive: two ships and two bad rows; weather; ocean; coastline."""
    import geopandas as gpd
    from shapely.geometry import box

    ais = root / "P1_AIS_Data"
    ais.mkdir(parents=True)
    rows = []
    for mmsi in (SHIP, SHIP + 1):
        for i in range(30):
            rows.append((mmsi, 0, 0, 5.0, 90.0, 90, -4.5 + 1e-4 * i, 48.4, T0 + 10 * i))
    rows.append((SHIP, 0, 0, 5.0, 90.0, 90, -4.5, 48.4, T0))              # duplicate
    rows.append((123456789, 0, 0, 5.0, 90.0, 90, -4.5, 48.4, T0))             # MMSI
    rows.append((SHIP, 0, 0, 5.0, 90.0, 90, 181.0, 91.0, T0 + 500))           # position
    rows.append((SHIP, 0, -128, 102.3, 360.0, 511, -4.5, 48.4, T0 + 600))  # n/a
    rows.append((SHIP, 99, 0, 250.0, 90.0, 90, -4.5, 48.4, T0 + 700))     # ranges
    rows.append((SHIP + 1, 0, 0, 5.0, 90.0, 90, -4.5, 49.9, T0 + 55))     # jump
    pd.DataFrame(rows, columns=[
        "sourcemmsi", "navigationalstatus", "rateofturn", "speedoverground",
        "courseoverground", "trueheading", "lon", "lat", "t"]
    ).to_csv(ais / "nari_dynamic.csv", index=False)
    (ais / "nari_static.csv").write_text(
        f"sourcemmsi,shiptype\n{SHIP},30\n{SHIP + 1},70\n")
    codes = root / "P1_AIS_Codes"
    codes.mkdir()
    (codes / "Ship Types List.csv").write_text(
        "id_shiptype,shiptype_min,shiptype_max,type_name,ais_type_summary\n"
        "4,30,30,Fishing,Fishing\n6,70,79,Cargo - all,Cargo\n")

    wx = root / "E2_Weather_Conditions"
    wx.mkdir()
    obs = weather_frame(id_station=[1] * 4 + [2] * 4,
                        local_time=[T0 + 3600 * i for i in range(4)] * 2,
                        T=[10.0, -65536, 11.0, 12.0] * 2)
    obs.loc[7, "P"] = 2000.0
    obs.to_csv(wx / "table_wheatherObservation.csv", index=False)
    pd.DataFrame({"id_station": [1, 2], "station_name": ["a", "b"],
                  "latitude": [48.0, 49.0], "longitude": [-4.0, -5.0],
                  "elevation": [1, 2]}).to_csv(
        wx / "table_weatherStation.csv", index=False)
    pd.DataFrame({"id_windDirection": range(1, 20)}).to_csv(
        wx / "table_windDirection.csv", index=False)

    oc = root / "E1_ocean"
    oc.mkdir()
    cells = [(-5.0, 48.0, 100.0), (-5.0, 48.0333, -16384.0), (-4.9667, 48.0, 50.0)]
    rows = [(lon, lat, dpt, 0.5 if dpt > 0 else -327.67, 1.5 if dpt > 0 else -65.534,
             60 if dpt > 0 else -32767, 200.0 if dpt > 0 else -3276.7, T0 + 10_800 * k)
            for lon, lat, dpt in cells for k in range(3)]
    pd.DataFrame(rows, columns=["lon", "lat", "dpt", "wlv", "hs", "lm", "dir", "ts"]
                 ).to_csv(oc / "oc_november.csv", index=False)

    coast = root / "C2_European_Coastline"
    coast.mkdir()
    shapes = gpd.GeoSeries([box(-5, 47, -3, 48), box(5, 40, 6, 41)],
                           crs=4326).to_crs(3035)
    gpd.GeoDataFrame({"Shape_Leng": [1.0, 2.0]}, geometry=shapes.values, crs=3035
                     ).to_file(coast / "Europe Coastline (Polygone).shp")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    base = tmp_path_factory.mktemp("archive")
    raw, out = base / "raw", base / "out"
    write_raw_root(raw)
    report = preprocess_archive(raw, out, window=30, log=None)
    return raw, out, report


def test_preprocess_archive_writes_every_dataset(processed):
    _, out, _ = processed
    for name in ("AIS/ais.parquet", "AIS/mmsi_check.csv", "Weather/weather.parquet",
                 "Ocean/ocean_november.parquet", "Coastline/coastline.parquet",
                 "preprocessing_report.csv", "preprocessing_report.txt"):
        assert (out / name).exists(), name
    assert not list(out.rglob("*.part"))


def test_preprocess_archive_ais(processed):
    _, out, report = processed
    ais = pd.read_parquet(out / "AIS" / "ais.parquet")
    assert {"mmsi", "unix_time", "lon", "lat", "x", "y", "speed_over_ground",
            "vessel_type", "ship_type_code", "navigational_status"} <= set(ais.columns)
    assert set(ais.mmsi) == {SHIP, SHIP + 1}
    assert ais.groupby("mmsi").vessel_type.first().to_dict() == {
        SHIP: "Fishing", SHIP + 1: "Cargo"}
    assert ais.sort_values(["mmsi", "unix_time"]).index.equals(ais.index)
    assert not ais.duplicated(["mmsi", "unix_time"]).any()
    assert len(ais) == 62                                      # 60 + the two kept rows
    jump = bad(report, "drop_position_jumps", "isolated jump above 50 knots")
    assert jump.rows_dropped == 1
    row = ais[(ais.mmsi == SHIP) & (ais.unix_time == T0 + 600)].iloc[0]
    assert np.isnan(row.true_heading) and np.isnan(row.rate_of_turn)
    row = ais[ais.unix_time == T0 + 700].iloc[0]
    assert np.isnan(row.speed_over_ground) and np.isnan(row.navigational_status)
    ids = pd.read_csv(out / "AIS" / "mmsi_check.csv")
    assert "placeholder" in set(ids.mmsi_class)
    assert bad(report, "check_ranges", "navigational_status").values_to_nan == 1
    assert bad(report, "unavailable_to_nan", "speed_over_ground").values_to_nan == 2
    assert bad(report, "drop_invalid_mmsi", "mmsi: placeholder").rows_dropped == 1
    valid = bad(report, "drop_invalid_position", "lon, lat in valid range")
    assert valid.rows_dropped == 1
    assert bad(report, "drop_duplicates_and_sort", "duplicate").rows_dropped == 1


def test_preprocess_archive_weather_ocean_coastline(processed):
    import geopandas as gpd

    _, out, report = processed
    w = pd.read_parquet(out / "Weather" / "weather.parquet")
    assert len(w) == 8 and w.air_temperature.isna().sum() == 2
    assert w.air_pressure.isna().sum() == 1                    # 2000 mm Hg
    assert {"x", "y", "wind_speed", "station_id"} <= set(w.columns)
    oc = pd.read_parquet(out / "Ocean" / "ocean_november.parquet")
    assert len(oc) == 6                                        # the land cell is gone
    assert oc.sea_level.notna().all()
    coast = gpd.read_parquet(out / "Coastline" / "coastline.parquet")
    assert coast.crs.to_epsg() == 32630 and len(coast) == 1
    assert bad(report, "drop_land_cells", "bottom_depth > -16000").rows_dropped == 3
    temp = bad(report, "weather_unavailable_to_nan", "air_temperature")
    assert temp.values_to_nan == 2
    rep = pd.read_csv(out / "preprocessing_report.csv")
    assert list(rep.columns) == PreprocessingReport.COLUMNS
    assert {"AIS", "weather", "ocean", "coastline"} == set(rep.dataset)


def test_preprocess_archive_skips_absent_and_existing(processed, tmp_path):
    raw, out, _ = processed
    messages = []
    again = preprocess_archive(raw, out, log=messages.append)
    assert len(again) == 0 and any("exists" in m for m in messages)
    assert any("absent" in m for m in messages)                # october, december, ...
    empty = preprocess_archive(tmp_path / "nothing", tmp_path / "out2", log=None)
    assert len(empty) == 0 and not (tmp_path / "out2").exists()
    with pytest.raises(ValueError):
        preprocess_archive(raw, out, window=0)


# ---- the committed sample ---------------------------------------------------------------------

@pytest.mark.data
def test_committed_preprocessed_sample():
    import geopandas as gpd

    root = ROOT / "data" / "sample" / "preprocessed"
    if not (root / "AIS" / "ais.parquet").exists():
        pytest.skip("data/sample/preprocessed not present")
    t0 = int(pd.Timestamp("2015-11-09", tz="UTC").timestamp())
    t1 = int(pd.Timestamp("2015-11-11", tz="UTC").timestamp())
    ais = pd.read_parquet(root / "AIS" / "ais.parquet")
    assert {"mmsi", "unix_time", "lon", "lat", "x", "y", "speed_over_ground",
            "course_over_ground", "true_heading", "navigational_status",
            "rate_of_turn", "vessel_type", "ship_type_code"} <= set(ais.columns)
    assert ais.unix_time.min() >= t0 and ais.unix_time.max() < t1
    assert ais.unix_time.min() < t0 + 3600 and ais.unix_time.max() > t1 - 3600
    assert ais.mmsi.nunique() > 500
    assert ais.x.between(-1e6, 1e6).all() and ais.y.between(5e6, 5.7e6).all()

    w = pd.read_parquet(root / "Weather" / "weather.parquet")
    assert {"station_id", "unix_time", "wind_speed", "air_temperature", "x",
            "y"} <= set(w.columns)
    assert w.station_id.nunique() == 16 and w.unix_time.min() >= t0
    assert w.unix_time.max() < t1

    oc = pd.read_parquet(root / "Ocean" / "ocean_november.parquet")
    assert {"lon", "lat", "bottom_depth", "sea_level", "wave_height", "wave_length",
            "wave_direction", "unix_time"} <= set(oc.columns)
    assert oc.unix_time.min() <= t0 and oc.unix_time.max() >= t1   # the window + a step

    coast = gpd.read_parquet(root / "Coastline" / "coastline.parquet")
    assert coast.crs.to_epsg() == 32630 and len(coast) > 0
    assert (root / "preprocessing_report.csv").exists()


def test_ocean_file_of_another_month_is_skipped(tmp_path):
    # The archive's oc_january.csv repeats November: it must not be written as
    # January.
    from catraj.preprocessing import ocean_month_matches, preprocess_archive
    nov = pd.Timestamp("2015-11-15", tz="UTC").timestamp()
    oc = pd.DataFrame({"unix_time": [nov, nov + 10800]})
    assert ocean_month_matches(oc, "november")
    assert not ocean_month_matches(oc, "january")
    folder = tmp_path / "raw" / "E1_ocean"
    folder.mkdir(parents=True)
    (folder / "oc_january.csv").write_text(
        "lon,lat,dpt,wlv,hs,lm,dir,ts\n"
        f"-4.5,48.4,100,0.5,1.0,50,180,{int(nov)}\n")
    report = preprocess_archive(tmp_path / "raw", tmp_path / "out",
                                months=("january",))
    assert not (tmp_path / "out" / "Ocean" / "ocean_january.parquet").exists()
    assert "file skipped" in report.to_text()
