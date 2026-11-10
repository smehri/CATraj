import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.annotation import DLT, IDW, Spline, annotate, annotate_points, build_series
from catraj.annotation.annotate import _times_of

from .conftest import linear_field

EPOCH0 = pd.Timestamp("2016-02-05 01:00")
UNIX0 = 1454634000.0  # EPOCH0 in UNIX seconds


def _track(n=12):
    """A straight track inside the station grid, crossing the epochs."""
    x = np.linspace(2_000.0, 28_000.0, n)
    y = np.linspace(3_000.0, 25_000.0, n)
    t = np.linspace(0.0, 5 * 3600.0, n)
    return x, y, t


def test_annotate_points_recovers_a_known_field(station_frame):
    # The field is a plane in space and linear in time, so TIN-free methods
    # still recover it exactly at a station-free position when the field is
    # spatially constant per epoch...
    x, y, t = _track()
    flat = station_frame.assign(value=3.0 + 0.01 * station_frame.t)
    out = annotate_points(flat, x, y, t, spatial_method=IDW(), temporal_method=DLT())
    assert out == pytest.approx(3.0 + 0.01 * t)


def test_annotate_points_with_exact_methods_recovers_a_plane(station_frame):
    from catraj.annotation import TIN

    x, y, t = _track()
    out = annotate_points(station_frame, x, y, t, spatial_method=TIN(), temporal_method=DLT())
    assert out == pytest.approx(linear_field(x, y, t), abs=1e-6)


def test_grouping_positions_changes_nothing(station_frame):
    x, y, t = _track()
    x = np.r_[x, x[:4]]  # repeated positions, as at anchor
    y = np.r_[y, y[:4]]
    t = np.r_[t, t[:4] + 1800.0]
    a = annotate_points(station_frame, x, y, t, temporal_method=Spline())
    b = annotate_points(
        station_frame, x, y, t, temporal_method=Spline(), group_positions=False
    )
    assert a == pytest.approx(b)


def test_prototypes_are_not_refitted_in_place(station_frame):
    spatial, temporal = IDW(), DLT()
    annotate_points(station_frame, *_track(), spatial_method=spatial, temporal_method=temporal)
    assert spatial._fitted is False and temporal._fitted is False


def test_annotate_points_edge_cases(station_frame):
    assert annotate_points(station_frame, [], [], []).size == 0
    with pytest.raises(ValueError):
        annotate_points(station_frame, [0.0, 1.0], [0.0], [0.0])


def test_build_series_gives_one_value_per_epoch(station_frame):
    times, values = build_series(station_frame, 10_000.0, 10_000.0)
    assert times == pytest.approx(np.arange(6) * 3600.0)
    assert values == pytest.approx(linear_field(10_000.0, 10_000.0, times))


# ---------------------------------------------------------- time handling


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_times_of_is_unix_seconds_whatever_the_resolution(unit):
    # pandas 3 infers datetime resolution; the integers behind an index are
    # not always nanoseconds, so the conversion must not assume they are.
    index = pd.DatetimeIndex([EPOCH0, EPOCH0 + pd.Timedelta(hours=1)]).as_unit(unit)
    assert _times_of(pd.DataFrame(index=index)) == pytest.approx([UNIX0, UNIX0 + 3600])


def test_times_of_takes_aware_indexes_in_utc():
    index = pd.DatetimeIndex([EPOCH0]).tz_localize("UTC").tz_convert("Asia/Tehran")
    assert _times_of(pd.DataFrame(index=index)) == pytest.approx([UNIX0])


def test_times_of_passes_numeric_indexes_through():
    assert _times_of(pd.DataFrame(index=[1.0, 2.5])) == pytest.approx([1.0, 2.5])


def _geoframe(station_frame):
    x, y, t = _track()
    gdf = gpd.GeoDataFrame(
        {"id": 1},
        geometry=gpd.points_from_xy(x, y),
        index=pd.DatetimeIndex(EPOCH0 + pd.to_timedelta(t, unit="s")).as_unit("us"),
        crs=32630,
    )
    stations = station_frame.assign(t=station_frame.t + UNIX0)
    expected = linear_field(x, y, t)
    return gdf, stations, expected


def test_annotate_writes_a_column_on_a_geodataframe(station_frame):
    from catraj.annotation import TIN

    gdf, stations, expected = _geoframe(station_frame)
    out = annotate(gdf, stations, "ctx", spatial_method=TIN(), temporal_method=DLT())
    assert out is gdf
    assert gdf["ctx"].to_numpy() == pytest.approx(expected, abs=1e-6)


@pytest.mark.movingpandas
def test_annotate_a_movingpandas_trajectory(station_frame):
    mpd = pytest.importorskip("movingpandas")
    from catraj.annotation import TIN

    gdf, stations, expected = _geoframe(station_frame)
    traj = mpd.Trajectory(gdf, traj_id=1)
    annotate(traj, stations, "ctx", spatial_method=TIN(), temporal_method=DLT())
    assert traj.df["ctx"].to_numpy() == pytest.approx(expected, abs=1e-6)
    # Context survives the .iloc slicing compression performs.
    assert traj.df.iloc[::3]["ctx"].to_numpy() == pytest.approx(expected[::3], abs=1e-6)
