import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.annotation import DLT, IDW, Spline, annotate, annotate_points, build_series
from catraj.annotation.annotate import _times_of

from helpers import linear_field

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


# ------------------------------------------------- large inputs: speed paths


class _LoopDLT(DLT):
    """A DLT subclass, so annotate_points takes the position-by-position path."""


def _gappy_stations(seed=0, n_epochs=40):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(4) * 10_000.0, np.arange(4) * 10_000.0)
    rows = []
    for e in range(n_epochs):
        v = 10 + 3 * np.sin(e / 5 + xs.ravel() / 20_000) + rng.normal(0, 0.3, 16)
        v[rng.random(16) < 0.15] = np.nan          # stations missing now and then
        rows.append(pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(),
                                  "t": e * 3600.0, "value": v}))
    st = pd.concat(rows, ignore_index=True)
    st_all_gone = st.t == 7 * 3600.0                 # one epoch with no station at all
    st.loc[st_all_gone, "value"] = np.nan
    return st


def _many_points(n=3000, seed=1, span=39 * 3600.0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-5_000, 35_000, n), rng.uniform(-5_000, 35_000, n),
            np.sort(rng.uniform(-1800, span + 1800, n)))


@pytest.mark.parametrize("circular", [False, True])
def test_vectorised_dlt_equals_the_fit_by_fit_path(circular):
    st = _gappy_stations()
    if circular:
        st = st.assign(value=np.mod(st.value * 30, 360))
    x, y, t = _many_points()
    fast = annotate_points(st, x, y, t, spatial_method=IDW(circular=circular),
                           temporal_method=DLT(circular=circular), chunk_epochs=None)
    slow = annotate_points(st, x, y, t, spatial_method=IDW(circular=circular),
                           temporal_method=_LoopDLT(circular=circular), chunk_epochs=None)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)


@pytest.mark.parametrize("chunk", [1, 4, 9])
def test_chunked_dlt_is_identical_to_unchunked(chunk):
    st = _gappy_stations()
    x, y, t = _many_points()
    whole = annotate_points(st, x, y, t, temporal_method=DLT(), chunk_epochs=None)
    blocks = annotate_points(st, x, y, t, temporal_method=DLT(), chunk_epochs=chunk)
    np.testing.assert_allclose(blocks, whole, rtol=0, atol=1e-9)


def test_chunked_spline_stays_close_to_unchunked():
    st = _gappy_stations()
    x, y, t = _many_points(800)
    whole = annotate_points(st, x, y, t, temporal_method=Spline(), chunk_epochs=None)
    blocks = annotate_points(st, x, y, t, temporal_method=Spline(), chunk_epochs=6)
    assert np.nanmax(np.abs(blocks - whole)) < 0.5
    assert np.isfinite(blocks).sum() == np.isfinite(whole).sum()


def test_auto_chunking_leaves_small_inputs_alone(station_frame):
    x, y, t = _track()
    a = annotate_points(station_frame, x, y, t, temporal_method=Spline())
    b = annotate_points(station_frame, x, y, t, temporal_method=Spline(), chunk_epochs=None)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("temporal", [DLT, Spline])
def test_parallel_blocks_equal_serial(temporal):
    st = _gappy_stations()
    x, y, t = _many_points(1500)
    serial = annotate_points(st, x, y, t, temporal_method=temporal(), chunk_epochs=5)
    parallel = annotate_points(st, x, y, t, temporal_method=temporal(), chunk_epochs=5,
                               n_jobs=2)
    np.testing.assert_array_equal(parallel, serial)


def _warn_and_double(x):
    import warnings

    warnings.warn("noisy", UserWarning)
    return 2 * x


def test_parallel_workers_honour_the_callers_warning_filters():
    import warnings

    from catraj._parallel import run_parallel

    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # a warning would raise...
        warnings.filterwarnings("ignore", message="noisy")   # ...except this one
        assert run_parallel(_warn_and_double, [(1,), (2,), (3,)], n_jobs=2) == [2, 4, 6]


def test_n_jobs_is_validated():
    st = _gappy_stations()
    with pytest.raises(ValueError):
        annotate_points(st, *_many_points(50), chunk_epochs=5, n_jobs=0)


@pytest.mark.parametrize("bad", [0, -2, "sometimes"])
def test_chunk_epochs_is_validated(station_frame, bad):
    with pytest.raises(ValueError):
        annotate_points(station_frame, *_track(), chunk_epochs=bad)


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
