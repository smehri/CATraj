import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.preprocessing import denoise, rolling_filter

from helpers import DATASETS


# ------------------------------------------------------------- references


def _slow(t, values, window, method):
    """The definition, point by point."""
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    flat = values.reshape(t.size, -1)
    out = np.full(flat.shape, np.nan)
    for i in range(t.size):
        near = np.abs(t - t[i]) <= window / 2
        for k in range(flat.shape[1]):
            got = flat[near, k]
            got = got[~np.isnan(got)]
            if got.size and not np.isnan(flat[i, k]):
                out[i, k] = np.median(got) if method == "median" else got.mean()
    return out.reshape(values.shape)


def _track(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(0, 5, n)).astype(float)  # irregular, repeats
    return t, rng


# ------------------------------------------------------------ array level


@pytest.mark.parametrize("method", ["median", "mean"])
@pytest.mark.parametrize("window", [1, 4, 7.5, 20, 1000])
@pytest.mark.parametrize("shape", [(), (3,)])
def test_rolling_filter_matches_the_definition(method, window, shape):
    t, rng = _track(60, seed=3)
    values = rng.normal(size=(t.size,) + shape)
    values[rng.random(values.shape) < 0.2] = np.nan
    fast = rolling_filter(t, values, window, method=method)
    assert fast.shape == values.shape
    np.testing.assert_allclose(
        fast, _slow(t, values, window, method), rtol=0, atol=1e-12,
        equal_nan=True,
    )


def test_rolling_filter_blocks_do_not_change_the_answer(monkeypatch):
    import catraj.preprocessing.filtering as module

    t, rng = _track(300, seed=5)
    values = rng.normal(size=(t.size, 2))
    whole = rolling_filter(t, values, 30.0)
    monkeypatch.setattr(module, "_BLOCK_VALUES", 50)
    np.testing.assert_array_equal(rolling_filter(t, values, 30.0), whole)


def test_a_median_removes_a_single_spike_and_a_mean_spreads_it():
    t = np.arange(7.0)
    x = np.zeros(7)
    x[3] = 70.0
    median = rolling_filter(t, x, 4.0)  # two points either side
    mean = rolling_filter(t, x, 4.0, method="mean")
    assert np.array_equal(median, np.zeros(7))
    # Points 2..4 average five points; 1 and 5 are cut at the ends of the
    # spike's reach (four points); 0 and 6 never see it.
    assert np.array_equal(mean, [0.0, 17.5, 14.0, 14.0, 14.0, 17.5, 0.0])


def test_the_window_is_in_time_not_in_points():
    t = np.array([0.0, 1.0, 61.0])
    x = np.array([0.0, 10.0, 100.0])
    assert np.array_equal(rolling_filter(t, x, 4.0, method="mean"),
                          [5.0, 5.0, 100.0])
    assert np.array_equal(rolling_filter(t, x, 60.0, method="mean"),
                          [5.0, 5.0, 100.0])
    assert np.allclose(rolling_filter(t, x, 122.0, method="mean"), [110 / 3] * 3)


def test_a_gap_longer_than_half_the_window_keeps_the_sides_apart():
    t = np.array([0, 1, 2, 100, 101, 102], dtype=float)
    x = np.array([0, 0, 0, 50, 50, 50], dtype=float)
    assert np.array_equal(rolling_filter(t, x, 90.0, method="mean"), x)
    assert np.allclose(rolling_filter(t, x, 210.0, method="mean"), 25.0)


def test_the_window_is_inclusive_at_half_its_width():
    t = np.array([0.0, 2.0, 4.0])
    x = np.array([0.0, 6.0, 12.0])
    assert np.array_equal(rolling_filter(t, x, 4.0, method="mean"),
                          [3.0, 6.0, 9.0])


def test_the_ends_are_truncated_not_padded():
    t = np.arange(5.0)
    x = np.array([0.0, 1.0, 2.0, 3.0, 40.0])
    out = rolling_filter(t, x, 4.0, method="mean")  # two either side
    assert out[0] == 1.0  # mean of 0, 1, 2
    assert out[4] == 45.0 / 3  # mean of 2, 3, 40
    assert rolling_filter(t, x, 4.0)[0] == 1.0  # median of 0, 1, 2


def test_repeated_times_fall_in_each_others_window():
    t = np.array([0.0, 0.0, 0.0, 10.0])
    x = np.array([1.0, 2.0, 9.0, 5.0])
    assert np.array_equal(rolling_filter(t, x, 2.0), [2.0, 2.0, 2.0, 5.0])


def test_missing_values_are_ignored_and_stay_missing():
    t = np.arange(5.0)
    v = np.array([1.0, np.nan, 3.0, 100.0, 5.0])
    out = rolling_filter(t, v, 2.0, method="mean")  # one either side
    assert out[0] == 1.0  # the missing neighbour is ignored
    assert np.isnan(out[1])  # no imputation
    assert out[2] == (3.0 + 100.0) / 2
    assert out[3] == (3.0 + 100.0 + 5.0) / 3


def test_an_empty_series_is_returned_empty():
    assert rolling_filter([], [], 5.0).shape == (0,)
    assert rolling_filter([], np.empty((0, 2)), 5.0).shape == (0, 2)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_a_window_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="window"):
        rolling_filter([0.0, 1.0], [0.0, 1.0], bad)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="method"):
        rolling_filter([0.0, 1.0], [0.0, 1.0], 2.0, method="mode")


def test_times_must_not_decrease_and_must_be_finite():
    with pytest.raises(ValueError, match="decrease"):
        rolling_filter([0.0, 2.0, 1.0], [0.0, 0.0, 0.0], 2.0)
    with pytest.raises(ValueError, match="finite"):
        rolling_filter([0.0, np.nan], [0.0, 0.0], 2.0)


def test_lengths_and_shapes_are_checked():
    with pytest.raises(ValueError, match="equal length"):
        rolling_filter([0.0, 1.0], [0.0], 2.0)
    with pytest.raises(ValueError, match="1-D"):
        rolling_filter([0.0], np.zeros((1, 1, 1)), 2.0)


# ------------------------------------------------------------- dataframes


def _frame(crs="EPSG:32630"):
    t = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3], dtype=float)
    x = np.array([0, 0, 0, 90, 0, 0, 0, 500, 500, 500, 500], dtype=float)
    return gpd.GeoDataFrame(
        {
            "mmsi": [1] * 7 + [2] * 4,
            "t": t,
            "sog": [5, 5, 5, 99, 5, 5, 5, 1, 1, 1, 1],
            "cog": np.arange(11, dtype=float),
            "lon": np.arange(11, dtype=float),
        },
        geometry=gpd.points_from_xy(x, np.zeros(11)),
        crs=crs,
    )


def test_denoise_keeps_rows_index_and_columns():
    df = _frame()
    df.index = pd.Index(np.arange(11)[::-1] * 10, name="row")
    out = denoise(df, 4.0, columns=["sog"], time_column="t", by="mmsi")
    assert isinstance(out, gpd.GeoDataFrame)
    assert out.crs == df.crs
    assert out.index.equals(df.index)
    assert list(out.columns) == list(df.columns)
    assert out.geometry.name == df.geometry.name
    assert len(out) == len(df)


def test_denoise_changes_geometry_and_named_columns_only():
    df = _frame()
    out = denoise(df, 4.0, columns=["sog"], time_column="t", by="mmsi")
    assert out.geometry.x.iloc[3] == 0.0 and df.geometry.x.iloc[3] == 90.0
    assert out["sog"].iloc[3] == 5.0 and df["sog"].iloc[3] == 99
    for untouched in ("mmsi", "t", "cog", "lon"):
        pd.testing.assert_series_equal(out[untouched], df[untouched])
    plain = denoise(df, 4.0, time_column="t", by="mmsi")
    pd.testing.assert_series_equal(plain["sog"], df["sog"])  # not named
    assert not np.allclose(plain.geometry.x, df.geometry.x)


def test_denoise_by_filters_each_track_alone_and_keeps_row_order():
    df = _frame()
    shuffled = df.iloc[[7, 0, 8, 1, 9, 2, 3, 10, 4, 5, 6]]
    out = denoise(shuffled, 4.0, time_column="t", by="mmsi")
    assert out.index.equals(shuffled.index)
    together = denoise(df, 4.0, time_column="t", by="mmsi")
    pd.testing.assert_series_equal(
        out.geometry.x.sort_index(), together.geometry.x.sort_index()
    )
    assert (out.geometry.x[out.mmsi == 2] == 500.0).all()  # not pulled to 0
    with pytest.raises(ValueError, match="decrease"):
        denoise(df, 4.0, time_column="t")  # two tracks read as one


def test_denoise_reads_time_from_the_index_by_default():
    one = _frame()
    one = one[one.mmsi == 1].copy()
    indexed = one.set_index(pd.to_datetime(one.t, unit="s")).drop(columns="t")
    a = denoise(one, 4.0, time_column="t")
    b = denoise(indexed, 4.0)
    assert np.array_equal(a.geometry.x.to_numpy(), b.geometry.x.to_numpy())
    assert b.index.equals(indexed.index)
    c = denoise(one.set_index("t", drop=False), 4.0, time_column="t")
    assert np.array_equal(a.geometry.x.to_numpy(), c.geometry.x.to_numpy())


def test_denoise_mean_and_median_differ_on_a_spike():
    one = _frame()
    one = one[one.mmsi == 1]
    med = denoise(one, 4.0, time_column="t")
    mean = denoise(one, 4.0, method="mean", time_column="t")
    assert med.geometry.x.iloc[3] == 0.0
    assert mean.geometry.x.iloc[3] == 18.0
    assert mean.geometry.x.iloc[2] == 18.0


def test_denoise_leaves_a_missing_value_missing():
    df = _frame()
    df.loc[3, "sog"] = np.nan
    out = denoise(df, 4.0, columns="sog", time_column="t", by="mmsi")
    assert np.isnan(out["sog"].iloc[3])
    assert out["sog"].iloc[2] == 5.0


def test_denoise_needs_finite_positions():
    df = _frame()
    df.loc[2, df.geometry.name] = gpd.points_from_xy([np.nan], [0.0])[0]
    with pytest.raises(ValueError, match="finite"):
        denoise(df, 4.0, time_column="t", by="mmsi")


def test_denoise_refuses_a_geographic_crs():
    with pytest.raises(ValueError, match="geographic"):
        denoise(_frame("EPSG:4326"), 4.0, time_column="t", by="mmsi")


def test_denoise_refuses_an_unknown_column_and_a_filtered_time_column():
    with pytest.raises(KeyError, match="nope"):
        denoise(_frame(), 4.0, columns=["nope"], time_column="t", by="mmsi")
    with pytest.raises(ValueError, match="time column"):
        denoise(_frame(), 4.0, columns=["t"], time_column="t", by="mmsi")


def test_denoise_validates_method_and_window():
    with pytest.raises(ValueError, match="method"):
        denoise(_frame(), 4.0, method="x", time_column="t", by="mmsi")
    with pytest.raises(ValueError, match="window"):
        denoise(_frame(), 0, time_column="t", by="mmsi")


def _mp_frame(track):
    df = _frame()
    one = df[df.mmsi == track].copy()
    one.index = pd.to_datetime(one.t, unit="s")
    return one.drop(columns="t")


@pytest.mark.movingpandas
def test_denoise_a_movingpandas_trajectory():
    mpd = pytest.importorskip("movingpandas")
    traj = mpd.Trajectory(_mp_frame(1), 1)
    out = denoise(traj, 4.0, columns=["sog"])
    assert isinstance(out, mpd.Trajectory)
    assert out.id == 1
    assert out.df.index.equals(traj.df.index)
    assert list(out.df.columns) == list(traj.df.columns)
    assert out.df.geometry.x.iloc[3] == 0.0
    assert out.df["sog"].iloc[3] == 5.0
    assert traj.df.geometry.x.iloc[3] == 90.0  # the input is not modified


@pytest.mark.movingpandas
def test_denoise_a_movingpandas_collection():
    mpd = pytest.importorskip("movingpandas")
    coll = mpd.TrajectoryCollection(
        [mpd.Trajectory(_mp_frame(i), i) for i in (1, 2)]
    )
    out = denoise(coll, 4.0)
    assert isinstance(out, mpd.TrajectoryCollection)
    assert [t.id for t in out.trajectories] == [1, 2]
    assert [len(t.df) for t in out.trajectories] == [7, 4]
    assert out.trajectories[0].df.geometry.x.iloc[3] == 0.0


# ------------------------------------------------------------- sample data


@pytest.mark.data
def test_denoise_on_the_sample_fleet():
    ais = gpd.read_file(DATASETS / "AIS" / "AIS_DLT_UTM.shp")
    ais = ais.sort_values(["sourcemmsi", "t"], kind="stable")
    ais["speedoverg"] = ais["speedoverg"].where(ais["speedoverg"] < 102.2)
    for method in ("median", "mean"):
        out = denoise(ais, 60.0, method=method, columns=["speedoverg"],
                      time_column="t", by="sourcemmsi")
        assert len(out) == len(ais)
        assert out.index.equals(ais.index)
        assert out.crs == ais.crs
        assert np.isfinite(out.geometry.x).all()
        assert np.isfinite(out.geometry.y).all()
        assert out["speedoverg"].isna().sum() == ais["speedoverg"].isna().sum()
        pd.testing.assert_series_equal(out["t"], ais["t"])


# ------------------------------------------------ bearings and the default


def _slow_bearings(t, deg, window, method):
    """Statistic of the sine and of the cosine, then atan2, point by point."""
    t = np.asarray(t, dtype=float)
    out = np.full(t.size, np.nan)
    stat = np.median if method == "median" else np.mean
    for i in range(t.size):
        if np.isnan(deg[i]):
            continue
        near = (np.abs(t - t[i]) <= window / 2) & ~np.isnan(deg)
        s = stat(np.sin(np.radians(deg[near])))
        c = stat(np.cos(np.radians(deg[near])))
        out[i] = np.degrees(np.arctan2(s, c)) % 360.0
    return out


@pytest.mark.parametrize("method", ["median", "mean"])
@pytest.mark.parametrize("seed", range(3))
def test_bearings_are_filtered_through_their_unit_vector(method, seed):
    t, rng = _track(60, seed=seed)
    deg = rng.uniform(-400, 760, t.size)
    deg[rng.random(t.size) < 0.15] = np.nan
    plain = rng.normal(size=t.size)
    got = rolling_filter(t, np.column_stack([plain, deg]), 8.0, method=method,
                         circular=[False, True])
    np.testing.assert_allclose(got[:, 0], _slow(t, plain, 8.0, method), atol=1e-12)
    want = _slow_bearings(t, deg, 8.0, method)
    seen = ~np.isnan(deg)
    gap = np.abs(got[seen, 1] - want[seen])
    assert np.isnan(got[~seen, 1]).all()
    assert np.minimum(gap, 360.0 - gap).max() < 1e-9
    assert ((got[seen, 1] >= 0) & (got[seen, 1] < 360)).all()


def test_a_bearing_across_north_is_not_averaged_through_south():
    t = np.arange(4.0)
    deg = np.array([359.0, 1.0, 359.0, 1.0])
    ordinary = rolling_filter(t, deg, 10.0, method="mean")
    bearing = rolling_filter(t, deg, 10.0, method="mean", circular=True)
    assert np.allclose(ordinary, 180.0)
    assert np.allclose(np.minimum(bearing, 360.0 - bearing), 0.0, atol=1e-9)


def test_denoise_filters_circular_columns_on_the_circle():
    df = _frame()
    df["cog"] = [359.0, 1.0, 359.0, 1.0, 359.0, 1.0, 359.0, 90.0, 90.0, 90.0, 90.0]
    out = denoise(df, 10.0, columns=["cog"], circular=["cog"], time_column="t",
                  by="mmsi")
    first = out.cog[:7].to_numpy()
    assert np.allclose(np.minimum(first, 360.0 - first), 0.0, atol=2.0)
    assert (out.cog[7:] == 90.0).all()
    ordinary = denoise(df, 10.0, columns=["cog"], time_column="t", by="mmsi")
    assert (ordinary.cog[:7] > 100).all()


def test_circular_columns_must_be_filtered_columns():
    with pytest.raises(ValueError, match="circular"):
        denoise(_frame(), 4.0, circular=["cog"], time_column="t", by="mmsi")


def test_the_default_window_is_30_seconds():
    one = _frame()
    one = one[one.mmsi == 1]
    a = denoise(one, time_column="t")
    b = denoise(one, 30, time_column="t")
    assert a.geometry.equals(b.geometry)
    assert not a.geometry.equals(denoise(one, 0.5, time_column="t").geometry)


def test_a_non_datetime_index_without_time_column_is_refused():
    one = _frame()
    one = one[one.mmsi == 1]
    with pytest.raises(ValueError, match="time_column"):
        denoise(one, 4.0)
    with pytest.raises(ValueError, match="filter_window=None"):
        denoise(one.set_index("t"), 4.0)
    assert len(denoise(one, 4.0, time_column="t")) == len(one)


def test_a_window_with_one_valid_value_returns_it_exactly():
    t = np.array([0.0, 100.0, 200.0, 300.0])
    deg = np.array([22.5, 123.456789, np.nan, 359.99999999])
    got = rolling_filter(t, np.column_stack([deg, deg]), 30.0,
                         circular=[True, False])
    assert np.array_equal(got[:, 0], deg, equal_nan=True)
    assert np.array_equal(got[:, 1], deg, equal_nan=True)
