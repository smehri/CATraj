import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.compression import (
    Context,
    cadp_select,
    compress,
    compression_ratio,
    context_errors,
    douglas_peucker,
    dtw,
    nominal_boundaries,
    segment_distances,
)

from helpers import DATASETS


def _walk(n=400, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=50.0, size=(n, 2)) + [30.0, 5.0]
    xy = np.cumsum(steps, axis=0)
    return xy[:, 0], xy[:, 1], np.arange(n) * 10.0


# ------------------------------------------------------- Douglas-Peucker


def test_segment_distance_is_to_the_segment_not_the_line():
    d = segment_distances([5.0, 15.0, -3.0], [4.0, 0.0, 4.0], (0, 0), (10, 0))
    assert d == pytest.approx([4.0, 5.0, 5.0])
    assert segment_distances([3.0], [4.0], (0, 0), (0, 0)) == pytest.approx([5.0])


def test_straight_line_keeps_only_the_ends():
    x = np.linspace(0, 1000, 50)
    keep = douglas_peucker(x, 2 * x, epsilon=1e-6)
    assert np.flatnonzero(keep).tolist() == [0, 49]


def test_one_corner_is_kept():
    x = np.r_[np.linspace(0, 100, 11), np.full(10, 100.0)]
    y = np.r_[np.zeros(11), np.linspace(10, 100, 10)]
    assert np.flatnonzero(douglas_peucker(x, y, 1.0)).tolist() == [0, 10, 20]


@pytest.mark.parametrize("epsilon", [10.0, 80.0, 300.0])
def test_matches_geos_simplify(epsilon):
    shapely = pytest.importorskip("shapely")
    x, y, _ = _walk()
    keep = douglas_peucker(x, y, epsilon)
    ours = np.column_stack([x[keep], y[keep]])
    line = shapely.LineString(np.column_stack([x, y]))
    geos = np.asarray(shapely.simplify(line, epsilon, preserve_topology=False).coords)
    assert ours == pytest.approx(geos)


def test_epsilon_zero_keeps_every_non_collinear_point():
    x, y, _ = _walk(50)
    assert douglas_peucker(x, y, 0.0).all()


def test_dp_edge_cases():
    assert douglas_peucker([], [], 1.0).size == 0
    assert douglas_peucker([1.0, 2.0], [1.0, 2.0], 1.0).all()
    with pytest.raises(ValueError):
        douglas_peucker([0.0, np.nan, 1.0], [0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError):
        douglas_peucker([0.0, 1.0], [0.0, 1.0], -1.0)


def test_dp_handles_a_long_trajectory_without_recursion_limits():
    # A spiral forces a deep, one-sided split sequence.
    theta = np.linspace(0, 60 * np.pi, 100_000)
    x, y = theta * np.cos(theta), theta * np.sin(theta)
    keep = douglas_peucker(x, y, 0.01)
    assert keep[0] and keep[-1] and 1000 < keep.sum() < 100_000


# ------------------------------------------------------- context errors


def test_context_error_is_interpolation_residual():
    t = np.array([0.0, 10.0, 30.0, 40.0])
    c = np.array([0.0, 7.0, 3.0, 4.0])
    keep = np.array([True, False, False, True])
    # c_hat at t=10 is 1.0, at t=30 is 3.0
    assert context_errors(t, c, keep) == pytest.approx([0.0, 6.0, 0.0, 0.0])


def test_linear_context_change_adds_nothing():
    t = np.arange(20.0)
    assert context_errors(t, 5.0 + 0.3 * t, np.r_[True, np.zeros(18, bool), True]) \
        == pytest.approx(np.zeros(20))


def test_bearing_errors_go_through_north():
    t = np.array([0.0, 10.0, 20.0])
    keep = np.array([True, False, True])
    # 350 -> 10 through north: the midpoint is 0, observed 0 -> no error.
    assert context_errors(t, [350.0, 0.0, 10.0], keep, circular=True)[1] == pytest.approx(0.0)
    # Read as plain numbers the midpoint would be 180: a 180-unit error.
    assert context_errors(t, [350.0, 0.0, 10.0], keep)[1] == pytest.approx(180.0)
    # A real deviation: 30 degrees off the interpolated 0.
    assert context_errors(t, [350.0, 330.0, 10.0], keep, circular=True)[1] == pytest.approx(30.0)


def test_missing_context_gives_no_error():
    t = np.arange(5.0)
    keep = np.array([True, False, False, False, True])
    err = context_errors(t, [0.0, np.nan, 9.0, 3.0, 4.0], keep)
    assert np.isnan(err[1]) and err[2] == pytest.approx(7.0)
    assert np.isnan(context_errors(t, [np.nan, 1, 2, 3, 4.0], keep)[1:4]).all()


def test_shared_timestamp_compares_with_the_midpoint():
    err = context_errors([5.0, 5.0, 5.0], [0.0, 4.0, 2.0], [True, False, True])
    assert err[1] == pytest.approx(3.0)


def test_context_errors_validate_input():
    with pytest.raises(ValueError):
        context_errors([0.0, 1.0, 2.0], [1, 2, 3], [False, False, True])
    with pytest.raises(ValueError):
        context_errors([0.0, 2.0, 1.0], [1, 2, 3], [True, False, True])


# --------------------------------------------------------------- nominal


def test_both_sides_of_a_category_change_are_kept():
    flagged = nominal_boundaries(["move", "move", "move", "anchor", "anchor", "move"])
    assert np.flatnonzero(flagged).tolist() == [2, 3, 4, 5]


def test_missing_category_counts_as_a_change():
    flagged = nominal_boundaries([1, 1, np.nan, np.nan, 1])
    assert np.flatnonzero(flagged).tolist() == [1, 2, 3, 4]


# ------------------------------------------------------------------ CADP


def test_no_context_is_plain_dp():
    x, y, t = _walk()
    sel = cadp_select(x, y, t, 100.0)
    assert (sel.mask == douglas_peucker(x, y, 100.0)).all()
    assert not sel.added.any()


def test_cadp_adds_the_point_where_context_jumps():
    x = np.linspace(0, 10_000, 101)  # a straight leg: DP keeps only the ends
    y = np.zeros(101)
    t = np.arange(101) * 60.0
    depth = np.full(101, 50.0)
    depth[40:] = 20.0  # shoaling halfway along
    sel = cadp_select(x, y, t, 1_000.0, contexts=[Context(depth, 5.0, name="depth")])
    assert sel.dp.sum() == 2
    assert sel.mask[40] or sel.mask[39]
    assert "depth" in sel.summary()


def test_cadp_is_a_superset_of_dp_and_compresses_less():
    x, y, t = _walk(seed=4)
    speed = np.abs(np.random.default_rng(4).normal(10, 3, x.size))
    dp = douglas_peucker(x, y, 150.0)
    sel = cadp_select(x, y, t, 150.0, contexts=[Context(speed, 4.0)])
    assert (sel.mask >= dp).all()
    assert compression_ratio(sel.mask.sum(), x.size) <= compression_ratio(dp.sum(), x.size)


def test_a_huge_threshold_changes_nothing():
    x, y, t = _walk(seed=5)
    sel = cadp_select(x, y, t, 150.0, contexts=[Context(np.sin(t), 1e9)])
    assert (sel.mask == sel.dp).all()


def test_several_variables_combine_with_or():
    x, y, t = np.linspace(0, 1000, 11), np.zeros(11), np.arange(11.0)
    a = np.zeros(11)
    a[3] = 10.0
    b = np.zeros(11)
    b[7] = 10.0
    sel = cadp_select(x, y, t, 50.0, contexts=[Context(a, 1.0, name="a"), Context(b, 1.0, name="b")])
    assert np.flatnonzero(sel.added).tolist() == [3, 7]
    assert np.flatnonzero(sel.context["a"]).tolist() == [3]


def test_threshold_is_strict():
    x, y, t = np.linspace(0, 1000, 3), np.zeros(3), np.arange(3.0)
    c = np.array([0.0, 2.0, 0.0])
    assert not cadp_select(x, y, t, 50.0, contexts=[Context(c, 2.0)]).added.any()
    assert cadp_select(x, y, t, 50.0, contexts=[Context(c, 1.999)]).added[1]


def test_nominal_boundaries_join_the_union():
    x, y, t = np.linspace(0, 1000, 8), np.zeros(8), np.arange(8.0)
    status = ["under way"] * 4 + ["anchored"] * 4
    sel = cadp_select(x, y, t, 50.0, nominal={"status": status})
    assert np.flatnonzero(sel.mask).tolist() == [0, 3, 4, 7]


def test_bad_threshold_is_rejected():
    x, y, t = _walk(20)
    with pytest.raises(ValueError):
        cadp_select(x, y, t, 10.0, contexts=[Context(t, -1.0)])


# ------------------------------------------------------- iterative mode


def _bow(n=201):
    """A straight leg (DP keeps only the ends) under a smooth context bump."""
    x, y, t = np.linspace(0, 20_000, n), np.zeros(n), np.arange(n) * 60.0
    return x, y, t, 10.0 + 3.0 * np.sin(np.pi * np.arange(n) / (n - 1))


def test_iterative_keeps_the_apex_where_post_keeps_the_whole_bow():
    x, y, t, c = _bow()
    post = cadp_select(x, y, t, 1_000.0, contexts=[Context(c, 0.5)])
    it = cadp_select(x, y, t, 1_000.0, contexts=[Context(c, 0.5)], mode="iterative")
    assert post.added.sum() > 100
    assert 0 < it.added.sum() < 10
    assert it.mask[100]  # the apex
    assert it.rounds >= 1 and "iterative mode" in it.summary()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_iterative_bounds_the_error_of_the_result(seed):
    x, y, t = _walk(seed=seed)
    rng = np.random.default_rng(seed)
    speed = 10 + np.cumsum(rng.normal(0, 0.5, x.size))
    bearing = np.mod(np.cumsum(rng.normal(0, 4, x.size)), 360)
    ctx = [Context(speed, 1.0, name="speed"), Context(bearing, 10.0, True, "bearing")]
    sel = cadp_select(x, y, t, 150.0, contexts=ctx, mode="iterative")
    assert context_errors(t, speed, sel.mask).max() <= 1.0
    assert context_errors(t, bearing, sel.mask, circular=True).max() <= 10.0


def test_post_mode_does_not_bound_the_result():
    # The documented difference: measured against its final neighbours, a
    # point post mode dropped can exceed the threshold.
    worst = 0.0
    for seed in range(10):
        x, y, t = _walk(seed=seed)
        c = 10 + np.cumsum(np.random.default_rng(seed).normal(0, 0.5, x.size))
        sel = cadp_select(x, y, t, 150.0, contexts=[Context(c, 1.0)])
        worst = max(worst, context_errors(t, c, sel.mask).max())
    assert worst > 1.0


def _context_dp(t, c, eps, first, last, keep):
    """Reference: Douglas-Peucker on the context curve alone, recursively."""
    if last - first < 2:
        return
    seg = np.zeros(t.size, bool)
    seg[[first, last]] = True
    sub = slice(first, last + 1)
    err = context_errors(t[sub], c[sub], seg[sub])
    k = int(np.argmax(err))
    if err[k] > eps:
        keep[first + k] = True
        _context_dp(t, c, eps, first, first + k, keep)
        _context_dp(t, c, eps, first + k, last, keep)


def test_iterative_is_douglas_peucker_on_the_context_curve():
    x, y, t, _ = _bow()
    c = np.random.default_rng(7).normal(0, 1, t.size).cumsum()
    sel = cadp_select(x, y, t, 1_000.0, contexts=[Context(c, 0.8)], mode="iterative")
    reference = np.zeros(t.size, bool)
    reference[[0, -1]] = True
    _context_dp(t, c, 0.8, 0, t.size - 1, reference)
    assert (sel.mask == reference).all()


def test_iterative_attributes_each_point_to_one_variable():
    x, y, t = np.linspace(0, 1000, 11), np.zeros(11), np.arange(11.0)
    a, b = np.zeros(11), np.zeros(11)
    a[3], b[7] = 10.0, 10.0
    sel = cadp_select(x, y, t, 50.0, mode="iterative",
                      contexts=[Context(a, 1.0, name="a"), Context(b, 1.0, name="b")])
    # A one-point spike needs its foot points too: once the apex is kept, the
    # line up to it misrepresents the zeros beside it.
    assert np.flatnonzero(sel.context["a"]).tolist() == [2, 3, 4]
    assert np.flatnonzero(sel.context["b"]).tolist() == [6, 7, 8]
    assert not (sel.context["a"] & sel.context["b"]).any()
    # Post mode keeps only the apex -- and so leaves point 2 off by 6.7.
    post = cadp_select(x, y, t, 50.0, contexts=[Context(a, 1.0)])
    assert np.flatnonzero(post.added).tolist() == [3]
    assert context_errors(t, a, post.mask)[2] == pytest.approx(20 / 3)


def test_iterative_measures_against_nominal_boundaries():
    x, y, t = np.linspace(0, 1000, 9), np.zeros(9), np.arange(9.0)
    c = np.array([0, 1, 2, 3, 8, 6, 4, 2, 0], float)  # linear either side
    status = ["a"] * 4 + ["b"] * 5  # boundary keeps points 3 and 4
    sel = cadp_select(x, y, t, 50.0, contexts=[Context(c, 0.5)],
                      nominal={"s": status}, mode="iterative")
    # Measured against the boundary points, both sides are exactly linear.
    assert np.flatnonzero(sel.mask).tolist() == [0, 3, 4, 8]
    assert not any(v.any() for v in sel.context.values())


def test_iterative_with_huge_threshold_is_dp_and_zero_threshold_keeps_all_changes():
    x, y, t, c = _bow(51)
    assert (cadp_select(x, y, t, 1e3, contexts=[Context(c, 1e9)], mode="iterative").mask
            == douglas_peucker(x, y, 1e3)).all()
    assert cadp_select(x, y, t, 1e3, contexts=[Context(c, 0.0)], mode="iterative").mask.all()


def test_unknown_mode_is_rejected():
    x, y, t = _walk(10)
    with pytest.raises(ValueError):
        cadp_select(x, y, t, 1.0, mode="recursive")


# ---------------------------------------------------------- dataframes


def _frame(n=60, crs=32630):
    x = np.linspace(0, 6_000, n)
    y = 200.0 * np.sin(x / 2_000.0)
    wave = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    return gpd.GeoDataFrame(
        {"wave_height": wave, "wave_dir": np.full(n, 355.0), "vid": 1},
        geometry=gpd.points_from_xy(x, y),
        index=pd.date_range("2016-02-05", periods=n, freq="30s"),
        crs=crs,
    )


def test_compress_a_geodataframe_keeps_every_column():
    gdf = _frame()
    out = compress(gdf, 50.0, thresholds={"wave_height": 0.5, "wave_dir": 2.0},
                   circular=["wave_dir"])
    assert list(out.columns) == list(gdf.columns)
    assert len(out) < len(gdf)
    assert out.index.isin(gdf.index).all()
    jump = gdf.index[len(gdf) // 2]
    assert jump in out.index or gdf.index[len(gdf) // 2 - 1] in out.index


def test_compress_by_group():
    a, b = _frame(), _frame()
    b["vid"] = 2
    out = compress(pd.concat([a, b]), 50.0, by="vid")
    assert set(out.vid) == {1, 2}
    assert len(out[out.vid == 1]) == len(compress(a, 50.0))


def test_compress_refuses_geographic_coordinates():
    with pytest.raises(ValueError, match="geographic"):
        compress(_frame(crs=4326), 50.0)


def test_circular_column_needs_a_threshold():
    with pytest.raises(ValueError):
        compress(_frame(), 50.0, circular=["wave_dir"])


@pytest.mark.movingpandas
def test_compress_movingpandas_objects():
    mpd = pytest.importorskip("movingpandas")
    traj = mpd.Trajectory(_frame(), traj_id="v1")
    out = compress(traj, 50.0, thresholds={"wave_height": 0.5})
    assert isinstance(out, mpd.Trajectory) and out.id == "v1"
    assert "wave_height" in out.df.columns and len(out.df) < len(traj.df)

    coll = mpd.TrajectoryCollection([traj, mpd.Trajectory(_frame(), traj_id="v2")])
    out_coll = compress(coll, 50.0, thresholds={"wave_height": 0.5})
    assert len(out_coll.trajectories) == 2
    out_it = compress(coll, 50.0, thresholds={"wave_height": 0.5}, mode="iterative")
    assert len(out_it.trajectories) == 2


def test_compress_passes_the_mode_through():
    gdf = _frame()
    post = compress(gdf, 50.0, thresholds={"wave_height": 0.1})
    it = compress(gdf, 50.0, thresholds={"wave_height": 0.1}, mode="iterative")
    assert len(it) <= len(post)
    assert len(compress(pd.concat([gdf, gdf.assign(vid=2)]), 50.0, by="vid",
                        thresholds={"wave_height": 0.1}, mode="iterative")) == 2 * len(it)


# ------------------------------------------------------------ evaluation


def test_compression_ratio():
    assert compression_ratio(25, 100) == pytest.approx(75.0)
    with pytest.raises(ValueError):
        compression_ratio(5, 4)


def _dtw_brute(a, b):
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = np.hypot(*(a[i - 1] - b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dtw_matches_the_textbook_recurrence(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(23, 2)), rng.normal(size=(9, 2))
    assert dtw(a, b) == pytest.approx(_dtw_brute(a, b))
    assert dtw(b, a) == pytest.approx(dtw(a, b))


def test_dtw_of_identical_sequences_is_zero_and_accepts_frames():
    gdf = _frame()
    assert dtw(gdf, gdf) == pytest.approx(0.0)
    assert dtw(gdf, compress(gdf, 50.0)) > 0


# ----------------------------------------------------------- sample data


@pytest.mark.data
def test_cadp_on_an_annotated_sample_vessel():
    from catraj.annotation import IDW, Spline, annotate_points

    ais = gpd.read_file(DATASETS / "AIS" / "AIS_DLT_UTM.shp")
    weather = gpd.read_file(DATASETS / "Weather" / "Weather_Observation_UTM.shp")
    stations = pd.DataFrame({
        "x": weather.geometry.x, "y": weather.geometry.y,
        "t": weather.local_time.astype(float), "value": weather.Ff.astype(float),
    })
    track = ais[ais.sourcemmsi == ais.sourcemmsi.value_counts().index[1]].sort_values("t")
    track = track.assign(Ff=annotate_points(
        stations, track.geometry.x, track.geometry.y, track.t.astype(float),
        spatial_method=IDW(), temporal_method=Spline()))

    dp = compress(track, 1_000.0, time_column="t")
    cadp = compress(track, 1_000.0, thresholds={"Ff": 0.2, "speedoverg": 2.0},
                    time_column="t")
    assert len(dp) <= len(cadp) < len(track)
    assert set(dp.index) <= set(cadp.index)
    assert dtw(track, cadp) <= dtw(track, dp)
