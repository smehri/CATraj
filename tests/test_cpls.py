import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from catraj.compression import (
    Barriers,
    CplsSelection,
    cpls_compress,
    cpls_select,
    douglas_peucker,
    segment_crossings,
)

from helpers import DATASETS


def _reference_cpls(x, y, epsilon, barrier):
    """Algorithm 1 read literally: one intersection test per candidate point."""
    n = len(x)
    dp = douglas_peucker(x, y, epsilon)
    keep = dp.copy()

    def seg(i, j):
        return shapely.LineString([(x[i], y[i]), (x[j], y[j])])

    orig = [barrier.intersects(seg(i, i + 1)) for i in range(n - 1)]
    kept = np.flatnonzero(dp)
    for s, e in zip(kept[:-1], kept[1:]):
        a = s
        for j in range(s + 1, e):
            if barrier.intersects(seg(a, j + 1)) and not any(orig[a:j + 1]):
                keep[j] = True
                a = j
    return keep


def _walk(n=400, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=50.0, size=(n, 2)) + [30.0, 5.0]
    xy = np.cumsum(steps, axis=0)
    return xy[:, 0], xy[:, 1]


def _barriers_for(x, y, seed, count=25):
    """Buffered points and short lines scattered over the walk's extent."""
    rng = np.random.default_rng(100 + seed)
    xlo, xhi, ylo, yhi = x.min(), x.max(), y.min(), y.max()
    out = []
    for k in range(count):
        if k % 5 == 0:  # deliberately on the path
            i = int(rng.integers(0, x.size))
            cx, cy = x[i] + rng.normal(scale=20), y[i] + rng.normal(scale=20)
        else:
            cx, cy = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
        if k % 2:
            out.append(shapely.Point(cx, cy).buffer(rng.uniform(20, 120)))
        else:
            dx, dy = rng.normal(scale=150, size=2)
            out.append(shapely.LineString([(cx, cy), (cx + dx, cy + dy)]))
    return out


def _mask(x, y, epsilon, barriers):
    return cpls_select(x, y, epsilon, barriers).mask


def _idx(mask):
    return np.flatnonzero(mask).tolist()


# ------------------------------------------------------ hand-worked cases


def test_wall_keeps_the_point_that_clears_it():
    x = [0, 2, 4, 6, 8, 10]
    y = [0, 0, 3, 3, 0, 0]
    wall = shapely.LineString([(5, -1), (5, 2)])
    sel = cpls_select(x, y, 10, wall)
    assert _idx(sel.dp) == [0, 5]
    assert _idx(sel.mask) == [0, 3, 5]


def test_detour_around_a_wall_is_kept():
    x = np.array([0, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000,
                  7000, 8000, 9000, 10000], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 80, 100, 80, 0, 0, 0, 0, 0], dtype=float)
    wall = shapely.LineString([(5000, -50), (5000, 50)])
    sel = cpls_select(x, y, 500, wall)
    assert _idx(sel.dp) == [0, 12]
    assert _idx(sel.mask) == [0, 7, 12]
    assert segment_crossings(x, y, wall, keep=sel.dp).any()
    assert not segment_crossings(x, y, wall, keep=sel.mask).any()


def test_touching_a_barrier_counts_as_crossing():
    x, y = [0, 5, 10], [0, 1, 0]
    assert _idx(_mask(x, y, 10, shapely.Point(5, 0))) == [0, 1, 2]
    assert _idx(_mask(x, y, 10, shapely.Point(5, 0.5))) == [0, 2]


def test_an_existing_crossing_is_tolerated():
    x = np.linspace(-10, 20, 31)
    y = np.full(31, 5.0)
    sel = cpls_select(x, y, 1, shapely.box(0, 0, 10, 10))
    assert _idx(sel.dp) == [0, 30]
    assert _idx(sel.mask) == [0, 30]


# ------------------------------------------------------------ properties

SEEDS = range(5)
EPSILONS = [30.0, 150.0, 600.0]


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cpls_matches_the_literal_algorithm(seed, epsilon):
    x, y = _walk(seed=seed)
    parts = _barriers_for(x, y, seed)
    got = _mask(x, y, epsilon, parts)
    assert np.array_equal(got, _reference_cpls(x, y, epsilon, shapely.union_all(parts)))


def test_cpls_matches_the_literal_algorithm_when_the_last_segment_is_affected():
    # The detour of the hand-worked case, behind a far-away first point, so
    # that the wall sits in the last Douglas-Peucker segment.
    x = np.array([-3000, 0, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000,
                  7000, 8000, 9000, 10000], dtype=float)
    y = np.array([3000, 0, 0, 0, 0, 0, 80, 100, 80, 0, 0, 0, 0, 0], dtype=float)
    wall = shapely.LineString([(5000, -50), (5000, 50)])
    sel = cpls_select(x, y, 500, wall)
    assert _idx(sel.dp) == [0, 1, 13]
    assert _idx(sel.mask) == [0, 1, 8, 13]
    assert np.array_equal(sel.mask, _reference_cpls(x, y, 500, wall))


def test_cpls_matches_the_literal_algorithm_with_many_retained_points():
    # A wide epsilon collapses the walk to its ends; barriers the path never
    # touches then force a point to be kept wherever a chord would reach one.
    for seed in SEEDS:
        x, y = _walk(n=300, seed=seed)
        path = shapely.linestrings(np.column_stack([x, y]))
        parts = [g for g in _barriers_for(x, y, seed, count=1500)
                 if not shapely.intersects(g, path)]
        sel = cpls_select(x, y, 5_000.0, parts)
        ref = _reference_cpls(x, y, 5_000.0, shapely.union_all(parts))
        assert np.array_equal(sel.mask, ref)
        assert sel.topology.sum() > 20


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cpls_is_a_superset_of_dp(seed, epsilon):
    x, y = _walk(seed=seed)
    sel = cpls_select(x, y, epsilon, _barriers_for(x, y, seed))
    assert np.array_equal(sel.dp, douglas_peucker(x, y, epsilon))
    assert (sel.mask >= sel.dp).all()
    assert not (sel.topology & sel.dp).any()
    assert np.array_equal(sel.added, sel.topology)


def test_cpls_never_introduces_a_crossing():
    dp_introduced = 0
    for seed in SEEDS:
        for epsilon in EPSILONS:
            x, y = _walk(seed=seed)
            parts = _barriers_for(x, y, seed)
            sel = cpls_select(x, y, epsilon, parts)
            assert not segment_crossings(
                x, y, parts, keep=sel.mask, new_only=True
            ).any()
            dp_introduced += segment_crossings(
                x, y, parts, keep=sel.dp, new_only=True
            ).sum()
    assert dp_introduced > 0


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_a_path_clear_of_barriers_stays_clear(seed, epsilon):
    x, y = _walk(seed=seed)
    parts = _barriers_for(x, y, seed)
    path = shapely.linestrings(np.column_stack([x, y]))
    clear = [g for g in parts if not shapely.intersects(g, path)]
    assert clear and len(clear) < len(parts)
    mask = _mask(x, y, epsilon, clear)
    assert not segment_crossings(x, y, clear, keep=mask).any()


def test_no_barriers_is_plain_dp():
    x, y = _walk(seed=2)
    far = shapely.box(1e7, 1e7, 1e7 + 10, 1e7 + 10)
    for barriers in ([], shapely.GeometryCollection(), far):
        sel = cpls_select(x, y, 100.0, barriers)
        assert np.array_equal(sel.mask, douglas_peucker(x, y, 100.0))
        assert not sel.added.any()


# ------------------------------------------------------------- interface


def test_barriers_accept_every_documented_form():
    x, y = _walk(seed=4)
    parts = _barriers_for(x, y, 4)
    expected = _mask(x, y, 150.0, parts)
    assert expected.sum() > douglas_peucker(x, y, 150.0).sum()

    forms = [
        shapely.GeometryCollection(parts),
        np.array(parts, dtype=object),
        gpd.GeoSeries(parts),
        gpd.GeoDataFrame(geometry=parts),
        Barriers(parts),
        [*parts, None, shapely.Point()],
    ]
    for form in forms:
        assert np.array_equal(_mask(x, y, 150.0, form), expected)

    one = shapely.LineString([(5, -1), (5, 2)])
    mp = shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(5, 5, 6, 6)])
    polygons = [p for p in parts if p.geom_type == "Polygon"]
    assert np.array_equal(
        _mask(x, y, 150.0, shapely.MultiPolygon(polygons)),
        _mask(x, y, 150.0, polygons),
    )
    assert len(Barriers(mp)) == 2
    assert len(Barriers(one)) == 1
    assert len(Barriers([one, None, shapely.Point()])) == 1
    assert len(Barriers([])) == 0


def test_barriers_take_the_crs_of_a_geoseries():
    series = gpd.GeoSeries([shapely.Point(0, 0)], crs=32630)
    assert Barriers(series).crs == series.crs
    assert Barriers(shapely.Point(0, 0)).crs is None
    assert Barriers(shapely.Point(0, 0), crs=32630).crs == 32630


def test_barriers_hits_is_vectorised_and_touching_counts():
    b = Barriers([shapely.box(0, 0, 10, 10)])
    hit = b.hits([-5, -5, 10, 20], [5, 20, 10, 20], [-1, 5, 15, 30], [5, 20, 10, 30])
    assert hit.tolist() == [False, False, True, False]
    assert Barriers([]).hits([0.0], [0.0], [1.0], [1.0]).tolist() == [False]


def test_segment_crossings_shapes_and_new_only():
    x = np.array([0.0, 2, 4, 6, 8, 10])
    y = np.array([0.0, 0, 3, 3, 0, 0])
    wall = shapely.LineString([(5, -1), (5, 2)])
    assert segment_crossings(x, y, wall).shape == (5,)
    keep = np.array([True, False, False, True, False, True])
    assert segment_crossings(x, y, wall, keep=keep).shape == (2,)
    assert segment_crossings(x[:1], y[:1], wall).shape == (0,)
    assert segment_crossings(x[:0], y[:0], wall).shape == (0,)
    assert segment_crossings(x, y, wall, keep=np.eye(6, dtype=bool)[0]).shape == (0,)

    dp = np.array([True, False, False, False, False, True])
    assert segment_crossings(x, y, wall, keep=dp).tolist() == [True]
    assert segment_crossings(x, y, wall, keep=dp, new_only=True).tolist() == [True]
    # The original path crosses the wall between points 2 and 3 only; a chord
    # that covers that stretch is a licensed crossing.
    crossing = shapely.LineString([(5, -1), (5, 4)])
    assert segment_crossings(x, y, crossing).tolist() == [False, False, True, False,
                                                            False]
    assert segment_crossings(x, y, crossing, keep=dp, new_only=True).tolist() == [False]
    assert segment_crossings(x, y, crossing, keep=dp).tolist() == [True]


def test_cpls_edge_cases():
    wall = shapely.LineString([(0, 0), (1, 1)])
    for n in (0, 1, 2):
        sel = cpls_select(np.arange(n, dtype=float), np.zeros(n), 1.0, wall)
        assert sel.mask.all() and sel.mask.size == n
        assert not sel.topology.any()
    with pytest.raises(ValueError):
        cpls_select([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], -1.0, wall)
    with pytest.raises(ValueError):
        cpls_select([0.0, np.nan, 2.0], [0.0, 1.0, 0.0], 1.0, wall)
    with pytest.raises(ValueError):
        cpls_select([0.0, 1.0], [0.0], 1.0, wall)
    for bad in (None, "coast", 3):
        with pytest.raises(TypeError):
            cpls_select([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 1.0, bad)
        with pytest.raises(TypeError):
            Barriers(bad)


def test_zero_length_segments_are_tested_as_points():
    x = [0.0, 5.0, 5.0, 10.0]
    y = [0.0, 3.0, 3.0, 0.0]
    inside = shapely.box(4, 2, 6, 4)
    outside = shapely.box(20, 20, 21, 21)
    assert segment_crossings(x, y, inside).tolist() == [True, True, True]
    assert not segment_crossings(x, y, outside).any()


def test_summary_reports_the_topology_stage():
    x = [0, 2, 4, 6, 8, 10]
    y = [0, 0, 3, 3, 0, 0]
    sel = cpls_select(x, y, 10, shapely.LineString([(5, -1), (5, 2)]))
    assert isinstance(sel, CplsSelection)
    text = sel.summary()
    assert "topology" in text
    assert text.splitlines()[0] == "6 points -> 3 kept (50.00% compression)"
    assert text.splitlines()[1].split() == ["Douglas-Peucker", "2"]
    assert text.splitlines()[2].split() == ["+", "topology", "1"]


def test_cpls_handles_a_long_trajectory():
    n = 100_000
    t = np.linspace(0, 400 * np.pi, n)
    x = t * 10.0
    y = 300.0 * np.sin(t) + 20.0 * np.sin(37 * t)
    rng = np.random.default_rng(0)
    cx = rng.uniform(x.min(), x.max(), 2_000)
    cy = rng.uniform(-350, 350, 2_000)
    parts = list(shapely.buffer(shapely.points(cx, cy), 15.0))
    sel = cpls_select(x, y, 50.0, parts)
    assert (sel.mask >= sel.dp).all()
    assert sel.topology.any()
    assert not segment_crossings(x, y, parts, keep=sel.mask, new_only=True).any()


# ------------------------------------------------------------ dataframes


def _frame(n=120, seed=0, crs=32630):
    x, y = _walk(n=n, seed=seed)
    return gpd.GeoDataFrame(
        {"speed": np.arange(n, dtype=float), "vid": 1},
        geometry=gpd.points_from_xy(x, y),
        index=pd.date_range("2016-02-05", periods=n, freq="30s"),
        crs=crs,
    )


def _frame_barriers(gdf, seed=0):
    return _barriers_for(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), seed)


def test_cpls_compress_keeps_every_column():
    gdf = _frame()
    barriers = _frame_barriers(gdf)
    out = cpls_compress(gdf, 150.0, barriers)
    assert list(out.columns) == list(gdf.columns)
    assert 2 < len(out) < len(gdf)
    assert out.index.isin(gdf.index).all()
    mask = _mask(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), 150.0, barriers)
    assert out.index.equals(gdf.index[mask])
    assert out.crs == gdf.crs


def test_cpls_compress_by_group():
    a, b = _frame(seed=0), _frame(seed=1)
    b["vid"] = 2
    barriers = _frame_barriers(a) + _frame_barriers(b, seed=1)
    out = cpls_compress(pd.concat([a, b]), 150.0, barriers, by="vid")
    assert set(out.vid) == {1, 2}
    assert out[out.vid == 1].equals(cpls_compress(a, 150.0, barriers))
    assert out[out.vid == 2].equals(cpls_compress(b, 150.0, barriers))


def test_cpls_compress_refuses_geographic_coordinates():
    with pytest.raises(ValueError, match="geographic"):
        cpls_compress(_frame(crs=4326), 50.0, [])


def test_cpls_compress_refuses_a_crs_mismatch():
    gdf = _frame()
    wrong = gpd.GeoSeries([shapely.Point(0, 0).buffer(10)], crs=3035)
    with pytest.raises(ValueError, match="CRS"):
        cpls_compress(gdf, 50.0, wrong)
    # Plain shapely geometries carry no CRS, and a matching CRS is fine.
    cpls_compress(gdf, 50.0, [shapely.Point(0, 0).buffer(10)])
    cpls_compress(gdf, 50.0, wrong.set_crs(32630, allow_override=True))


@pytest.mark.movingpandas
def test_cpls_compress_movingpandas_objects():
    mpd = pytest.importorskip("movingpandas")
    gdf = _frame()
    barriers = Barriers(_frame_barriers(gdf), crs=32630)
    traj = mpd.Trajectory(gdf, traj_id="v1")
    out = cpls_compress(traj, 150.0, barriers)
    assert isinstance(out, mpd.Trajectory) and out.id == "v1"
    assert "speed" in out.df.columns and len(out.df) < len(traj.df)

    coll = mpd.TrajectoryCollection([traj, mpd.Trajectory(_frame(seed=1), traj_id="v2")])
    out_coll = cpls_compress(coll, 150.0, barriers)
    assert isinstance(out_coll, mpd.TrajectoryCollection)
    assert len(out_coll.trajectories) == len(coll.trajectories) == 2


@pytest.mark.data
def test_cpls_on_the_sample_fleet():
    ais = gpd.read_file(DATASETS / "AIS" / "AIS_DLT_UTM.shp")
    ais = ais.sort_values(["sourcemmsi", "t"], kind="stable")
    land = Barriers(gpd.read_file(DATASETS / "Coastline" / "Coastline_UTM.shp"))
    out = cpls_compress(ais, 1_000, land, by="sourcemmsi")

    dp_index, dp_new, cpls_new = [], 0, 0
    for _, g in ais.groupby("sourcemmsi", sort=False):
        x, y = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
        dp = douglas_peucker(x, y, 1_000)
        dp_index.extend(g.index[dp])
        dp_new += segment_crossings(x, y, land, keep=dp, new_only=True).sum()
        cpls_keep = g.index.isin(out.index)
        cpls_new += segment_crossings(x, y, land, keep=cpls_keep, new_only=True).sum()

    assert dp_new > 0
    assert cpls_new == 0
    assert len(dp_index) <= len(out) < len(ais)
    assert set(dp_index) <= set(out.index)
