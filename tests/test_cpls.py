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
from catraj.compression.cpls import RULES, _direction_cells, _segments_meet



_FAR = 1e9


def _reference_cells(line, box):
    """Nine-cell signature by clipping the line with each cell's rectangle.

    Independent of the library's parametric computation: a cell counts when
    the part of the line inside it, less the cell's boundary, has length.
    """
    xmin, ymin, xmax, ymax = box
    xs, ys = (-_FAR, xmin, xmax, _FAR), (-_FAR, ymin, ymax, _FAR)
    cells = set()
    for i in range(3):
        for j in range(3):
            if xs[i] == xs[i + 1] or ys[j] == ys[j + 1]:
                continue  # a degenerate box has empty middle cells
            cell = shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            inside = line.intersection(cell).difference(cell.boundary)
            if inside.length > 1e-6:
                cells.add((i, j))
    return cells


def _reference_cpls(x, y, epsilon, barriers, **flags):
    """Algorithm 1 read literally, with every relation of the signature.

    One test per candidate point, against the whole current trajectory
    ``Tc``: the final units before ``a``, the candidate ``a -> k``, and the
    original segments after ``k``.  Everything is plain shapely; a geometry
    that does not move is a point.  Returns ``(keep, rules)``.
    """
    on = dict(topology=True, direction=True, avoid_self_intersection=True,
              keep_self_intersection=True, inversion=True)
    on.update(flags)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    parts = [p for g in barriers for p in shapely.get_parts(g) if not p.is_empty]
    dp = douglas_peucker(x, y, epsilon)
    keep = dp.copy()
    rules = {name: np.zeros(n, bool) for name in RULES}

    def path(i, j):
        pts = np.column_stack([x[i:j + 1], y[i:j + 1]])
        if (pts == pts[0]).all():
            return shapely.Point(pts[0])
        return shapely.LineString(pts)

    def chord(i, j):
        return path(i, j) if j == i + 1 else path(i, i).union(path(j, j)).convex_hull

    def hull(i, j):
        return shapely.MultiPoint(np.column_stack([x[i:j + 1], y[i:j + 1]])).convex_hull

    def point(i):
        return shapely.Point(x[i], y[i])

    def meet(g1, g2, v):
        if v is None:
            return g1.intersects(g2)
        return not g1.intersection(g2).difference(point(v)).is_empty

    def broken(a, k, retained):
        sub, c, h = path(a, k), chord(a, k), hull(a, k)
        near = [p for p in parts if p.intersects(h)]
        if on["topology"] and any(sub.relate(p) != c.relate(p) for p in near):
            return "topology"
        if on["direction"]:
            for p in near:
                piece = p.intersection(h)
                if piece.is_empty:
                    continue
                if _reference_cells(sub, piece.bounds) != _reference_cells(c, piece.bounds):
                    return "direction"
        if on["avoid_self_intersection"] or on["keep_self_intersection"]:
            units = [(retained[u], retained[u + 1]) for u in range(len(retained) - 1)]
            units += [(j, j + 1) for j in range(k, n - 1)]
            for p, q in units:
                v = a if q == a else (k if p == k else None)
                in_sub, in_chord = meet(sub, path(p, q), v), meet(c, chord(p, q), v)
                if on["avoid_self_intersection"] and in_chord and not in_sub:
                    return "self_intersection"
                if on["keep_self_intersection"] and in_sub and not in_chord:
                    return "self_intersection"
        if on["inversion"]:
            if len(retained) >= 2:
                p = retained[-2]
                if hull(p, a).contains(point(k)) or h.contains(point(p)):
                    return "inversion"
            if k + 1 < n:
                if h.contains(point(k + 1)) or chord(k, k + 1).contains(point(a)):
                    return "inversion"
        return None

    retained = [0]
    kept = np.flatnonzero(dp)
    for s, e in zip(kept[:-1], kept[1:]):
        a = s
        for j in range(s + 1, e):
            failed = broken(a, j + 1, retained)
            if failed:
                keep[j] = True
                rules[failed][j] = True
                retained.append(j)
                a = j
        retained.append(e)
    return keep, rules


def _check_signature(x, y, mask, barriers):
    """Every relation, checked on a finished result independently of the pass.

    Each final unit keeps D4 and D5 against its neighbours; every pair of
    final units meets as their sub-paths do (D6); consecutive units do not
    invert (D7).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    parts = [p for g in barriers for p in shapely.get_parts(g) if not p.is_empty]
    r = np.flatnonzero(mask)

    def path(i, j):
        pts = np.column_stack([x[i:j + 1], y[i:j + 1]])
        return shapely.Point(pts[0]) if (pts == pts[0]).all() else shapely.LineString(pts)

    def chord(i, j):
        return path(i, j) if j == i + 1 else path(i, i).union(path(j, j)).convex_hull

    def hull(i, j):
        return shapely.MultiPoint(np.column_stack([x[i:j + 1], y[i:j + 1]])).convex_hull

    units = list(zip(r[:-1], r[1:]))
    for p, q in units:
        h = hull(p, q)
        for part in parts:
            if part.intersects(h):
                assert path(p, q).relate(part) == chord(p, q).relate(part)
                box = part.intersection(h).bounds
                assert _reference_cells(path(p, q), box) == _reference_cells(chord(p, q), box)
    for u, (p, q) in enumerate(units):
        for w, (s, t) in enumerate(units[u + 1:], start=u + 1):
            v = q if w == u + 1 else None
            a = path(p, q).intersection(path(s, t))
            b = chord(p, q).intersection(chord(s, t))
            if v is not None:
                a, b = (g.difference(shapely.Point(x[v], y[v])) for g in (a, b))
            assert a.is_empty == b.is_empty, (u, w)
    for (p, q), (_, t) in zip(units[:-1], units[1:]):
        assert not hull(p, q).contains(shapely.Point(x[t], y[t]))
        assert not hull(q, t).contains(shapely.Point(x[p], y[p]))


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
    # Below the chord, outside the path's hull: not a neighbour.
    assert _idx(_mask(x, y, 10, shapely.Point(5, -0.5))) == [0, 2]


def test_an_island_may_not_change_sides():
    # The vessel passes north of the island; the chord would pass south of it
    # without touching it.  The nine cells differ, so the point is kept (D5).
    x, y = [0, 500, 1000], [0, 300, 0]
    island = shapely.box(450, 100, 550, 150)
    sel = cpls_select(x, y, 5_000, island)
    assert _idx(sel.dp) == [0, 2]
    assert not segment_crossings(x, y, island, keep=sel.dp).any()
    assert _idx(sel.mask) == [0, 1, 2]
    # The same island south of the chord is outside the hull and changes nothing.
    assert _idx(_mask(x, y, 5_000, shapely.box(450, -150, 550, -100))) == [0, 2]
    # A point between path and chord is an island too.
    assert _idx(_mask([0, 5, 10], [0, 1, 0], 10, shapely.Point(5, 0.5))) == [0, 1, 2]


def test_direction_cells_are_the_nine_cell_model():
    box = (0.0, 0.0, 1.0, 1.0)
    bits = {}  # (column, row) -> bit, column west to east, row south to north
    for i in range(3):
        for j in range(3):
            bits[i, j] = 1 << (3 * i + j)
    below = _direction_cells(np.array([-1.0, 2.0]), np.array([-1.0, -1.0]), box)
    assert below == bits[0, 0] | bits[1, 0] | bits[2, 0]
    across = _direction_cells(np.array([0.5, 0.5]), np.array([-1.0, 2.0]), box)
    assert across == bits[1, 0] | bits[1, 1] | bits[1, 2]
    # Along an edge, through a corner, or standing still: no interior passed.
    assert _direction_cells(np.array([-1.0, 2.0]), np.array([0.0, 0.0]), box) == 0
    assert _direction_cells(np.array([5.0, 5.0]), np.array([5.0, 5.0]), box) == 0
    diagonal = _direction_cells(np.array([-1.0, 2.0]), np.array([-1.0, 2.0]), box)
    assert diagonal == bits[0, 0] | bits[1, 1] | bits[2, 2]


def test_an_existing_crossing_is_tolerated():
    x = np.linspace(-10, 20, 31)
    y = np.full(31, 5.0)
    sel = cpls_select(x, y, 1, shapely.box(0, 0, 10, 10))
    assert _idx(sel.dp) == [0, 30]
    assert _idx(sel.mask) == [0, 30]


# ------------------------------------------------------------ properties

SEEDS = range(5)
EPSILONS = [30.0, 150.0, 600.0]


OFF = dict(topology=False, direction=False, avoid_self_intersection=False,
           keep_self_intersection=False, inversion=False)
BARRIERS_ONLY = dict(OFF, topology=True, direction=True)


def _assert_same(sel, reference):
    keep, rules = reference
    assert np.array_equal(sel.mask, keep)
    for name in RULES:
        assert np.array_equal(sel.rules[name], rules[name]), name


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cpls_matches_the_literal_algorithm(seed, epsilon):
    x, y = _walk(n=120, seed=seed)
    parts = _barriers_for(x, y, seed)
    _assert_same(cpls_select(x, y, epsilon, parts),
                 _reference_cpls(x, y, epsilon, parts))


@pytest.mark.parametrize("name", sorted(OFF))
@pytest.mark.parametrize("seed", range(3))
def test_each_relation_alone_matches_the_literal_algorithm(seed, name):
    x, y = _walk(n=120, seed=seed)
    parts = _barriers_for(x, y, seed)
    flags = dict(OFF, **{name: True})
    _assert_same(cpls_select(x, y, 150.0, parts, **flags),
                 _reference_cpls(x, y, 150.0, parts, **flags))


def test_repeated_positions_match_the_literal_algorithm():
    # A vessel that stops: runs of identical positions and GPS jitter round a
    # mooring, then moves on.  Zero-length segments are points (D9).
    rng = np.random.default_rng(3)
    x, y = _walk(n=60, seed=3)
    moor = np.repeat(np.column_stack([x[30], y[30]]), 25, axis=0)
    moor[5:20] += np.round(rng.normal(0, 3, size=(15, 2)))
    x = np.r_[x[:30], moor[:, 0], x[30:]]
    y = np.r_[y[:30], moor[:, 1], y[30:]]
    parts = _barriers_for(x, y, 3)
    for eps in (20.0, 200.0):
        sel = cpls_select(x, y, eps, parts)
        _assert_same(sel, _reference_cpls(x, y, eps, parts))
        _check_signature(x, y, sel.mask, parts)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_result_keeps_every_relation(seed):
    x, y = _walk(n=150, seed=seed)
    parts = _barriers_for(x, y, seed)
    for eps in (60.0, 400.0):
        _check_signature(x, y, cpls_select(x, y, eps, parts).mask, parts)


def test_every_relation_off_is_plain_dp():
    x, y = _walk(seed=1)
    parts = _barriers_for(x, y, 1)
    sel = cpls_select(x, y, 100.0, parts, **OFF)
    assert np.array_equal(sel.mask, douglas_peucker(x, y, 100.0))


def test_a_touch_must_survive_simplification():
    # D4, DE-9IM: the path touches the barrier at a vertex; the chord would
    # not touch it at all, so the vertex stays.
    sel = cpls_select([0, 5, 10], [0, 1, 0], 10, shapely.Point(5, 1))
    assert _idx(sel.mask) == [0, 1, 2]
    assert sel.rules["topology"][1]


def test_a_new_self_intersection_is_refused():
    x, y = np.array([(0, 0), (5, 4), (10, 0), (10, -2), (5, 2.5), (5, -3),
                     (20, -3)], float).T
    assert _idx(cpls_select(x, y, 100, [], **OFF).mask) == [0, 6]
    sel = cpls_select(x, y, 100, [], **dict(OFF, avoid_self_intersection=True))
    # The chord 0 -> 2 would cross segment 3-4, which the path never meets.
    assert _idx(sel.mask) == [0, 1, 6]
    assert sel.rules["self_intersection"][1]


def test_an_existing_self_intersection_is_kept():
    # The path comes back through its own vertex 1; dropping 3 and 4 would
    # lose that meeting.
    x, y = np.array([(0, 0), (4, 0), (6, 1), (6, 3), (4, 3), (4, -2),
                     (10, -2)], float).T
    keep = dict(OFF, keep_self_intersection=True)
    assert _idx(cpls_select(x, y, 100, [], **keep).mask) == [0, 3, 4, 6]
    avoid = dict(OFF, avoid_self_intersection=True)
    assert _idx(cpls_select(x, y, 100, [], **avoid).mask) == [0, 6]


def test_a_turn_back_into_the_hull_is_an_inversion():
    x, y = np.array([(0, 0), (4, 3), (8, 0), (5, 1), (3, -4)], float).T
    sel = cpls_select(x, y, 100, [], **dict(OFF, inversion=True))
    # Point 3 lies inside the hull of 0..2 (heuristic 3.7): 1 is kept.
    assert _idx(sel.mask) == [0, 1, 4]
    assert sel.rules["inversion"][1]
    zigzag = np.array([(0, 0), (3, 2), (6, 0), (9, 2), (12, 0)], float).T
    assert _idx(cpls_select(*zigzag, 100, [], **dict(OFF, inversion=True)).mask) == [0, 4]


def test_segments_meet_matches_shapely_with_points_for_zero_length():
    rng = np.random.default_rng(0)
    pts = np.round(rng.uniform(0, 6, size=(4000, 8)))  # many collinear, touching
    pts[::7, 2:4] = pts[::7, 0:2]  # zero-length first segments
    pts[::11, 6:8] = pts[::11, 4:6]  # zero-length second segments

    def geom(ax, ay, bx, by):
        if ax == bx and ay == by:
            return shapely.Point(ax, ay)
        return shapely.LineString([(ax, ay), (bx, by)])

    ours = _segments_meet(*pts.T)
    expected = [geom(*r[:4]).intersects(geom(*r[4:])) for r in pts]
    assert np.array_equal(ours, expected)


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
    _assert_same(sel, _reference_cpls(x, y, 500, [wall]))


def test_cpls_matches_the_literal_algorithm_with_many_retained_points():
    # A wide epsilon collapses the walk to its ends; barriers the path never
    # touches then force a point to be kept wherever a chord would reach one.
    for seed in SEEDS:
        x, y = _walk(n=300, seed=seed)
        path = shapely.linestrings(np.column_stack([x, y]))
        parts = [g for g in _barriers_for(x, y, seed, count=1500)
                 if not shapely.intersects(g, path)]
        sel = cpls_select(x, y, 5_000.0, parts, **BARRIERS_ONLY)
        ref, _ = _reference_cpls(x, y, 5_000.0, parts, **BARRIERS_ONLY)
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
    # With only the barrier relations on; the others act without barriers.
    x, y = _walk(seed=2)
    far = shapely.box(1e7, 1e7, 1e7 + 10, 1e7 + 10)
    for barriers in ([], shapely.GeometryCollection(), far):
        sel = cpls_select(x, y, 100.0, barriers, **BARRIERS_ONLY)
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
    # A winding track among many small islands; long enough to show nothing
    # in the pass grows out of hand (100,000 points take about half a minute).
    n = 20_000
    t = np.linspace(0, 80 * np.pi, n)
    x = t * 10.0
    y = 300.0 * np.sin(t) + 20.0 * np.sin(37 * t)
    rng = np.random.default_rng(0)
    cx = rng.uniform(x.min(), x.max(), 400)
    cy = rng.uniform(-350, 350, 400)
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
    from catraj.datasets import load_ais, load_coastline

    ais = load_ais()    # sorted by (mmsi, unix_time)
    land = Barriers(load_coastline())
    # The barrier relations over the whole fleet; every relation is checked on
    # moving vessels below (moored ones are slow with all of them).
    out = cpls_compress(ais, 1_000, land, by="mmsi", filter_window=None,
                        **BARRIERS_ONLY)

    dp_index, dp_new, cpls_new = [], 0, 0
    for _, g in ais.groupby("mmsi", sort=False):
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


@pytest.mark.data
def test_every_relation_on_moving_sample_vessels():
    from catraj.datasets import load_ais, load_coastline

    ais = load_ais()
    land = load_coastline()
    moving = ais.groupby("mmsi").speed_over_ground.median()
    moving = moving[(moving > 5) & (moving < 40)].index[:12]
    for mmsi in moving:
        g = ais[ais.mmsi == mmsi]
        x, y = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
        sel = cpls_select(x, y, 1_000, land)
        assert (sel.mask >= sel.dp).all()
        assert not segment_crossings(x, y, land, keep=sel.mask, new_only=True).any()
        _check_signature(x, y, sel.mask, list(land.geometry))


# ----------------------------------------- filtering first (default, 30 s)


def _spiky_frame(n=60):
    x = np.arange(n) * 100.0
    y = np.zeros(n)
    y[30] = 500.0
    return gpd.GeoDataFrame(
        {"vid": 1}, geometry=gpd.points_from_xy(x, y),
        index=pd.to_datetime(1_454_000_000 + np.arange(n) * 10.0, unit="s"),
        crs=32630,
    )


def test_cpls_compress_filters_the_positions_first_by_default():
    gdf = _spiky_frame()
    raw = cpls_compress(gdf, 50.0, [], filter_window=None)
    out = cpls_compress(gdf, 50.0, [])
    assert gdf.index[30] in raw.index and gdf.index[30] not in out.index
    assert (out.geometry.y == 0.0).all()  # the rows carry the filtered geometry
    assert out.equals(cpls_compress(gdf, 50.0, [], filter_window=30))


def test_cpls_compress_without_filtering_is_the_unfiltered_algorithm():
    gdf = _spiky_frame()
    mask = cpls_select(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(),
                       50.0, []).mask
    out = cpls_compress(gdf, 50.0, [], filter_window=None)
    assert out.equals(gdf.iloc[mask])


def test_cpls_compress_reads_time_from_a_column_or_the_index():
    gdf = _spiky_frame()
    plain = gdf.reset_index(drop=True)
    with pytest.raises(ValueError, match="time_column"):
        cpls_compress(plain, 50.0, [])
    plain["clock"] = np.arange(len(plain)) * 10.0
    out = cpls_compress(plain, 50.0, [], time_column="clock")
    want = cpls_compress(gdf, 50.0, [])
    assert out.index.tolist() == [gdf.index.get_loc(k) for k in want.index]
    assert len(cpls_compress(plain, 50.0, [], filter_window=None)) > len(out)


def test_cpls_compress_by_group_filters_each_group():
    a, b = _spiky_frame(), _spiky_frame()
    b["vid"] = 2
    out = cpls_compress(pd.concat([a, b]), 50.0, [], by="vid")
    assert out[out.vid == 1].equals(cpls_compress(a, 50.0, []))
    assert out[out.vid == 2].equals(cpls_compress(b, 50.0, []))


@pytest.mark.movingpandas
def test_cpls_compress_passes_the_filter_window_to_trajectories_and_collections():
    mpd = pytest.importorskip("movingpandas")
    gdf = _spiky_frame()
    traj = mpd.Trajectory(gdf, traj_id="v1")
    coll = mpd.TrajectoryCollection([traj])
    for window in (None, 30):
        want = cpls_compress(gdf, 50.0, [], filter_window=window)
        got = cpls_compress(traj, 50.0, [], filter_window=window)
        assert got.df.index.equals(want.index)
        again = cpls_compress(coll, 50.0, [], filter_window=window).trajectories[0]
        assert again.df.index.equals(want.index)
