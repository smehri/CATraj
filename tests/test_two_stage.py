import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from catraj.compression import (
    Barriers,
    TwoStageSelection,
    cpls_select,
    douglas_peucker,
    segment_crossings,
    stay_points,
    two_stage_cpls_compress,
    two_stage_cpls_select,
)
from catraj.compression.cpls import RULES
from catraj.compression.two_stage import _nearest, _runs, _speed_dp, _speeds



_FAR = 1e9
OFF = dict(topology=False, direction=False, avoid_self_intersection=False,
           keep_self_intersection=False, inversion=False)


# ------------------------------------------------------------- references


def _reference_cells(line, box):
    xmin, ymin, xmax, ymax = box
    xs, ys = (-_FAR, xmin, xmax, _FAR), (-_FAR, ymin, ymax, _FAR)
    cells = set()
    for i in range(3):
        for j in range(3):
            if xs[i] == xs[i + 1] or ys[j] == ys[j + 1]:
                continue
            cell = shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            inside = line.intersection(cell).difference(cell.boundary)
            if inside.length > 1e-6:
                cells.add((i, j))
    return cells


def _reference_stage2(x, y, stage1, barriers, **flags):
    """CPLS lines 13-20 read literally, after a given first stage.

    A copy of the literal CPLS reference, generalised to start from a mask of
    kept points instead of Douglas--Peucker.  One test per candidate, against
    the whole current trajectory ``Tc``; plain shapely.  Returns ``(keep,
    rules)``.
    """
    on = dict(topology=True, direction=True, avoid_self_intersection=True,
              keep_self_intersection=True, inversion=True)
    on.update(flags)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    parts = [p for g in barriers for p in shapely.get_parts(g) if not p.is_empty]
    keep = stage1.copy()
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
    kept = np.flatnonzero(stage1)
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


def _median_loop(values):
    ordered = sorted(values)
    k = len(ordered)
    return (ordered[(k - 1) // 2] + ordered[k // 2]) / 2.0


def _reference_two_stage(x, y, t, epsilon, speed_epsilon, barriers, stays,
                         speed_window=None, **flags):
    """Algorithm 2 read literally (A1-A4, B4, P2) with plain loops."""
    n = len(x)
    # A2: collapse each stay run to its centroid at its first and last point.
    vx, vy = list(x), list(y)
    index, is_end = [], []
    i = 0
    while i < n:
        if not stays[i]:
            index.append(i)
            is_end.append(False)
            i += 1
            continue
        j = i
        while j + 1 < n and stays[j + 1]:
            j += 1
        cx, cy = np.mean(x[i:j + 1]), np.mean(y[i:j + 1])
        for k in range(i, j + 1):
            vx[k], vy[k] = cx, cy
        index += [i] if i == j else [i, j]
        is_end += [True] if i == j else [True, True]
        i = j + 1
    m = len(index)
    px = np.array([vx[k] for k in index])
    py = np.array([vy[k] for k in index])
    pt = np.array([t[k] for k in index], float)
    # A3: central-difference speed, nearest defined neighbour where undefined.
    raw_v = []
    for i in range(m):
        lo, hi = max(i - 1, 0), min(i + 1, m - 1)
        span = pt[hi] - pt[lo]
        raw_v.append(np.hypot(px[hi] - px[lo], py[hi] - py[lo]) / span if span > 0 else None)
    v = []
    for i in range(m):
        value = raw_v[i]
        d = 1
        while value is None and d < m:
            for k in (i - d, i + d):
                if 0 <= k < m and raw_v[k] is not None:
                    value = raw_v[k]
                    break
            d += 1
        v.append(0.0 if value is None else value)
    if speed_window is not None:
        # P2: the speed series, median-filtered in time (|dt| <= window / 2).
        v = [_median_loop([v[j] for j in range(m)
                           if abs(pt[j] - pt[i]) <= speed_window / 2])
             for i in range(m)]
    speed_kept = [False] * m
    speed_kept[0] = speed_kept[-1] = True

    def recurse(f, l):
        if l - f < 2:
            return
        worst, at = -1.0, None
        for i in range(f + 1, l):
            if pt[l] > pt[f]:
                frac = (pt[i] - pt[f]) / (pt[l] - pt[f])
            else:
                frac = (i - f) / (l - f)
            err = abs(v[i] - (v[f] + frac * (v[l] - v[f])))
            if err > worst:
                worst, at = err, i
        if worst > speed_epsilon:
            speed_kept[at] = True
            recurse(f, at)
            recurse(at, l)

    if m > 2:
        recurse(0, m - 1)
    stage1 = np.array([a or b for a, b in zip(speed_kept, is_end)])
    forced = np.flatnonzero(stage1)
    for f, l in zip(forced[:-1], forced[1:]):
        if l - f >= 2:
            stage1[f:l + 1] |= douglas_peucker(px[f:l + 1], py[f:l + 1], epsilon)
    keep, rules = _reference_stage2(px, py, stage1, barriers, **flags) if m > 2 else (
        stage1, {name: np.zeros(m, bool) for name in RULES})
    out = np.zeros(n, bool)
    out[np.array(index)[keep]] = True
    out_rules = {}
    for name, flagged in rules.items():
        out_rules[name] = np.zeros(n, bool)
        out_rules[name][np.array(index)[flagged]] = True
    return out, out_rules, np.array(vx), np.array(vy)


def _li_reference(x, y, t, radius, duration):
    """Li et al.'s loop with the tail run tested (A1), point by point."""
    n = len(x)
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.hypot(x[j] - x[i], y[j] - y[i]) <= radius:
            j += 1
        if t[j - 1] - t[i] >= duration:
            out[i:j] = True
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------ data


def _walk(n=120, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=50.0, size=(n, 2)) + [30.0, 5.0]
    xy = np.cumsum(steps, axis=0)
    return xy[:, 0], xy[:, 1]


def _barriers_for(x, y, seed, count=25):
    rng = np.random.default_rng(100 + seed)
    xlo, xhi, ylo, yhi = x.min(), x.max(), y.min(), y.max()
    out = []
    for k in range(count):
        if k % 5 == 0:
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


def _with_stays(n=120, seed=0, repeated_times=False):
    """A random walk in which three random runs are stays, with GPS jitter."""
    rng = np.random.default_rng(1000 + seed)
    x, y = _walk(n, seed)
    t = np.cumsum(rng.uniform(10, 60, n))
    if repeated_times:
        t[rng.integers(1, n, 12)] = t[rng.integers(0, n - 1, 12)]
        t = np.maximum.accumulate(t)
    stays = np.zeros(n, bool)
    for start in rng.choice(np.arange(3, n - 15), 3, replace=False):
        length = int(rng.integers(1, 12))
        stays[start:start + length] = True
        x[start:start + length] = x[start] + rng.normal(scale=8, size=length)
        y[start:start + length] = y[start] + rng.normal(scale=8, size=length)
    return x, y, t, stays


def _assert_same(sel, reference):
    keep, rules = reference[:2]
    assert np.array_equal(sel.mask, keep)
    for name in RULES:
        assert np.array_equal(sel.rules[name], rules[name]), name


def _signature_differences(rx, ry, vx, vy, mask, barriers, stays):
    """Every relation of the signature, a final result against a reference track.

    Independent of the library's pass, plain shapely, in the manner of
    ``_check_signature`` in ``test_cpls.py``.  The reference track ``rx, ry``
    gives each final unit its sub-path (the reference points between the
    unit's two rows); the chord comes from the final vertices ``vx, vy``.
    For each relation it counts the differences -- units for topology
    (DE-9IM), direction (nine cells) and inversion, pairs of units for new
    and lost self-intersections -- and how many of them involve a stay: a unit
    whose raw rows ``p..q`` hold a stay point.  Returns
    ``{relation: (count, count at a stay)}``.

    One difference is not counted among the relations but apart, under
    ``"inversion_exempt"``: a unit of one segment of the collapsed track is
    never a candidate, so it is never tested against the unit before it (B5).
    """
    rx, ry = np.asarray(rx, float), np.asarray(ry, float)
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    parts = [p for g in barriers for p in shapely.get_parts(g) if not p.is_empty]
    tree = shapely.STRtree(parts) if parts else None
    r = np.flatnonzero(mask)
    units = list(zip(r[:-1], r[1:]))

    def make(pts):
        return shapely.Point(pts[0]) if (pts == pts[0]).all() else shapely.LineString(pts)

    def sub(i, j):
        return make(np.column_stack([rx[i:j + 1], ry[i:j + 1]]))

    def chord(i, j):
        return make(np.array([[vx[i], vy[i]], [vx[j], vy[j]]]))

    def hull(i, j):
        pts = np.vstack([np.column_stack([rx[i:j + 1], ry[i:j + 1]]),
                         [[vx[i], vy[i]], [vx[j], vy[j]]]])
        return shapely.MultiPoint(pts).convex_hull

    def subhull(i, j):
        return shapely.MultiPoint(np.column_stack([rx[i:j + 1], ry[i:j + 1]])).convex_hull

    at_stay = [bool(stays[p:q + 1].any()) for p, q in units]
    interior = np.zeros(len(rx), dtype=bool)
    for first, last in zip(*_runs(np.asarray(stays, bool))):
        interior[first + 1:last] = True
    one_segment = [int((~interior[p:q + 1]).sum()) == 2 for p, q in units]
    found = {name: [0, 0] for name in RULES[:2] + ("avoid_self_intersection",
             "keep_self_intersection", "inversion", "inversion_exempt")}

    def note(name, involved):
        found[name][0] += 1
        found[name][1] += bool(involved)

    subs = [sub(p, q) for p, q in units]
    chords = [chord(p, q) for p, q in units]
    for u, (p, q) in enumerate(units):
        if tree is None:
            break
        h = hull(p, q)
        near = [parts[i] for i in tree.query(h, predicate="intersects")]
        if any(subs[u].relate(part) != chords[u].relate(part) for part in near):
            note("topology", at_stay[u])
        for part in near:
            piece = part.intersection(h)
            if piece.is_empty:
                continue
            if _reference_cells(subs[u], piece.bounds) != _reference_cells(
                    chords[u], piece.bounds):
                note("direction", at_stay[u])
                break

    if len(units) > 1:
        # Pairs whose boxes (sub-path or chord) do not meet cannot meet.
        pairs = set()
        for geoms in (subs, chords):
            first, second = shapely.STRtree(geoms).query(geoms)
            pairs.update((int(a), int(b)) for a, b in zip(first, second) if a < b)
        for u, w in sorted(pairs):
            v = units[u][1] if w == u + 1 else None
            if v is None:
                in_sub, in_chord = subs[u].intersects(subs[w]), chords[u].intersects(chords[w])
            else:
                in_sub = not subs[u].intersection(subs[w]).difference(
                    shapely.Point(rx[v], ry[v])).is_empty
                in_chord = not chords[u].intersection(chords[w]).difference(
                    shapely.Point(vx[v], vy[v])).is_empty
            if in_chord and not in_sub:
                note("avoid_self_intersection", at_stay[u] or at_stay[w])
            if in_sub and not in_chord:
                note("keep_self_intersection", at_stay[u] or at_stay[w])

    for u in range(len(units) - 1):
        (p, q), (_, t) = units[u], units[u + 1]
        if subhull(p, q).contains(shapely.Point(vx[t], vy[t])) or subhull(q, t).contains(
                shapely.Point(vx[p], vy[p])):
            note("inversion_exempt" if one_segment[u + 1] else "inversion",
                 at_stay[u] or at_stay[u + 1])
    return {name: tuple(c) for name, c in found.items()}


SEEDS = range(5)
EPSILONS = [30.0, 150.0, 600.0]


# ----------------------------------------------------- the literal algorithm


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_two_stage_matches_the_literal_algorithm(seed, epsilon):
    x, y, t, stays = _with_stays(seed=seed)
    parts = _barriers_for(x, y, seed)
    for speed_epsilon in (0.5, 2.0):
        sel = two_stage_cpls_select(x, y, t, epsilon, speed_epsilon, parts, stays=stays,
                                    speed_window=None)
        ref = _reference_two_stage(x, y, t, epsilon, speed_epsilon, parts, stays)
        _assert_same(sel, ref)
        assert np.array_equal(sel.x, ref[2]) and np.array_equal(sel.y, ref[3])
        assert not (sel.mask & stays & ~sel.stay_ends).any()


@pytest.mark.parametrize("name", sorted(OFF))
@pytest.mark.parametrize("seed", range(3))
def test_each_relation_alone_matches_the_literal_algorithm(seed, name):
    x, y, t, stays = _with_stays(seed=seed)
    parts = _barriers_for(x, y, seed)
    flags = dict(OFF, **{name: True})
    sel = two_stage_cpls_select(x, y, t, 150.0, 1.0, parts, stays=stays,
                                speed_window=None, **flags)
    _assert_same(sel, _reference_two_stage(x, y, t, 150.0, 1.0, parts, stays, **flags))


@pytest.mark.parametrize("seed", range(3))
def test_a_walk_without_stays_matches_the_literal_algorithm(seed):
    x, y, t, _ = _with_stays(seed=seed)
    none = np.zeros(x.size, bool)
    parts = _barriers_for(x, y, seed)
    sel = two_stage_cpls_select(x, y, t, 150.0, 1.0, parts, stays=none,
                                speed_window=None)
    _assert_same(sel, _reference_two_stage(x, y, t, 150.0, 1.0, parts, none))
    assert not sel.stay_ends.any()


@pytest.mark.parametrize("seed", range(3))
def test_repeated_times_match_the_literal_algorithm(seed):
    x, y, t, stays = _with_stays(seed=seed, repeated_times=True)
    assert (np.diff(t) == 0).any()
    parts = _barriers_for(x, y, seed)
    sel = two_stage_cpls_select(x, y, t, 150.0, 1.0, parts, stays=stays,
                                speed_window=None)
    _assert_same(sel, _reference_two_stage(x, y, t, 150.0, 1.0, parts, stays))


def test_without_stays_and_speed_off_it_is_cpls():
    for seed in SEEDS:
        x, y, t, _ = _with_stays(seed=seed)
        parts = _barriers_for(x, y, seed)
        two = two_stage_cpls_select(x, y, t, 150.0, np.inf, parts)
        one = cpls_select(x, y, 150.0, parts)
        assert np.array_equal(two.mask, one.mask)
        assert np.array_equal(two.dp | two.speed | two.stay_ends, one.dp)
        for name in RULES:
            assert np.array_equal(two.rules[name], one.rules[name])


# ------------------------------------------------------------ stay points


def _vessel():
    """10 m/s for 40 points, 25 points idling within 10 m for 750 s, 40 more."""
    rng = np.random.default_rng(0)
    xs = np.r_[np.arange(40) * 100.0, 4000.0 + rng.uniform(-5, 5, 25),
               4000.0 + 100.0 * np.arange(1, 41)]
    ys = np.r_[np.zeros(40), rng.uniform(-5, 5, 25), np.zeros(40)]
    t = np.r_[np.arange(40) * 10.0, 400.0 + np.arange(25) * 30.0,
              1150.0 + np.arange(1, 41) * 10.0]
    return xs, ys, t


def test_stay_points_by_status_alone():
    x, y, t = _vessel()
    status = np.zeros(x.size)
    status[40:50] = 1
    status[60] = 5
    status[70] = 7  # not a stay status
    got = stay_points(x, y, t, status=status)
    assert np.flatnonzero(got).tolist() == list(range(40, 50)) + [60]
    assert np.flatnonzero(stay_points(x, y, t, status=status, stay_status=(7,))) == [70]
    assert not stay_points(x, y, t).any()


def test_stay_points_by_movement_alone():
    x, y, t = _vessel()
    got = stay_points(x, y, t, stay_radius=50.0, stay_duration=300.0)
    assert np.flatnonzero(got).tolist() == list(range(40, 65))
    # Too short a stop is not a stay.
    assert not stay_points(x, y, t, stay_radius=50.0, stay_duration=2000.0).any()


def test_stay_points_by_status_or_movement():
    x, y, t = _vessel()
    status = np.zeros(x.size)
    status[10:12] = 1
    got = stay_points(x, y, t, status=status, stay_radius=50.0, stay_duration=300.0)
    assert np.flatnonzero(got).tolist() == [10, 11] + list(range(40, 65))


def test_a_stay_that_runs_to_the_end_is_found():
    x, y, t = _vessel()
    x, y, t = x[:55], y[:55], t[:55]
    got = stay_points(x, y, t, stay_radius=50.0, stay_duration=300.0)
    assert np.flatnonzero(got).tolist() == list(range(40, 55))


@pytest.mark.parametrize("seed", range(4))
def test_movement_stays_match_li_s_loop(seed):
    rng = np.random.default_rng(seed)
    x, y, t, _ = _with_stays(n=300, seed=seed)
    x = np.round(x / 20) * 20 + rng.normal(0, 2, x.size)  # slow, wandering
    for radius, duration in ((30.0, 120.0), (100.0, 400.0), (400.0, 900.0)):
        assert np.array_equal(
            stay_points(x, y, t, stay_radius=radius, stay_duration=duration),
            _li_reference(x, y, t, radius, duration),
        )


def test_stay_thresholds_come_in_pairs():
    x, y, t = _vessel()
    with pytest.raises(ValueError):
        stay_points(x, y, t, stay_radius=50.0)
    with pytest.raises(ValueError):
        stay_points(x, y, t, stay_duration=300.0)
    with pytest.raises(ValueError):
        two_stage_cpls_select(x, y, t, 100.0, 1.0, [], stay_duration=300.0)


def test_a_given_mask_overrides_detection():
    x, y, t = _vessel()
    mine = np.zeros(x.size, bool)
    mine[5:9] = True
    sel = two_stage_cpls_select(
        x, y, t, 100.0, 1.0, [], stays=mine, status=np.ones(x.size),
        stay_radius=50.0, stay_duration=300.0,
    )
    assert np.array_equal(sel.stays, mine)


# ---------------------------------------------------------------- the stay


def test_a_stay_is_two_vertices_at_its_centroid():
    x, y, t = _vessel()
    stays = np.zeros(x.size, bool)
    stays[40:65] = True
    sel = two_stage_cpls_select(x, y, t, 100.0, np.inf, [], stays=stays)
    assert np.flatnonzero(sel.stay_ends).tolist() == [40, 64]
    assert sel.mask[40] and sel.mask[64] and not sel.mask[41:64].any()
    cx, cy = x[40:65].mean(), y[40:65].mean()
    assert np.allclose(sel.x[40:65], cx) and np.allclose(sel.y[40:65], cy)
    assert np.array_equal(sel.x[~stays], x[~stays])
    assert not (sel.mask & sel.stays & ~sel.stay_ends).any()


def test_a_one_point_stay_is_one_unmoved_vertex():
    x, y, t = _vessel()
    stays = np.zeros(x.size, bool)
    stays[20] = True
    sel = two_stage_cpls_select(x, y, t, 100.0, np.inf, [], stays=stays)
    assert np.flatnonzero(sel.stay_ends).tolist() == [20]
    assert sel.x[20] == x[20] and sel.y[20] == y[20]


def test_the_stop_keeps_its_duration():
    x, y, t = _vessel()
    sel = two_stage_cpls_select(x, y, t, 100.0, 1.54, [], stay_radius=50.0,
                                stay_duration=300.0)
    kept = np.flatnonzero(sel.mask)
    arrival, departure = kept[sel.stay_ends[kept]]
    assert t[departure] - t[arrival] == 24 * 30.0
    assert sel.x[arrival] == sel.x[departure]


def test_the_collapsed_signature_is_kept_on_every_relation():
    found_in_raw = 0
    for seed in SEEDS:
        for epsilon in EPSILONS:
            x, y, t, stays = _with_stays(seed=seed)
            parts = _barriers_for(x, y, seed)
            sel = two_stage_cpls_select(x, y, t, epsilon, 2.0, parts, stays=stays)
            against = _signature_differences(sel.x, sel.y, sel.x, sel.y, sel.mask,
                                             parts, stays)
            against.pop("inversion_exempt")
            assert all(count == (0, 0) for count in against.values()), against
            raw = _signature_differences(x, y, sel.x, sel.y, sel.mask, parts, stays)
            found_in_raw += sum(count for count, _ in raw.values())
            assert raw["inversion_exempt"][0] >= 0
    # The raw track is a different reference: stays are where it differs.
    assert found_in_raw > 0


def test_the_signature_check_sees_what_stage_two_prevents():
    flagged = 0
    for seed in SEEDS:
        x, y, t, stays = _with_stays(seed=seed)
        parts = _barriers_for(x, y, seed)
        sel = two_stage_cpls_select(x, y, t, 150.0, 2.0, parts, stays=stays)
        before = sel.mask & ~sel.topology
        found = _signature_differences(sel.x, sel.y, sel.x, sel.y, before, parts, stays)
        flagged += sum(count for count, _ in found.values())
    assert flagged > 0


# ------------------------------------------------------------- the speed


def test_the_speed_stage_keeps_the_braking_point():
    # 8 m/s, then 2 m/s, on one straight line: Douglas-Peucker drops the turn.
    x = np.r_[np.arange(21) * 80.0, 1600.0 + np.arange(1, 21) * 20.0]
    t = np.arange(41) * 10.0
    y = np.zeros(41)
    assert douglas_peucker(x, y, 1.0).sum() == 2
    sel = two_stage_cpls_select(x, y, t, 1.0, 1.54, [])
    assert sel.speed[18:23].any()
    assert sel.mask.sum() > 2
    off = two_stage_cpls_select(x, y, t, 1.0, np.inf, [])
    assert off.mask.sum() == 2


def test_speed_is_interpolated_in_time():
    # By position 5 lies on the line from 0 to 10; in time, at t = 1 of 100,
    # the line is at 0.1 and the point is 4.9 away.
    v = np.array([0.0, 5.0, 10.0])
    assert _speed_dp(v, np.array([0.0, 1.0, 100.0]), 1.0).tolist() == [True, True, True]
    assert _speed_dp(v, np.array([0.0, 50.0, 100.0]), 1.0).tolist() == [True, False, True]
    # A zero time span falls back on the position in the sequence.
    assert _speed_dp(v, np.zeros(3), 1.0).tolist() == [True, False, True]
    assert _speed_dp(v, np.array([0.0, 1.0, 100.0]), np.inf).tolist() == [True, False, True]


def test_speed_is_the_central_difference():
    x = np.array([0.0, 10.0, 40.0, 40.0])
    t = np.array([0.0, 10.0, 20.0, 30.0])
    assert _speeds(x, np.zeros(4), t).tolist() == [1.0, 2.0, 1.5, 0.0]


def test_undefined_speeds_take_the_nearest_defined_neighbour():
    assert _nearest(np.array([0, 2]), 3).tolist() == [0, 0, 1]
    assert _nearest(np.array([2]), 4).tolist() == [0, 0, 0, 0]
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert _speeds(x, np.zeros(4), np.array([0.0, 0.0, 0.0, 10.0])).tolist() == [
        0.2, 0.2, 0.2, 0.1]
    assert _speeds(x, np.zeros(4), np.zeros(4)).tolist() == [0.0] * 4


# ------------------------------------------------------------------- times


def test_times_may_repeat_but_not_decrease():
    x, y, t = _vessel()
    t2 = t.copy()
    t2[10] = t2[9]
    assert two_stage_cpls_select(x, y, t2, 100.0, 1.0, []).mask.sum() >= 2
    t3 = t.copy()
    t3[10] = t3[9] - 1.0
    with pytest.raises(ValueError, match="decrease"):
        two_stage_cpls_select(x, y, t3, 100.0, 1.0, [])
    with pytest.raises(ValueError):
        two_stage_cpls_select(x, y, t[:-1], 100.0, 1.0, [])
    with pytest.raises(ValueError):
        two_stage_cpls_select(x, y, t, 100.0, -1.0, [])
    with pytest.raises(TypeError):
        two_stage_cpls_select(x, y, t, 100.0, 1.0, [], speed=True)


def test_edge_cases_and_summary():
    for n in (0, 1, 2):
        sel = two_stage_cpls_select(np.arange(n, dtype=float), np.zeros(n),
                                    np.arange(n, dtype=float), 1.0, 1.0, [])
        assert sel.mask.all() and sel.mask.size == n
    x, y, t = _vessel()
    sel = two_stage_cpls_select(x, y, t, 100.0, 1.54, [], stay_radius=50.0,
                                stay_duration=300.0)
    assert isinstance(sel, TwoStageSelection)
    text = sel.summary().splitlines()
    assert text[0].startswith(f"{x.size} points -> {int(sel.mask.sum())} kept")
    assert "1 (25 points)" in text[1]
    assert [line.split()[0] for line in text[2:6]] == ["stay", "+", "+", "+"]


# ---------------------------------------------------------------- dataframes


def _frame(n=120, seed=0, crs=32630):
    x, y, t, stays = _with_stays(n=n, seed=seed)
    return gpd.GeoDataFrame(
        {"speed": np.arange(n, dtype=float), "vid": 1,
         "status": np.where(stays, 1.0, 0.0)},
        geometry=gpd.points_from_xy(x, y),
        index=pd.to_datetime(1_454_000_000 + t, unit="s"),
        crs=crs,
    )


def _barriers_of(gdf, seed=0):
    return _barriers_for(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), seed)


def test_compress_keeps_every_column_and_moves_only_stay_ends():
    gdf = _frame()
    barriers = _barriers_of(gdf)
    out = two_stage_cpls_compress(gdf, 150.0, 1.0, barriers, status_column="status",
                                  filter_window=None)
    assert list(out.columns) == list(gdf.columns) + ["stay"]
    assert out.index.isin(gdf.index).all() and len(out) < len(gdf)
    assert out.crs == gdf.crs
    assert out.stay.sum() > 0
    rows = gdf.loc[out.index]
    for col in gdf.columns.drop("geometry"):
        assert (out[col] == rows[col]).all()
    same = out.geometry.geom_equals_exact(rows.geometry, 0.0)
    assert same[~out.stay.to_numpy()].all()
    assert not same[out.stay.to_numpy()].all()
    runs = out[out.stay]
    assert runs.geometry.x.nunique() <= len(runs) // 2 + 1
    plain = two_stage_cpls_compress(gdf, 150.0, 1.0, barriers, status_column="status",
                                    stay_column=None, filter_window=None)
    assert "stay" not in plain.columns and plain.index.equals(out.index)


def test_compress_matches_select_and_time_column():
    gdf = _frame()
    barriers = _barriers_of(gdf)
    t = (gdf.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    sel = two_stage_cpls_select(
        gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), t.to_numpy(), 150.0, 1.0,
        barriers, status=gdf.status.to_numpy(), speed_window=None)
    out = two_stage_cpls_compress(gdf, 150.0, 1.0, barriers, status_column="status",
                                  filter_window=None)
    assert out.index.equals(gdf.index[sel.mask])
    by_column = gdf.reset_index(drop=True).assign(clock=t.to_numpy())
    again = two_stage_cpls_compress(by_column, 150.0, 1.0, barriers,
                                    time_column="clock", status_column="status",
                                    filter_window=None)
    assert again.index.tolist() == np.flatnonzero(sel.mask).tolist()


def test_compress_by_group():
    a, b = _frame(seed=0), _frame(seed=1)
    b["vid"] = 2
    barriers = _barriers_of(a) + _barriers_of(b, seed=1)
    out = two_stage_cpls_compress(pd.concat([a, b]), 150.0, 1.0, barriers, by="vid",
                                  status_column="status")
    assert set(out.vid) == {1, 2}
    assert out[out.vid == 1].equals(
        two_stage_cpls_compress(a, 150.0, 1.0, barriers, status_column="status"))
    assert out[out.vid == 2].equals(
        two_stage_cpls_compress(b, 150.0, 1.0, barriers, status_column="status"))


def test_compress_refuses_geographic_coordinates_and_unknown_relations():
    with pytest.raises(ValueError, match="geographic"):
        two_stage_cpls_compress(_frame(crs=4326), 50.0, 1.0, [])
    wrong = gpd.GeoSeries([shapely.Point(0, 0).buffer(10)], crs=3035)
    with pytest.raises(ValueError, match="CRS"):
        two_stage_cpls_compress(_frame(), 50.0, 1.0, wrong)
    with pytest.raises(TypeError):
        two_stage_cpls_compress(_frame(), 50.0, 1.0, [], speed=True)


@pytest.mark.movingpandas
def test_compress_movingpandas_objects():
    mpd = pytest.importorskip("movingpandas")
    gdf = _frame()
    barriers = Barriers(_barriers_of(gdf), crs=32630)
    traj = mpd.Trajectory(gdf, traj_id="v1")
    out = two_stage_cpls_compress(traj, 150.0, 1.0, barriers, status_column="status")
    assert isinstance(out, mpd.Trajectory) and out.id == "v1"
    assert "stay" in out.df.columns and len(out.df) < len(traj.df)
    coll = mpd.TrajectoryCollection([traj, mpd.Trajectory(_frame(seed=1), traj_id="v2")])
    out_coll = two_stage_cpls_compress(coll, 150.0, 1.0, barriers, status_column="status")
    assert isinstance(out_coll, mpd.TrajectoryCollection)
    assert len(out_coll.trajectories) == 2


# ------------------------------------------------------------- sample data


@pytest.mark.data
def test_two_stage_on_the_sample_fleet():
    from catraj.datasets import load_ais, load_coastline

    ais = load_ais()    # sorted by (mmsi, unix_time)
    land = Barriers(load_coastline())
    two_stage = 0
    for _, g in ais.groupby("mmsi", sort=False):
        x, y = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
        sel = two_stage_cpls_select(
            x, y, g.unix_time.to_numpy(float), 1_000, 1.54, land,
            status=g.navigational_status.to_numpy(float), stay_radius=100.0,
            stay_duration=600.0,
        )
        two_stage += sel.mask.sum()
        assert not (sel.mask & sel.stays & ~sel.stay_ends).any()
        # The signature is that of the collapsed track: no crossing it did
        # not have.
        assert not segment_crossings(
            sel.x, sel.y, land, keep=sel.mask, new_only=True).any()
    # The jitter of moored vessels keeps most points under plain CPLS with
    # every relation.  With the stays collapsed, under a tenth: measured on
    # the 2015-11-09/10 sample, 15,674 of 237,318 (6.6 %).
    assert two_stage < len(ais) // 10


# ------------------------------------------------- filtering (P1, P2)


@pytest.mark.parametrize("window", [30.0, 90.0, 400.0])
@pytest.mark.parametrize("seed", range(4))
def test_the_speed_series_median_matches_the_literal_algorithm(seed, window):
    x, y, t, stays = _with_stays(seed=seed)
    parts = _barriers_for(x, y, seed)
    for speed_epsilon in (0.5, 2.0):
        sel = two_stage_cpls_select(x, y, t, 150.0, speed_epsilon, parts,
                                    stays=stays, speed_window=window)
        _assert_same(sel, _reference_two_stage(
            x, y, t, 150.0, speed_epsilon, parts, stays, speed_window=window))


@pytest.mark.parametrize("seed", range(3))
def test_the_speed_median_with_repeated_times_matches_the_literal_algorithm(seed):
    x, y, t, stays = _with_stays(seed=seed, repeated_times=True)
    parts = _barriers_for(x, y, seed)
    sel = two_stage_cpls_select(x, y, t, 150.0, 1.0, parts, stays=stays,
                                speed_window=60.0)
    _assert_same(sel, _reference_two_stage(x, y, t, 150.0, 1.0, parts, stays,
                                           speed_window=60.0))


def test_the_speed_window_defaults_to_30_seconds_and_none_switches_it_off():
    x, y, t, stays = _with_stays(seed=1, repeated_times=True)
    parts = _barriers_for(x, y, 1)
    default = two_stage_cpls_select(x, y, t, 150.0, 0.5, parts, stays=stays)
    thirty = two_stage_cpls_select(x, y, t, 150.0, 0.5, parts, stays=stays,
                                   speed_window=30.0)
    off = two_stage_cpls_select(x, y, t, 150.0, 0.5, parts, stays=stays,
                                speed_window=None)
    assert np.array_equal(default.mask, thirty.mask)
    _assert_same(off, _reference_two_stage(x, y, t, 150.0, 0.5, parts, stays))
    assert not np.array_equal(off.mask, two_stage_cpls_select(
        x, y, t, 150.0, 0.5, parts, stays=stays, speed_window=400.0).mask)


def test_a_speed_spike_is_filtered_out_of_the_speed_stage():
    n = 60
    x = np.arange(n) * 10.0
    y = np.zeros(n)
    x[30] += 400.0  # one position spike: the speeds either side jump
    t = np.arange(n) * 10.0
    raw = two_stage_cpls_select(x, y, t, 1e6, 1.0, [], speed_window=None)
    smooth = two_stage_cpls_select(x, y, t, 1e6, 1.0, [], speed_window=50.0)
    assert raw.speed.sum() > 2
    assert smooth.speed.sum() <= raw.speed.sum()
    assert smooth.speed.sum() == 2  # only the ends: the median removes it


def test_the_speed_window_is_validated():
    x, y, t, _ = _with_stays()
    for bad in (0.0, -5.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="window"):
            two_stage_cpls_select(x, y, t, 100.0, 1.0, [], speed_window=bad)


def _noisy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * 8.0 + rng.normal(scale=25.0, size=n)
    y = rng.normal(scale=25.0, size=n)
    return gpd.GeoDataFrame(
        {"vid": 1}, geometry=gpd.points_from_xy(x, y),
        index=pd.to_datetime(1_454_000_000 + np.arange(n) * 5.0, unit="s"),
        crs=32630,
    )


def test_compress_filters_positions_then_the_speed_series_with_one_window():
    from catraj.preprocessing import denoise

    gdf = _noisy_frame()
    t = (gdf.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    clean = denoise(gdf, 30)
    sel = two_stage_cpls_select(
        clean.geometry.x.to_numpy(), clean.geometry.y.to_numpy(), t.to_numpy(),
        100.0, 1.0, [], speed_window=30)
    out = two_stage_cpls_compress(gdf, 100.0, 1.0, [])
    assert out.index.equals(gdf.index[sel.mask])
    assert np.array_equal(out.geometry.x.to_numpy(), clean.geometry.x.to_numpy()[sel.mask])
    explicit = two_stage_cpls_compress(gdf, 100.0, 1.0, [], filter_window=30)
    assert out.equals(explicit)


def test_compress_without_filtering_is_the_unfiltered_algorithm():
    gdf = _noisy_frame()
    t = (gdf.index - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    sel = two_stage_cpls_select(
        gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), t.to_numpy(),
        100.0, 1.0, [], speed_window=None)
    out = two_stage_cpls_compress(gdf, 100.0, 1.0, [], filter_window=None)
    assert out.index.equals(gdf.index[sel.mask])
    assert out.geometry.geom_equals_exact(gdf.geometry[sel.mask], 0.0).all()
    filtered = two_stage_cpls_compress(gdf, 100.0, 1.0, [])
    assert not filtered.index.equals(out.index)


def test_compress_filtering_needs_times():
    gdf = _noisy_frame().reset_index(drop=True)
    with pytest.raises(ValueError, match="time_column"):
        two_stage_cpls_compress(gdf, 100.0, 1.0, [])
    gdf["clock"] = np.arange(len(gdf)) * 5.0
    two_stage_cpls_compress(gdf, 100.0, 1.0, [], time_column="clock")
    two_stage_cpls_compress(gdf, 100.0, 1.0, [], filter_window=None)  # numeric index


@pytest.mark.movingpandas
def test_compress_passes_the_filter_window_to_trajectories_and_collections():
    mpd = pytest.importorskip("movingpandas")
    gdf = _noisy_frame()
    traj = mpd.Trajectory(gdf, traj_id="v1")
    coll = mpd.TrajectoryCollection([traj])
    for window in (None, 30):
        want = two_stage_cpls_compress(gdf, 100.0, 1.0, [], filter_window=window)
        got = two_stage_cpls_compress(traj, 100.0, 1.0, [], filter_window=window)
        assert got.df.index.equals(want.index)
        again = two_stage_cpls_compress(coll, 100.0, 1.0, [], filter_window=window)
        assert again.trajectories[0].df.index.equals(want.index)
