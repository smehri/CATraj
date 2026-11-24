"""Context-aware Piecewise Linear Segmentation (CPLS) -- [Mehri2021] section IV-A.

Douglas--Peucker keeps the points that carry spatial information and is blind
to everything else.  Near a coast that blindness has a cost: a chord that
replaces a bend in the track can cut across a headland the vessel sailed
round, put an island on the wrong side, or cross the track itself.  CPLS
refuses to drop a point when dropping it would change the trajectory's
topological signature.

The method, as Algorithm 1 states it::

    for the points between T1 and Tn, find the one furthest from the segment
        T1--Tn (Eq. 4)
    if that distance exceeds the threshold:
        split there and recurse on both halves                (lines 9-12)
    else:
        for each point Tj from T2 to Tn-1, front to back:
            remove Tj
            if the topological signature is still consistent, move on
            otherwise add Tj back                             (lines 13-20)

Lines 2-12 are exactly Douglas--Peucker.  Lines 13-20 run once inside each
segment that Douglas--Peucker would have collapsed.  So CPLS is
Douglas--Peucker followed by a topological pass inside each of its segments,
and the result is always a superset of the Douglas--Peucker result.  This
mirrors CADP ("DP first, then add points back"), which is why it sits next to
:mod:`.cadp`.

The paper takes the topological signature from Tienaah et al. [63]: "the
topological signature of the original and simplified trajectory should be the
same [63]", and CPLS is built "with the idea of Tienaah [63]".  Every relation
of Tienaah's model the author chose is implemented, each can be switched off,
and all are on by default.  The decisions below were taken with the author
(2026-11-23); the sentence behind each is quoted.  D10 and D11 are
implementation details neither source addresses.

Terms
-----

* **Unit.**  A chord with the sub-path of the original it replaces.  While the
  pass runs, the trajectory ``Tc`` of line 15 is: the final units before the
  last retained point ``a``, the candidate unit ``a -> k``, and the original
  segments after ``k``, each a unit of its own.
* **Neighbour.**  A barrier part that intersects the convex hull of the
  candidate's sub-path: consistency "may be violated if these geometries are
  found within the convex hull of the set of vertices forming the trajectory"
  (the paper, after [63] and [65]).  Each part of a multi-part barrier is a
  neighbour of its own.

Decisions, and the text behind them
-----------------------------------

* **D1. Stage 1 is** :func:`.douglas_peucker`, called, not reimplemented.
  Lines 2-12 are Douglas--Peucker.  Line 7, ``imax = 1``, is read as
  ``imax = i`` and line 4's range as the interior points -- the same typo and
  the same reading as in the CADP paper, documented in :mod:`.douglas_peucker`.
* **D2. Stage 2 runs inside each Douglas--Peucker segment, front to back.**
  "When there is no error greater than the given threshold, next for all other
  trajectory vertices p2, ..., p(n-1), it eliminates the first point p2.  After
  that, it checks the consistency of the topological signature.  If removing
  p2 destroys the topological signature, it adds it to the trajectory, again.
  Otherwise, it eliminates p2 and continues the algorithm with the next
  point."  Removing ``Tj`` leaves ``Tc = [T1 ... Tj-1, Tj+1 ... Tn]`` (line
  15), so the chord tested runs from the last retained point to ``Tj+1``, and
  the signature is that of the whole ``Tc`` at that moment.
* **D3. The signature is Tienaah's, relation by relation.**  A chord is
  consistent if it keeps every relation switched on, tested in this order;
  the first one broken is recorded in :attr:`CplsSelection.rules`.
* **D4. Topology: the same DE-9IM relation.**  "Geometric relations are
  computed using the Dimensionally Extended nine-Intersection Model ... a
  simplified road going through a park is valid if the original geometry
  shares the same relation" (Tienaah 2018, section 2.3.1, published as [63];
  heuristic 3.9; section 4.5.1.2 for trajectories and islands).  The chord's
  DE-9IM matrix with each neighbour must equal the sub-path's.  So a chord may
  not cross land the sub-path did not cross -- "if there are some
  intersections generated through the compression process, the generated
  segment is void" (the paper) -- and, as strictly, a sub-path that touched or
  crossed land keeps a point where the chord would not.
* **D5. Direction: the same nine cells.**  "The simplified geometry must
  conform to the same quadrant relation as the original to avoid side (left,
  right, up, down, diagonal, and interior) conflicts" (Tienaah 2018, section
  2.3.2).  Around each neighbour, the nine cells are its minimum bounding box
  *within the hull* (``part & hull``) and the eight regions around it; the
  chord must pass through the same cells as the sub-path.  A line passes
  through a cell when it runs through the cell's interior for a positive
  length: running along a cell boundary, or a stretch shorter than 1e-9 of a
  segment (rounding), does not count.  A vessel that went north of an island
  and a chord that would go south of it register different cells.  The rule
  is strict: a chord that cuts a bend can miss a cell without changing sides,
  and that keeps a point too.
* **D6. Self-intersection: the chord meets the same units as its sub-path.**
  Tienaah's constraints "avoid introducing new self-intersections as a result
  of simplification" and "preserve non-planar intersections" (2018, section
  3.2, items 4-5; heuristics 3.4/3.5 and 4.3 for new crossings, 3.2 and 4.1
  for lost ones).  For every other unit of ``Tc``, the chord meets the unit's
  chord if and only if the sub-path meets the unit's sub-path; for a unit
  that shares a vertex with the candidate, meeting at that vertex does not
  count.  Meetings within one unit are ignored, as for trajectories in
  section 4.5.1: "we ignore self-intersections within aggregation units since
  their simplification error is at most epsilon".  The two halves are
  separate switches, ``avoid_self_intersection`` and
  ``keep_self_intersection``, like Tienaah's "avoid new self-intersects" and
  "keep non-planar self-intersects".  Tienaah's optional relax distance on
  where a kept crossing may move (heuristics 3.3, 4.2) is not implemented:
  a crossing must survive, wherever it lands.
* **D7. Inversion: no reflection at a shared vertex.**  "Consecutive segment
  of L' should not invert the topological relation in L" (section 3.2, item
  6), tested as heuristics 3.7/3.8: units ``i..k`` and ``k..j`` are
  inconsistent if ``Pj`` lies inside the hull of ``i..k`` or ``Pi`` inside the
  hull of ``k..j``.  The candidate is tested against the final unit before it
  and the original segment after it.  "Inside" is the hull's interior.
* **D8. The offset distance is tested once, by Douglas--Peucker.**  Lines
  13-20 test only the topological signature.  A point added back is not
  followed by a new offset test, so a dropped point is within ``epsilon`` of
  its Douglas--Peucker segment, but not necessarily of the segment between its
  final neighbours -- as in :mod:`.cadp`'s ``post`` mode.
* **D9. The pass is the greedy of lines 14-20, as written.**  One point at a
  time, kept or dropped for good.  It does not search for the fewest points
  (Tienaah instead reinstates the vertex with the largest offset; the paper's
  own pass is kept).
* **D10. Barriers and positions share one projected CRS.**  A mismatch
  raises; nothing is reprojected silently.
* **D11. Repeated positions** give zero-length segments, and a zero-length
  segment (or chord, or sub-path that never moves) is the point it sits at.
  GEOS reports a two-point line with equal ends disjoint from a line through
  its point, so such segments are made points before any test.

What it costs
-------------

D6 and D7 act on the track itself, with or without barriers.  On raw AIS the
GPS jitter of a moored vessel crosses itself thousands of times, and keeping
those crossings keeps most of its points: on the 24-hour sample at 1 km, the
barrier relations alone keep 607 of 106,894 points, all relations 43,074
(unfiltered, ``filter_window=None``; with the default 30 s median, 604 and
35,020; about 3 s and 40 s unfiltered).
The paper's two-stage CPLS replaces stay points by their centroid first, which
removes that jitter (see :mod:`.two_stage`); otherwise switch D6 and D7 off for
raw data with long stops if compression matters more than the track's own
topology.

Filtering first (decided with the author, 2026-11-24)
-----------------------------------------------------

:func:`cpls_compress`, the dataframe-level step, first denoises the positions
with :func:`catraj.preprocessing.denoise`: a centred median over
``filter_window`` seconds, 30 by default; ``None`` switches it off and gives
the unfiltered result exactly.  CPLS then runs on the filtered track, and the
result rows carry the filtered geometry.  Time comes from ``time_column`` or a
DatetimeIndex (CPLS itself does not use time); any other index raises a
``ValueError``.  The guarantees below hold for the filtered track; the
filtered track is not the raw one.  :func:`cpls_select`, the algorithm on
arrays, filters nothing.

What the result guarantees
--------------------------

* The result contains every Douglas--Peucker point; with every relation off,
  it equals Douglas--Peucker.
* Each final unit has the same DE-9IM relation (D4) and the same nine cells
  (D5) as its sub-path, for every neighbour; in particular compression never
  introduces a crossing (:func:`segment_crossings` with ``new_only=True``
  measures it), and a path that never touches a barrier stays clear of it.
* Every two final units meet if and only if their sub-paths do, a shared
  vertex aside (D6); no two consecutive units invert (D7).
* The first and last points are retained.

Implementation
--------------

The barriers are indexed once (:class:`Barriers`).  With only the barrier
relations on, a Douglas--Peucker segment whose bounding box meets no barrier
is skipped: every chord lies inside that box.  Otherwise the greedy goes point
by point.  A chord whose hull meets no barrier passes D4 and D5 at once;
otherwise the neighbours are clipped clear of the hull, compared by DE-9IM,
and their nine cells computed for all segments of the sub-path at once.  For
D6 every pair of original segments that meet is found once, by bounding box
and a vectorised orientation test; the meetings of a sub-path are then
looked up, and a chord is tested against later segments and earlier chords
the same way.

Coordinates must be projected -- ``epsilon`` is a distance in their units.

References
----------
[Mehri2021] Mehri, S., Alesheikh, A. A., Basiri, A. (2021).  A Contextual
    Hybrid Model for Vessel Movement Prediction.  IEEE Access 9, 45600-45613.
[63] Tienaah, T., Stefanakis, E., Coleman, D. (2015).  Contextual
    Douglas-Peucker simplification.  Geomatica 69(3), 327-338.  Chapter 2 of
    Tienaah, T. (2018), Line Simplification Under Spatial Constraints, UNB
    Technical Report 314, whose chapters 3 and 4 add D6 and D7.
[65] Saalfeld, A. (1999).  Topologically consistent line simplification with
    the Douglas-Peucker algorithm.  Cartography and GIS 26(1), 7-18.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely

from .._crs import _check_crs
from ..preprocessing.filtering import denoise
from .douglas_peucker import douglas_peucker

__all__ = [
    "Barriers",
    "CplsSelection",
    "segment_crossings",
    "cpls_select",
    "cpls_compress",
    "RULES",
]

# A line "passes through" a cell of the direction model only for longer than
# this fraction of a segment; shorter overlaps are rounding at a cell edge.
_ROUNDING = 1e-9
_CELL_BITS = 1 << np.arange(9)
# Segments tested for self-intersection pairs at a time.
_PAIR_BLOCK = 4096


def _segments(x0, y0, x1, y1):
    """Segments as geometries, a zero-length one as the point it is (D11).

    GEOS does not treat a two-point line with equal ends as a point: it
    reports it disjoint from a line through that point.  A point is right.
    """
    x0, y0, x1, y1 = (
        np.atleast_1d(np.asarray(v, dtype=float)) for v in (x0, y0, x1, y1)
    )
    coords = np.empty((x0.size, 2, 2))
    coords[:, 0, 0], coords[:, 0, 1] = x0, y0
    coords[:, 1, 0], coords[:, 1, 1] = x1, y1
    out = shapely.linestrings(coords)
    still = (x0 == x1) & (y0 == y1)
    if still.any():
        out[still] = shapely.points(x0[still], y0[still])
    return out


def _path(xs, ys):
    """A sub-path as one geometry; a point if it never moves (D11)."""
    if (xs == xs[0]).all() and (ys == ys[0]).all():
        return shapely.points(xs[0], ys[0])
    return shapely.linestrings(np.column_stack([xs, ys]))


class Barriers:
    """Barrier geometries, indexed once so many trajectories can be tested.

    Parameters
    ----------
    geometries : shapely geometry, iterable of geometries, GeoSeries or GeoDataFrame
        Lines (a shoreline) and polygons (land) both work.  ``None`` entries
        and empty geometries are dropped; an empty input means "no barriers".
    crs : optional
        Used when ``geometries`` carries no CRS of its own.  A ``GeoSeries`` or
        ``GeoDataFrame`` brings its own.
    """

    def __init__(self, geometries, crs=None):
        own_crs = getattr(geometries, "crs", None)
        self.crs = own_crs if own_crs is not None else crs

        if isinstance(geometries, shapely.geometry.base.BaseGeometry):
            items = [geometries]
        elif hasattr(geometries, "geometry"):  # GeoSeries, GeoDataFrame
            items = list(geometries.geometry.values)
        elif isinstance(geometries, (str, bytes)) or not hasattr(
            geometries, "__iter__"
        ):
            raise TypeError(
                "barriers must be shapely geometries, a GeoSeries or a "
                f"GeoDataFrame; got {type(geometries).__name__}"
            )
        else:
            items = list(geometries)

        items = [g for g in items if g is not None]
        for g in items:
            if not isinstance(g, shapely.geometry.base.BaseGeometry):
                raise TypeError(
                    "barriers must be shapely geometries; "
                    f"found a {type(g).__name__} entry"
                )
        array = np.empty(len(items), dtype=object)
        array[:] = items
        parts = shapely.get_parts(array)
        parts = parts[~shapely.is_empty(parts)]
        shapely.prepare(parts)
        self._parts = parts
        self._tree = shapely.STRtree(parts)

    def __len__(self):
        return int(self._parts.size)

    def hits(self, x0, y0, x1, y1):
        """Whether each segment ``(x0, y0)--(x1, y1)`` intersects a barrier.

        Returns
        -------
        ndarray of bool
            One value per segment; touching counts as intersecting.
        """
        x0, y0, x1, y1 = (
            np.asarray(v, dtype=float).ravel() for v in (x0, y0, x1, y1)
        )
        out = np.zeros(x0.size, dtype=bool)
        if x0.size == 0 or len(self) == 0:
            return out
        lines = _segments(x0, y0, x1, y1)
        line_index, part_index = self._tree.query(lines)
        if line_index.size:
            touches = shapely.intersects(self._parts[part_index], lines[line_index])
            out[line_index[touches]] = True
        return out

    def _boxes_meet(self, xmin, ymin, xmax, ymax):
        """Whether each bounding box meets the bounding box of some barrier."""
        out = np.zeros(np.size(xmin), dtype=bool)
        if out.size == 0 or len(self) == 0:
            return out
        box_index, _ = self._tree.query(shapely.box(xmin, ymin, xmax, ymax))
        out[box_index] = True
        return out


def _as_barriers(barriers):
    return barriers if isinstance(barriers, Barriers) else Barriers(barriers)


def _original_hits(x, y, barriers):
    return barriers.hits(x[:-1], y[:-1], x[1:], y[1:])


def _prefix_counts(hit):
    cs = np.zeros(hit.size + 1, dtype=np.int64)
    np.cumsum(hit, out=cs[1:])
    return cs


def segment_crossings(x, y, barriers, keep=None, new_only=False):
    """Which segments of a (simplified) trajectory intersect a barrier.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    barriers : Barriers or anything :class:`Barriers` accepts
    keep : array-like of bool, optional
        The retained points.  All points when omitted.
    new_only : bool
        Count a segment only if the original sub-path it replaces does *not*
        intersect a barrier: ``hit(a, b) and not orig(a, b)``.  This is the
        evaluation measure for CPLS, which never leaves one.

    Returns
    -------
    ndarray of bool
        One value per segment of the polyline through the retained points, so
        ``keep.sum() - 1`` of them (``n - 1`` without ``keep``); empty for
        fewer than two points.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have equal length; got {x.size}, {y.size}")
    barriers = _as_barriers(barriers)
    if keep is None:
        idx = np.arange(x.size)
    else:
        keep = np.asarray(keep, dtype=bool).ravel()
        if keep.size != x.size:
            raise ValueError("keep must have one value per point")
        idx = np.flatnonzero(keep)
    if idx.size < 2:
        return np.zeros(0, dtype=bool)
    a, b = idx[:-1], idx[1:]
    hit = barriers.hits(x[a], y[a], x[b], y[b])
    if new_only and hit.any():
        cs = _prefix_counts(_original_hits(x, y, barriers))
        hit &= (cs[b] - cs[a]) == 0
    return hit


#: The relations of the topological signature, in the order they are tested.
RULES = ("topology", "direction", "self_intersection", "inversion")


@dataclass
class CplsSelection:
    """Which points CPLS keeps, and why.

    ``dp`` is the Douglas--Peucker stage's choice; ``topology`` the points
    stage 2 retained on top of it to keep the topological signature, so it
    never overlaps ``dp``.  ``rules`` maps each relation in :data:`RULES` to
    the points retained because that relation, the first one tested to fail,
    would have broken.
    """

    dp: np.ndarray
    topology: np.ndarray
    rules: dict = field(default_factory=dict)

    @property
    def mask(self):
        return self.dp | self.topology

    @property
    def added(self):
        """Points kept for topology that Douglas--Peucker alone would drop."""
        return self.topology

    def summary(self):
        """Point counts, for reading at a glance."""
        n = int(self.dp.size)
        kept = int(self.mask.sum())
        lines = [
            f"{n} points -> {kept} kept "
            f"({100.0 * (1 - kept / n) if n else 0.0:.2f}% compression)",
            f"  Douglas-Peucker      {int(self.dp.sum())}",
            f"  + topology           {int(self.topology.sum())}",
        ]
        for name, flagged in self.rules.items():
            lines.append(f"      {name:<18} {int(flagged.sum())}")
        return "\n".join(lines)


def _band_intervals(a0, a1, lo, hi):
    """Where along each segment the coordinate is below, inside and above a band.

    For segments from ``a0`` to ``a1`` and the band edges ``lo <= hi``, the
    parameter interval ``[t_start, t_end]`` (``t`` from 0 to 1 along the
    segment) spent strictly below ``lo``, strictly between the edges, and
    strictly above ``hi``.  Shape ``(m, 3, 2)``; an empty interval has
    ``t_end <= t_start``.
    """
    d = a1 - a0
    moving = d != 0
    step = np.where(moving, d, 1.0)
    t_lo = (lo - a0) / step
    t_hi = (hi - a0) / step
    up = d > 0
    out = np.empty(a0.shape + (3, 2))
    out[:, 0, 0] = np.where(up, 0.0, t_lo)
    out[:, 0, 1] = np.where(up, t_lo, 1.0)
    out[:, 1, 0] = np.where(up, t_lo, t_hi)
    out[:, 1, 1] = np.where(up, t_hi, t_lo)
    out[:, 2, 0] = np.where(up, t_hi, 0.0)
    out[:, 2, 1] = np.where(up, 1.0, t_hi)
    # A coordinate that does not change stays in one band, or on an edge.
    still = ~moving
    inside = (a0 < lo, (a0 > lo) & (a0 < hi), a0 > hi)
    for band, here in enumerate(inside):
        out[still, band, 0] = 0.0
        out[still, band, 1] = here[still].astype(float)
    return np.clip(out, 0.0, 1.0)


def _direction_cells(x, y, box):
    """Tienaah's nine-cell direction signature of a polyline around ``box``.

    Bit ``3 * i + j`` is set when the line runs through the interior of the
    cell in column ``i`` (west of, across, east of the box) and row ``j``
    (south of, across, north of the box) for more than a rounding fraction of
    some segment.  Zero-length segments (repeated positions) pass through
    nothing.
    """
    xmin, ymin, xmax, ymax = box
    moving = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    x0, x1, y0, y1 = x[:-1][moving], x[1:][moving], y[:-1][moving], y[1:][moving]
    tx = _band_intervals(x0, x1, xmin, xmax)
    ty = _band_intervals(y0, y1, ymin, ymax)
    start = np.maximum(tx[:, :, None, 0], ty[:, None, :, 0])
    end = np.minimum(tx[:, :, None, 1], ty[:, None, :, 1])
    visited = ((end - start) > _ROUNDING).any(axis=0)
    return int(visited.ravel() @ _CELL_BITS)


def _segments_meet(ax, ay, bx, by, cx, cy, dx, dy):
    """Whether segments ``a-b`` and ``c-d`` meet (touching counts); vectorised.

    Orientation tests, with the collinear and zero-length cases settled by a
    bounding-box check on the end point that lies on the other segment's line.
    """
    def orient(px, py, qx, qy, rx, ry):
        return np.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))

    def on(px, py, qx, qy, rx, ry):
        return ((np.minimum(px, qx) <= rx) & (rx <= np.maximum(px, qx))
                & (np.minimum(py, qy) <= ry) & (ry <= np.maximum(py, qy)))

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    meet = (o1 * o2 < 0) & (o3 * o4 < 0)
    meet |= (o1 == 0) & on(ax, ay, bx, by, cx, cy)
    meet |= (o2 == 0) & on(ax, ay, bx, by, dx, dy)
    meet |= (o3 == 0) & on(cx, cy, dx, dy, ax, ay)
    meet |= (o4 == 0) & on(cx, cy, dx, dy, bx, by)
    return meet


def _folded(ax, ay, vx, vy, bx, by):
    """Whether segments ``v-a`` and ``v-b``, sharing ``v``, meet beyond it.

    Only if they are collinear and point the same way; a zero-length one is
    the point ``v`` and meets the other only there.
    """
    ux, uy, wx, wy = ax - vx, ay - vy, bx - vx, by - vy
    return (ux * wy - uy * wx == 0) & (ux * wx + uy * wy > 0)


class _Stage2:
    """Stage 2 over one trajectory: lines 14-20 with the full signature.

    The state is the trajectory ``Tc`` of Algorithm 1 line 15 as the pass goes
    along: the final units before the last retained point ``a`` (the prefix),
    the candidate unit ``a -> k``, and the original segments after ``k``.  A
    unit is a chord with the sub-path it replaces; an original segment is a
    unit of its own.
    """

    def __init__(self, x, y, barriers, rules):
        self.x, self.y = x, y
        self.n = x.size
        self.barriers = barriers
        self.topology, self.direction = rules["topology"], rules["direction"]
        self.avoid_self = rules["avoid_self_intersection"]
        self.keep_self = rules["keep_self_intersection"]
        self.inversion = rules["inversion"]
        self.use_barriers = (self.topology or self.direction) and len(barriers) > 0
        self.use_self = self.avoid_self or self.keep_self
        self.everywhere = self.use_self or self.inversion
        # Retained points so far; final unit u runs from retained[u] to
        # retained[u + 1].
        self.retained = np.zeros(self.n, dtype=np.intp)
        self.count = 1
        # Bounding box of each final unit's chord, for quick rejection.
        self.unit_box = np.empty((4, self.n))
        self.prev_hull = None
        if self.use_self:
            self.segments = _segments(x[:-1], y[:-1], x[1:], y[1:])
            self.segment_tree = shapely.STRtree(self.segments)
            # Every pair of original segments that meet, grouped by the
            # first: candidates by bounding box, then the exact test, a block
            # of segments at a time so that a tangle of GPS jitter cannot
            # exhaust memory.
            found = []
            for lo in range(0, self.n - 1, _PAIR_BLOCK):
                block = np.arange(lo, min(lo + _PAIR_BLOCK, self.n - 1))
                bi, j = self.segment_tree.query(self.segments[block])
                i = block[bi]
                off = i != j
                i, j = i[off], j[off]
                meet = _segments_meet(x[i], y[i], x[i + 1], y[i + 1],
                                      x[j], y[j], x[j + 1], y[j + 1])
                found.append(j[meet][np.argsort(i[meet], kind="stable")])
                found.append(i[meet])
            self.pair_j = np.concatenate(found[0::2]) if found else np.empty(0, int)
            firsts = np.concatenate(found[1::2]) if found else np.empty(0, int)
            self.pair_start = np.concatenate(
                [[0], np.cumsum(np.bincount(firsts, minlength=self.n - 1))]
            )

    # ------------------------------------------------------------ the pass

    def run(self, dp, keep, rules):
        x, y = self.x, self.y
        kept = np.flatnonzero(dp)
        starts, ends = kept[:-1], kept[1:]
        if self.everywhere:
            todo = ends - starts >= 2
        else:
            # Only barriers matter: a segment whose box meets none drops
            # everything between its ends (the end point is outside
            # reduceat's range for all but the last segment).
            xmin = np.minimum(np.minimum.reduceat(x, starts), x[ends])
            xmax = np.maximum(np.maximum.reduceat(x, starts), x[ends])
            ymin = np.minimum(np.minimum.reduceat(y, starts), y[ends])
            ymax = np.maximum(np.maximum.reduceat(y, starts), y[ends])
            meets = self.barriers._boxes_meet(xmin, ymin, xmax, ymax)
            todo = meets & (ends - starts >= 2) & self.use_barriers
        for s, e, run in zip(starts, ends, todo):
            if run:
                self._greedy(int(s), int(e), keep, rules)
            elif self.everywhere:
                self._finalize(int(s), int(e))

    def _greedy(self, start, end, keep, rules):
        a, k = start, start + 2
        self._begin(a)
        while k <= end:
            failed = self._check(a, k)
            if failed is None:
                k += 1
                continue
            j = k - 1
            keep[j] = True
            rules[failed][j] = True
            self._finalize(a, j)
            a, k = j, j + 2
            self._begin(a)
        self._finalize(a, end)

    def _begin(self, a):
        self.box, self.box_upto = None, a
        self.hull_xy, self.hull_upto, self.hull_at = None, a, -1
        self.sub_i = self.sub_j = np.empty(0, dtype=np.intp)
        self.sub_upto = a

    def _finalize(self, p, q):
        """Unit ``p -> q`` is final."""
        if not self.everywhere:
            return
        x, y = self.x, self.y
        self.unit_box[:, self.count - 1] = (min(x[p], x[q]), min(y[p], y[q]),
                                            max(x[p], x[q]), max(y[p], y[q]))
        self.retained[self.count] = q
        self.count += 1
        if self.inversion:
            points = shapely.multipoints(np.column_stack([x[p:q + 1], y[p:q + 1]]))
            self.prev_hull = shapely.convex_hull(points)
            shapely.prepare(self.prev_hull)

    # ------------------------------------------------------------ geometry

    def _grow_box(self, a, k):
        x, y = self.x, self.y
        lo = a if self.box is None else self.box_upto + 1
        if lo <= k:
            box = [x[lo:k + 1].min(), y[lo:k + 1].min(),
                   x[lo:k + 1].max(), y[lo:k + 1].max()]
            if self.box is not None:
                box = [min(self.box[0], box[0]), min(self.box[1], box[1]),
                       max(self.box[2], box[2]), max(self.box[3], box[3])]
            self.box, self.box_upto = box, k

    def _hull(self, k):
        """Convex hull of the sub-path ``a..k``, grown from the last one."""
        if self.hull_at == k:
            return self.hull
        x, y = self.x, self.y
        lo = self.hull_upto if self.hull_xy is None else self.hull_upto + 1
        new = np.column_stack([x[lo:k + 1], y[lo:k + 1]])
        points = new if self.hull_xy is None else np.vstack([self.hull_xy, new])
        self.hull = shapely.convex_hull(shapely.multipoints(points))
        self.hull_xy = shapely.get_coordinates(self.hull)
        self.hull_upto = self.hull_at = k
        return self.hull

    def _inside_hull(self, k, px, py):
        """Whether a point lies strictly inside the hull of ``a..k``."""
        xmin, ymin, xmax, ymax = self.box
        if not (xmin <= px <= xmax and ymin <= py <= ymax):
            return False
        return bool(shapely.contains_xy(self._hull(k), px, py))

    def _units_near(self, xmin, ymin, xmax, ymax):
        """Final units whose chord's box meets the given box."""
        b = self.unit_box[:, :self.count - 1]
        return np.flatnonzero((b[0] <= xmax) & (b[2] >= xmin)
                              & (b[1] <= ymax) & (b[3] >= ymin))

    def _beyond_at(self, si, sj, v):
        """Whether any pair of segments ``si[m]``, ``sj[m]`` meets beyond ``v``.

        Two consecutive segments share ``v`` and meet beyond it only when
        folded back; any other pair is asked of GEOS.
        """
        x, y = self.x, self.y
        adjacent = np.abs(si - sj) == 1
        if adjacent.any():
            lo = np.minimum(si[adjacent], sj[adjacent])
            if _folded(x[lo], y[lo], x[lo + 1], y[lo + 1], x[lo + 2], y[lo + 2]).any():
                return True
        rest = ~adjacent
        return bool(rest.any() and self._meets_beyond(
            self.segments[si[rest]], self.segments[sj[rest]], v).any())

    def _meets_beyond(self, g1, g2, v):
        """Whether ``g1`` and ``g2`` meet anywhere but at the vertex ``v``."""
        rest = shapely.difference(
            shapely.intersection(g1, g2), shapely.points(self.x[v], self.y[v])
        )
        return ~shapely.is_empty(rest)

    # ------------------------------------------------------------ the test

    def _check(self, a, k):
        """The first relation the chord ``a -> k`` would break, or ``None``."""
        self._grow_box(a, k)
        if self.use_barriers:
            failed = self._barrier_relations(a, k)
            if failed:
                return failed
        if self.use_self and self._self_intersection(a, k):
            return "self_intersection"
        if self.inversion and self._inverts(a, k):
            return "inversion"
        return None

    def _barrier_relations(self, a, k):
        """D4 (DE-9IM topology) and D5 (direction) against every neighbour."""
        barriers = self.barriers
        near = barriers._tree.query(shapely.box(*self.box))
        if near.size == 0:
            return None
        hull = self._hull(k)
        near = near[shapely.intersects(barriers._parts[near], hull)]
        if near.size == 0:
            return None
        x, y = self.x, self.y
        sub_x, sub_y = x[a:k + 1], y[a:k + 1]
        chord_x, chord_y = x[[a, k]], y[[a, k]]
        xmin, ymin, xmax, ymax = shapely.bounds(hull)
        pad = max(1.0, 0.01 * max(xmax - xmin, ymax - ymin))
        # Clipped clear of the hull, so every relation inside it is unchanged.
        local = shapely.clip_by_rect(
            barriers._parts[near], xmin - pad, ymin - pad, xmax + pad, ymax + pad
        )
        if self.topology:
            sub = _path(sub_x, sub_y)
            chord = _path(chord_x, chord_y)
            if (shapely.relate(sub, local) != shapely.relate(chord, local)).any():
                return "topology"
        if self.direction:
            for box in shapely.bounds(shapely.intersection(local, hull)):
                if np.isnan(box).any():
                    continue
                if _direction_cells(sub_x, sub_y, box) != _direction_cells(
                    chord_x, chord_y, box
                ):
                    return "direction"
        return None

    def _self_intersection(self, a, k):
        """D6: whether the chord meets other units of ``Tc`` than its sub-path."""
        x, y = self.x, self.y
        if self.sub_upto < k:  # the sub-path now ends at k
            lo, hi = self.pair_start[self.sub_upto], self.pair_start[k]
            new_i = np.repeat(np.arange(self.sub_upto, k),
                              np.diff(self.pair_start[self.sub_upto:k + 1]))
            self.sub_i = np.concatenate([self.sub_i, new_i])
            self.sub_j = np.concatenate([self.sub_j, self.pair_j[lo:hi]])
            self.sub_upto = k
        # Meetings within the sub-path do not count, and stay within it as
        # it grows, so they are dropped for good.
        outside = (self.sub_j < a) | (self.sub_j >= k)
        self.sub_i, self.sub_j = self.sub_i[outside], self.sub_j[outside]
        si, sj = self.sub_i, self.sub_j
        retained = self.retained[:self.count]
        last = self.count - 2  # the final unit ending at a, if any

        # Units the sub-path meets: suffix segment j is unit j, prefix unit u
        # is -(u + 1).  At a shared vertex, only meetings elsewhere count.
        sub_units = [np.empty(0, int)]
        suffix = sj >= k
        js = sj[suffix]
        sub_units.append(js[js > k])
        near_i = si[suffix][js == k]
        if near_i.size and self._beyond_at(near_i, np.full(near_i.size, k), k):
            sub_units.append(np.array([k]))
        prefix = ~suffix
        if prefix.any():
            units = np.searchsorted(retained, sj[prefix], side="right") - 1
            sub_units.append(-(units[units != last] + 1))
            touching = units == last
            if touching.any() and self._beyond_at(
                si[prefix][touching], sj[prefix][touching], a
            ):
                sub_units.append(np.array([-(last + 1)]))
        sub_units = np.concatenate(sub_units)

        chord = _segments(x[a], y[a], x[k], y[k])[0]
        chord_units = [np.empty(0, int)]
        cj = self.segment_tree.query(chord)  # by bounding box
        cj = cj[cj >= k]
        cj = cj[_segments_meet(x[a], y[a], x[k], y[k],
                               x[cj], y[cj], x[cj + 1], y[cj + 1])]
        chord_units.append(cj[cj > k])
        if (cj == k).any() and _folded(x[a], y[a], x[k], y[k], x[k + 1], y[k + 1]):
            chord_units.append(np.array([k]))
        if last >= 0:
            near = self._units_near(min(x[a], x[k]), min(y[a], y[k]),
                                    max(x[a], x[k]), max(y[a], y[k]))
            p, q = retained[near], retained[near + 1]
            meet = _segments_meet(x[a], y[a], x[k], y[k], x[p], y[p], x[q], y[q])
            at_last = near == last
            if (meet & at_last).any():
                p_last = retained[last]
                meet[at_last] = _folded(x[k], y[k], x[a], y[a], x[p_last], y[p_last])
            chord_units.append(-(near[meet] + 1))
        chord_units = np.concatenate(chord_units)

        # Compare the two sets of units: suffix segments j and prefix units
        # -(u + 1).  Both are usually empty.
        if sub_units.size == 0 and chord_units.size == 0:
            return False
        if self.avoid_self and np.setdiff1d(chord_units, sub_units).size:
            return True
        if self.keep_self and np.setdiff1d(sub_units, chord_units).size:
            return True
        return False

    def _inverts(self, a, k):
        """D7: Tienaah's heuristics 3.7/3.8 against both contiguous units."""
        x, y = self.x, self.y
        if self.count >= 2:
            p = self.retained[self.count - 2]
            if shapely.contains_xy(self.prev_hull, x[k], y[k]):
                return True
            if self._inside_hull(k, x[p], y[p]):
                return True
        if k + 1 < self.n:
            if self._inside_hull(k, x[k + 1], y[k + 1]):
                return True
            following = _segments(x[k], y[k], x[k + 1], y[k + 1])[0]
            if shapely.contains_xy(following, x[a], y[a]):
                return True
        return False


def cpls_select(
    x,
    y,
    epsilon,
    barriers,
    *,
    topology=True,
    direction=True,
    avoid_self_intersection=True,
    keep_self_intersection=True,
    inversion=True,
):
    """Run CPLS on arrays and report which points are kept, and why.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    epsilon : float
        The Douglas--Peucker threshold, in the positions' units.
    barriers : Barriers or anything :class:`Barriers` accepts
        Shorelines or land, in the same projected CRS as the positions.
    topology : bool
        D4: a chord must have the same DE-9IM relation to every neighbouring
        barrier as the sub-path it replaces.
    direction : bool
        D5: a chord must pass through the same nine cells around every
        neighbouring barrier as the sub-path it replaces.
    avoid_self_intersection : bool
        D6: a chord may not meet a unit of the trajectory that its sub-path
        does not meet -- no new self-intersection.
    keep_self_intersection : bool
        D6: a chord must meet every unit its sub-path meets -- the
        trajectory's own crossings are preserved.
    inversion : bool
        D7: Tienaah's linear-inversion test against the units either side.

    Each relation is one of Tienaah's constraints and can be switched off;
    all are on by default.  See the module docstring.

    Returns
    -------
    CplsSelection
        ``selection.mask`` is the CPLS result.  With every relation off it is
        exactly Douglas--Peucker.
    """
    rules = dict(
        topology=topology, direction=direction,
        avoid_self_intersection=avoid_self_intersection,
        keep_self_intersection=keep_self_intersection, inversion=inversion,
    )
    barriers = _as_barriers(barriers)
    dp = douglas_peucker(x, y, epsilon)  # validates the input
    keep = dp.copy()
    n = dp.size
    retained = {name: np.zeros(n, dtype=bool) for name in RULES}
    if n > 2:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        _Stage2(x, y, barriers, rules).run(dp, keep, retained)
    return CplsSelection(dp=dp, topology=keep & ~dp, rules=retained)


# ------------------------------------------------------------ dataframes

_RULE_NAMES = {
    "topology", "direction", "avoid_self_intersection",
    "keep_self_intersection", "inversion",
}


def _select_frame(df, epsilon, barriers, rules):
    _check_crs(df, barriers)
    geom = df.geometry
    return cpls_select(
        geom.x.to_numpy(), geom.y.to_numpy(), epsilon, barriers, **rules
    )


def cpls_compress(
    traj, epsilon, barriers, by=None, *, filter_window=30, time_column=None,
    **rules,
):
    """Compress trajectories with CPLS.

    The result is a row subset of the input, so every column of every retained
    point survives.  CPLS itself does not use time: row order is trajectory
    order.  Time matters only to the filtering that runs first.

    Parameters
    ----------
    traj : movingpandas.Trajectory, TrajectoryCollection or GeoDataFrame
        Positions in a projected CRS, in trajectory order.
    epsilon : float
        Douglas--Peucker threshold, in the CRS's units (metres).
    barriers : Barriers or anything :class:`Barriers` accepts
        Shorelines or land.  Built once and reused for every trajectory.  When
        both the trajectory and the barriers have a CRS they must match; if
        either is unknown they are assumed to.
    by : str, optional
        For a GeoDataFrame holding several trajectories: the column naming
        each one.  Each is compressed on its own.
    filter_window : float or None
        Width, in seconds (the unit of the times), of the time-windowed median
        that denoises the positions before CPLS runs: ``30`` by default,
        ``None`` for no filtering.  See "Filtering first" above.
    time_column : str, optional
        Numeric time column (UNIX seconds) for the filter; the index
        otherwise, which must then be a DatetimeIndex (as MovingPandas keeps
        it).  Unused when ``filter_window`` is ``None``.
    **rules
        ``topology``, ``direction``, ``avoid_self_intersection``,
        ``keep_self_intersection``, ``inversion``: switch a relation of the
        signature off, as in :func:`cpls_select`.

    Returns
    -------
    The same kind of object, holding only the retained points; with filtering
    on, their geometry is the filtered one.
    """
    unknown = set(rules) - _RULE_NAMES
    if unknown:
        raise TypeError(f"unknown relation(s): {sorted(unknown)}")
    barriers = _as_barriers(barriers)

    def one(df):
        if filter_window is not None:
            df = denoise(df, filter_window, time_column=time_column)
        return df.iloc[_select_frame(df, epsilon, barriers, rules).mask]

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        import movingpandas as mpd

        return mpd.TrajectoryCollection([
            cpls_compress(tr, epsilon, barriers, filter_window=filter_window,
                          time_column=time_column, **rules)
            for tr in traj.trajectories
        ])

    if hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        import movingpandas as mpd

        return mpd.Trajectory(one(traj.df).copy(), traj.id)

    if by is not None:
        parts = [one(group) for _, group in traj.groupby(by, sort=False)]
        return pd.concat(parts) if parts else traj.iloc[:0]

    return one(traj).copy()
