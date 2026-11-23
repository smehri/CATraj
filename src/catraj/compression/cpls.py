"""Context-aware Piecewise Linear Segmentation (CPLS) -- [Mehri2021] section IV-A.

Douglas--Peucker keeps the points that carry spatial information and is blind
to everything else.  Near a coast that blindness has a cost: a chord that
replaces a bend in the track can cut across a headland the vessel sailed
round.  CPLS refuses to drop a point when dropping it would make the simplified
trajectory cross a barrier -- a shoreline -- that the original did not cross.

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

Every decision below was checked against the text of section IV-A and
Algorithm 1, and follows it; the sentence that settles each point is quoted.
D1-D7 were confirmed with the author (2026-11-23).  D8 and D9 are
implementation details the paper does not address.

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
  15), so the segment tested runs from the last retained point to ``Tj+1``.
* **D3. "Consistent" is an intersection test.**  "To compare the topological
  signature of the original and simplified trajectories, the proposed method
  checks the intersection of the trajectories segments and the coastlines."
  It is GEOS ``intersects``, so a segment that only touches a barrier
  intersects it.
* **D4. Only new crossings are refused.**  "The topological signature of the
  original and simplified trajectory should be the same", and "if there are
  some intersections generated through the compression process, the generated
  segment is void."  Line 1 builds the original's topological relations to
  compare against.  A new segment may therefore intersect a barrier only if
  the original sub-path it replaces already does.
* **D5. The signature includes the side: Tienaah's direction relation.**  The
  paper takes the topological signature from Tienaah et al. [63] ("should be
  the same [63]"; "with the idea of Tienaah [63]"), whose model has two parts:
  the topological relation above, and a *direction relation* -- "the
  simplified geometry must conform to the same quadrant relation as the
  original to avoid side (left, right, up, down, diagonal, and interior)
  conflicts" (Tienaah 2018, section 2.3.2, the chapter published as [63]).
  So a chord is consistent only if, for every barrier that is a *neighbour*
  of the sub-path it replaces, it passes through the same cells of the
  nine-cell model as that sub-path:

  - A neighbour is a barrier part that intersects the convex hull of the
    sub-path's vertices: consistency "may be violated if these geometries are
    found within the convex hull of the set of vertices forming the
    trajectory" (the paper, after [63] and [65]).  Each part of a multi-part
    barrier is a neighbour of its own.
  - The nine cells are the neighbour's minimum bounding box *within the
    hull* (``part & hull``), and the eight regions around it: N, NE, E, SE,
    S, SW, W, NW.
  - A line passes through a cell when it runs through the cell's interior for
    a positive length.  Running exactly along a cell boundary, or a stretch
    shorter than 1e-9 of a segment (rounding), does not count.

  A vessel that passed north of an island and a chord that passes south of it
  register different cells, so the point is kept even though the chord
  touches nothing.  The rule is strict: a chord that cuts a bend can miss a
  cell the bend went through without changing sides, and that keeps a point
  too.  That is the model as published.
* **D6. The offset distance is tested once, by Douglas--Peucker.**  Lines
  13-20 test only the topological signature.  A point added back is not
  followed by a new offset test, so a dropped point is within ``epsilon`` of
  its Douglas--Peucker segment, but not necessarily of the segment between its
  final neighbours -- as in :mod:`.cadp`'s ``post`` mode.
* **D7. The pass is the greedy of lines 14-20, as written.**  One point at a
  time, kept or dropped for good.  It does not search for the fewest points:
  it can keep a point where another choice would have needed fewer.
* **D8. Barriers and positions share one projected CRS.**  A mismatch raises;
  nothing is reprojected silently.
* **D9. Repeated positions** give zero-length segments, which are tested as
  they are.  GEOS treats one as a point: inside a polygon or on its boundary
  is a hit, outside is not.

What the result guarantees
--------------------------

* The result contains every Douglas--Peucker point; with no barriers, or none
  near the trajectory, it equals Douglas--Peucker.
* For every pair of consecutive retained points, the segment between them
  either misses the barriers or replaces an original sub-path that already
  touched them: compression never introduces a crossing
  (:func:`segment_crossings` with ``new_only=True`` measures it).
* If the original never intersects a barrier, neither does the result.
* For every pair of consecutive retained points, each barrier neighbouring the
  original sub-path between them is passed through the same nine cells by the
  segment as by the sub-path: no barrier changes side.
* The first and last points are retained.

Implementation
--------------

The barriers are indexed once (:class:`Barriers`).  Every chord between two
points of a Douglas--Peucker segment lies inside the bounding box of that
segment's points; if that box meets no barrier, stage 2 would drop everything
there, so only the segments whose box does meet one are searched.  Inside
those the greedy goes point by point.  A chord whose hull meets no barrier is
accepted at once; otherwise the neighbours are clipped to the hull and the
nine-cell signatures are computed for all segments of the sub-path at once,
from where each segment enters and leaves each band of the grid.

Coordinates must be projected -- ``epsilon`` is a distance in their units.

References
----------
[Mehri2021] Mehri, S., Alesheikh, A. A., Basiri, A. (2021).  A Contextual
    Hybrid Model for Vessel Movement Prediction.  IEEE Access 9, 45600-45613.
[63] Tienaah, T., Stefanakis, E., Coleman, D. (2015).  Contextual
    Douglas-Peucker simplification.  Geomatica 69(3), 327-338.  Also chapter 2
    of Tienaah, T. (2018), Line Simplification Under Spatial Constraints, UNB
    Technical Report 314.
[65] Saalfeld, A. (1999).  Topologically consistent line simplification with
    the Douglas-Peucker algorithm.  Cartography and GIS 26(1), 7-18.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely

from .douglas_peucker import douglas_peucker

__all__ = [
    "Barriers",
    "CplsSelection",
    "segment_crossings",
    "cpls_select",
    "cpls_compress",
]

# A line "passes through" a cell of the direction model only for longer than
# this fraction of a segment; shorter overlaps are rounding at a cell edge.
_ROUNDING = 1e-9
_CELL_BITS = 1 << np.arange(9)


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
        coords = np.empty((x0.size, 2, 2))
        coords[:, 0, 0], coords[:, 0, 1] = x0, y0
        coords[:, 1, 0], coords[:, 1, 1] = x1, y1
        lines = shapely.linestrings(coords)
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


@dataclass
class CplsSelection:
    """Which points CPLS keeps, and why.

    ``dp`` is the Douglas--Peucker stage's choice; ``topology`` the points
    stage 2 retained on top of it, so it never overlaps ``dp``.
    """

    dp: np.ndarray
    topology: np.ndarray

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
        return "\n".join([
            f"{n} points -> {kept} kept "
            f"({100.0 * (1 - kept / n) if n else 0.0:.2f}% compression)",
            f"  Douglas-Peucker      {int(self.dp.sum())}",
            f"  + topology           {int(self.topology.sum())}",
        ])


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


def _consistent(x, y, barriers, cs, a, k, hull, near):
    """Whether the chord ``a -> k`` keeps the signature of the sub-path ``a..k``.

    ``hull`` is the convex hull of the sub-path's vertices and ``near`` the
    barrier parts that intersect it, the neighbours; there is at least one.
    D3/D4: the chord creates no intersection the sub-path did not have.  D5:
    every neighbour is passed through the same nine cells by the chord as by
    the sub-path.
    """
    parts = barriers._parts[near]
    chord = shapely.linestrings([[x[a], y[a]], [x[k], y[k]]])
    if cs[k] == cs[a] and shapely.intersects(parts, chord).any():
        return False
    xmin, ymin, xmax, ymax = shapely.bounds(hull)
    if xmax > xmin and ymax > ymin:  # clipping first spares GEOS large polygons
        parts = shapely.clip_by_rect(parts, xmin, ymin, xmax, ymax)
    neighbours = shapely.intersection(parts, hull)
    sub_x, sub_y = x[a:k + 1], y[a:k + 1]
    chord_x, chord_y = x[[a, k]], y[[a, k]]
    for box in shapely.bounds(neighbours):
        if np.isnan(box).any():
            continue
        if _direction_cells(sub_x, sub_y, box) != _direction_cells(
            chord_x, chord_y, box
        ):
            return False
    return True


def _greedy(x, y, barriers, cs, start, end, keep):
    """Stage 2 inside one Douglas--Peucker segment ``start``--``end``.

    Lines 14-20, point by point: with ``a`` the last retained point, removing
    point ``k - 1`` replaces the sub-path ``a..k`` by the chord ``a -> k``; if
    that breaks the signature, point ``k - 1`` is retained.  A chord whose
    sub-path has no neighbour is consistent at once: it lies inside the hull,
    so it meets nothing.  The bounding box, and the hull once needed, grow with
    ``k`` instead of being rebuilt from the whole sub-path.
    """
    a, k = start, start + 2
    box = hull_xy = None
    while k <= end:
        if box is None:  # a new sub-path from a
            box = [x[a:k + 1].min(), y[a:k + 1].min(),
                   x[a:k + 1].max(), y[a:k + 1].max()]
            hull_xy, hull_upto = np.column_stack([x[a:a + 1], y[a:a + 1]]), a
        else:
            box = [min(box[0], x[k]), min(box[1], y[k]),
                   max(box[2], x[k]), max(box[3], y[k])]
        near = barriers._tree.query(shapely.box(*box))
        hull = None
        if near.size:
            new = np.column_stack([x[hull_upto + 1:k + 1], y[hull_upto + 1:k + 1]])
            hull = shapely.convex_hull(shapely.multipoints(np.vstack([hull_xy, new])))
            hull_xy, hull_upto = shapely.get_coordinates(hull), k
            near = near[shapely.intersects(barriers._parts[near], hull)]
        if near.size == 0 or _consistent(x, y, barriers, cs, a, k, hull, near):
            k += 1
        else:
            a = k - 1
            keep[a] = True
            k = a + 2
            box = None


def cpls_select(x, y, epsilon, barriers):
    """Run CPLS on arrays and report which points are kept, and why.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    epsilon : float
        The Douglas--Peucker threshold, in the positions' units.
    barriers : Barriers or anything :class:`Barriers` accepts
        Shorelines or land, in the same projected CRS as the positions.

    Returns
    -------
    CplsSelection
        ``selection.mask`` is the CPLS result.  With no barrier near the
        trajectory it is exactly Douglas--Peucker.
    """
    barriers = _as_barriers(barriers)
    dp = douglas_peucker(x, y, epsilon)  # validates the input
    keep = dp.copy()
    n = dp.size
    if n <= 2 or len(barriers) == 0:
        return CplsSelection(dp=dp, topology=np.zeros(n, dtype=bool))

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    kept = np.flatnonzero(dp)
    starts, ends = kept[:-1], kept[1:]

    # Chords of a segment stay inside the box of its points (the end point
    # is not in reduceat's range for all but the last segment).
    xmin = np.minimum(np.minimum.reduceat(x, starts), x[ends])
    xmax = np.maximum(np.maximum.reduceat(x, starts), x[ends])
    ymin = np.minimum(np.minimum.reduceat(y, starts), y[ends])
    ymax = np.maximum(np.maximum.reduceat(y, starts), y[ends])
    flagged = barriers._boxes_meet(xmin, ymin, xmax, ymax) & (ends - starts >= 2)

    if flagged.any():
        cs = _prefix_counts(_original_hits(x, y, barriers))
        for s, e in zip(starts[flagged], ends[flagged]):
            _greedy(x, y, barriers, cs, int(s), int(e), keep)
    return CplsSelection(dp=dp, topology=keep & ~dp)


# ------------------------------------------------------------ dataframes


def _select_frame(df, epsilon, barriers):
    crs = getattr(df, "crs", None)
    if crs is not None and crs.is_geographic:
        raise ValueError(
            "the trajectory is in a geographic CRS; epsilon is a distance, so "
            "reproject to a projected CRS in metres first (e.g. .to_crs(...))"
        )
    if crs is not None and barriers.crs is not None:
        from pyproj import CRS

        if CRS.from_user_input(barriers.crs) != CRS.from_user_input(crs):
            raise ValueError(
                "the barriers and the trajectory are in different CRS; "
                "reproject the barriers with .to_crs(...) to the trajectory's"
            )
    geom = df.geometry
    return cpls_select(geom.x.to_numpy(), geom.y.to_numpy(), epsilon, barriers)


def cpls_compress(traj, epsilon, barriers, by=None):
    """Compress trajectories with CPLS.

    The result is a row subset of the input, so every column of every retained
    point survives.  CPLS does not use time: row order is trajectory order.

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

    Returns
    -------
    The same kind of object, holding only the retained points.
    """
    barriers = _as_barriers(barriers)

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        import movingpandas as mpd

        return mpd.TrajectoryCollection([
            cpls_compress(tr, epsilon, barriers) for tr in traj.trajectories
        ])

    if hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        import movingpandas as mpd

        mask = _select_frame(traj.df, epsilon, barriers).mask
        return mpd.Trajectory(traj.df.iloc[mask].copy(), traj.id)

    if by is not None:
        parts = [
            group.iloc[_select_frame(group, epsilon, barriers).mask]
            for _, group in traj.groupby(by, sort=False)
        ]
        return pd.concat(parts) if parts else traj.iloc[:0]

    return traj.iloc[_select_frame(traj, epsilon, barriers).mask].copy()
