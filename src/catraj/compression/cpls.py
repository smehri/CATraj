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

The paper compares topological signatures by "checking the intersection of the
trajectories segments and the coastlines".  The decisions below say how that
sentence is read.  Each is decided here; none has been confirmed with the
author yet.

Decisions where the paper leaves room
-------------------------------------

* **D1. Stage 1 is** :func:`.douglas_peucker`, called, not reimplemented.
  Line 7, ``imax = 1``, is read as ``imax = i`` and line 4's range as the
  interior points -- the same typo and the same reading as in the CADP paper,
  documented in :mod:`.douglas_peucker`.
* **D2. Stage 2 is the literal greedy of lines 14-20**: front to back, one
  point at a time, testing the segment from the last retained point to the
  point *after* the candidate.  ``Tc = [T1 ... Tj-1, Tj+1 ... Tn]`` still
  contains ``Tj+1`` when the test runs.
* **D3. "Consistent" is an intersection test** between the new segment and the
  barriers, with GEOS ``intersects``.  A segment that only touches a barrier
  counts as intersecting: the conservative reading.
* **D4. Existing crossings are tolerated.**  A new segment may intersect a
  barrier only if the original sub-path it replaces already does.  The paper
  asks for the *same* signature in the original and the simplified trajectory,
  not for a crossing-free result, and real positions do fall on the mapped
  land.  A strict rule would freeze every such stretch.
* **D5. Sidedness is not checked.**  If the original passes north of a small
  island and the new segment passes south of it without touching it, CPLS
  accepts the change.  The paper tests intersection only.  This is a
  limitation.
* **D6. The epsilon bound holds against the Douglas--Peucker segment.**  Once
  stage 2 retains extra points, a dropped point is no longer guaranteed to lie
  within ``epsilon`` of the segment between its final neighbours -- the same
  consequence as in :mod:`.cadp`'s ``post`` mode.  It is not "fixed".
* **D7. The greedy is not minimal.**  It can retain a point even when the
  Douglas--Peucker segment itself would be consistent, because an intermediate
  chord was not.  That is the literal reading of lines 14-20.
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
* The first and last points are retained.

Implementation
--------------

The barriers are indexed once (:class:`Barriers`).  Every chord between two
points of a Douglas--Peucker segment lies inside the bounding box of that
segment's points; if that box meets no barrier, stage 2 would drop everything
there, so only the segments whose box does meet one are searched.  Inside
those, the greedy tests a block of chords from the last retained point in one
vectorised call -- the first failing chord is the same one the point-by-point
loop would find -- and doubles the block while nothing fails.

Coordinates must be projected -- ``epsilon`` is a distance in their units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .douglas_peucker import douglas_peucker

__all__ = [
    "Barriers",
    "CplsSelection",
    "segment_crossings",
    "cpls_select",
]

_FIRST_BLOCK = 64


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
        x0, y0, x1, y1 = (np.asarray(v, dtype=float).ravel() for v in (x0, y0, x1, y1))
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

    def boxes_meet(self, xmin, ymin, xmax, ymax):
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


def _greedy(x, y, barriers, cs, start, end, keep):
    """Stage 2 inside one Douglas--Peucker segment ``start``--``end``.

    From the last retained point ``a``, the chord ``a -> k`` replaces the path
    through points ``a+1 .. k-1``; the first ``k`` whose chord is not
    consistent means point ``k - 1`` must be retained.
    """
    a = start
    lo, size = a + 2, _FIRST_BLOCK
    while lo <= end:
        hi = min(lo + size - 1, end)
        ks = np.arange(lo, hi + 1)
        bad = np.zeros(ks.size, dtype=bool)
        open_ = (cs[ks] - cs[a]) == 0  # chords over an untouched original
        if open_.any():
            probe = ks[open_]
            bad[open_] = barriers.hits(
                np.full(probe.size, x[a]), np.full(probe.size, y[a]),
                x[probe], y[probe],
            )
        if bad.any():
            a = int(ks[np.argmax(bad)]) - 1
            keep[a] = True
            lo, size = a + 2, _FIRST_BLOCK
        else:
            lo, size = hi + 1, size * 2


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
    flagged = barriers.boxes_meet(xmin, ymin, xmax, ymax) & (ends - starts >= 2)

    if flagged.any():
        cs = _prefix_counts(_original_hits(x, y, barriers))
        for s, e in zip(starts[flagged], ends[flagged]):
            _greedy(x, y, barriers, cs, int(s), int(e), keep)
    return CplsSelection(dp=dp, topology=keep & ~dp)
