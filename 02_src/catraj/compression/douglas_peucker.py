"""Douglas--Peucker -- the spatial stage of CADP.

[Mehri2025] section 2.1, Algorithm 1.  The first and last points are kept; for
the points between them, the error of removing each one is its distance to the
segment joining the two (Eq. 1).  If the largest error exceeds ``epsilon`` that
point is kept and the two halves are processed in turn; otherwise every point
between is dropped.

Two departures from the pseudo-code, neither changing the result:

* **No recursion.**  Algorithm 1 recurses on each half.  A pending-segment
  stack does the same work in the same order of decisions, without Python's
  recursion limit -- which a 100,000-point trajectory (the paper's largest
  benchmark) can exceed when the simplification is deep and one-sided.
* **Typo in line 6.**  ``imax = 1`` is read as ``imax = i``: the index of the
  point with the largest error, which is what lines 9-10 split on.

The distance is to the *segment*, not to the infinite line through its ends:
Eq. 1 projects the point "on the line segment connecting t_start, t_end".  A
point whose projection falls beyond an end is measured to that end.  This also
matches GEOS (and so Shapely and MovingPandas' ``DouglasPeuckerGeneralizer``).

Coordinates must be projected -- ``epsilon`` is a distance in their units.
"""

from __future__ import annotations

import numpy as np

__all__ = ["douglas_peucker", "segment_distances"]


def segment_distances(x, y, start, end):
    """Distance from each point ``(x, y)`` to the segment ``start``--``end``.

    Eq. 1: the Euclidean distance from the point to its projection on the
    segment.  A zero-length segment -- a loop that returns to where it began,
    or an object at rest -- measures distance to that single point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax, ay = map(float, start)
    bx, by = map(float, end)
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return np.hypot(x - ax, y - ay)
    u = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0)
    return np.hypot(x - (ax + u * dx), y - (ay + u * dy))


def douglas_peucker(x, y, epsilon):
    """Points the Douglas--Peucker algorithm keeps.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    epsilon : float
        Largest allowed distance between a dropped point and the simplified
        line, in the same units.  A point is kept when its error is strictly
        greater, as in Algorithm 1 line 8.

    Returns
    -------
    ndarray of bool
        ``True`` for every point kept.  The first and last always are.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have equal length; got {x.size}, {y.size}")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite, non-negative distance")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("positions must be finite; drop missing points first")

    n = x.size
    keep = np.zeros(n, dtype=bool)
    if n <= 2:
        keep[:] = True
        return keep
    keep[0] = keep[-1] = True

    pending = [(0, n - 1)]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        inner = slice(first + 1, last)
        errors = segment_distances(
            x[inner], y[inner], (x[first], y[first]), (x[last], y[last])
        )
        worst = int(np.argmax(errors))
        if errors[worst] > epsilon:
            split = first + 1 + worst
            keep[split] = True
            pending.append((split, last))
            pending.append((first, split))
    return keep
