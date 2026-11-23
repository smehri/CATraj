"""Judging a compression -- [Mehri2025] section 2.4.

Two measures, read together: how much was removed, and how much the shape
changed.  Each alone is meaningless -- dropping every point but the ends
compresses superbly and preserves nothing.

``compression_ratio``  Eq. 9, ``R = (1 - N / M) * 100``
``dtw``                Eq. 10, dynamic time warping with Euclidean distance

DTW aligns the two point sequences non-linearly, so a compressed trajectory --
far fewer points, unevenly spaced -- can be compared with the original at all.
Lower is more similar.  It is a cumulative distance in the coordinates' units,
so it grows with trajectory length and is comparable only between
compressions of the same trajectories.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compression_ratio", "dtw"]


def compression_ratio(n_compressed, n_original):
    """Eq. 9: the percentage of points removed."""
    if n_original <= 0:
        raise ValueError("the original must have at least one point")
    if not 0 <= n_compressed <= n_original:
        raise ValueError("a compression cannot add points")
    return (1.0 - n_compressed / n_original) * 100.0


def _xy(points):
    if hasattr(points, "geometry"):
        return np.column_stack([points.geometry.x, points.geometry.y])
    if hasattr(points, "df"):  # movingpandas.Trajectory
        return _xy(points.df)
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array, a GeoDataFrame or a Trajectory")
    return arr


def dtw(original, compressed):
    """Eq. 10: dynamic time warping distance, Euclidean ground distance.

    The minimum, over all monotone alignments of the two sequences, of the
    summed distances between aligned points.

    Parameters
    ----------
    original, compressed : (n, 2) array, GeoDataFrame or movingpandas.Trajectory
        Point sequences in trajectory order, in projected units.

    Returns
    -------
    float
        Zero for identical sequences; larger is less similar.

    Notes
    -----
    Memory is linear in the shorter sequence: the cost matrix is never held
    whole, so an original of tens of thousands of points is no problem.  Each
    row is solved in one vectorised pass -- the within-row dependency
    ``D[i, j-1]`` is a running minimum over prefix sums.
    """
    a, b = _xy(original), _xy(compressed)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both sequences need at least one point")
    if len(b) > len(a):
        a, b = b, a  # DTW is symmetric; iterate over the longer one

    m = len(b)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for point in a:
        cost = np.hypot(point[0] - b[:, 0], point[1] - b[:, 1])
        # D[i, j] = cost[j] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
        diagonal_or_up = cost + np.minimum(previous[:-1], previous[1:])
        prefix = np.cumsum(cost)
        row = prefix + np.minimum.accumulate(diagonal_or_up - prefix)
        previous = np.concatenate(([np.inf], row))
    return float(previous[m])
