"""Trajectory compression that keeps context.

Classic compression keeps the points that carry spatial information and drops
the rest.  A point on a straight leg is spatially redundant, yet it may be the
one where the waves rose, the water shoaled or the vessel slowed -- and once
dropped, that is gone from every analysis downstream.

:mod:`.cadp` implements the Context-Aware Douglas--Peucker method of
[Mehri2025]: plain Douglas--Peucker first, then every point whose removal
would misrepresent a context variable by more than its threshold is added
back.  The context it tests is ordinary columns on the trajectory -- written
by :mod:`catraj.annotation`, or recorded with the positions, like speed.

:mod:`.evaluation` scores a compression by its ratio and by DTW similarity.

Each module's docstring carries the part of the method it implements, and the
decisions taken where the paper leaves room.
"""

from .cadp import (
    Context,
    Selection,
    cadp_select,
    compress,
    context_errors,
    nominal_boundaries,
)
from .douglas_peucker import douglas_peucker, segment_distances
from .evaluation import compression_ratio, dtw

__all__ = [
    "compress", "cadp_select", "Context", "Selection",
    "context_errors", "nominal_boundaries",
    "douglas_peucker", "segment_distances",
    "compression_ratio", "dtw",
]
