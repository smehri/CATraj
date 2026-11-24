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

:mod:`.cpls` implements Context-aware Piecewise Linear Segmentation of
[Mehri2021]: Douglas--Peucker first, then a pass inside each of its segments
that keeps any point whose removal would change the topological signature of
Tienaah et al. -- five relations, each switchable and all on by default: the
DE-9IM relation with each barrier (a shoreline) and the nine-cell direction
around it, and, for the track itself, no new or lost self-intersection and no
inversion at a shared vertex.  The barriers are shapely geometries, indexed
once by :class:`.Barriers`.  The track's own relations keep many points on raw
AIS with moored vessels; the paper's two-stage variant is meant to remove that.

:mod:`.two_stage` implements 2stage-CPLS (Algorithm 2 of the same paper): stay
points -- by AIS status or by Li et al.'s stay-point detection -- are replaced
by their centroid, Douglas--Peucker on the speed series marks where the vessel
changes speed, and CPLS then runs on the collapsed track with those points
forced.  The topological signature is that of the collapsed trajectory, so the
jitter of a moored vessel no longer keeps its points.

Every dataframe-level step -- :func:`compress`, :func:`cpls_compress` and
:func:`two_stage_cpls_compress` -- first median-filters its inputs over a
30-second window (``filter_window=30``; ``None`` switches it off), with
:func:`catraj.preprocessing.denoise`: the positions, and for CADP every numeric
threshold column; the result rows carry the filtered values.
:func:`two_stage_cpls_select` also filters its speed series (``speed_window``).
The array-level algorithms (``douglas_peucker``, ``cadp_select``,
``cpls_select``) filter nothing.  Time comes from ``time_column`` or a
DatetimeIndex.

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
from .cpls import (
    Barriers,
    CplsSelection,
    cpls_compress,
    cpls_select,
    segment_crossings,
)
from .douglas_peucker import douglas_peucker, segment_distances
from .evaluation import compression_ratio, dtw
from .two_stage import (
    TwoStageSelection,
    stay_points,
    two_stage_cpls_compress,
    two_stage_cpls_select,
)

__all__ = [
    "compress", "cadp_select", "Context", "Selection",
    "context_errors", "nominal_boundaries",
    "Barriers", "CplsSelection", "segment_crossings", "cpls_select", "cpls_compress",
    "TwoStageSelection", "stay_points", "two_stage_cpls_select",
    "two_stage_cpls_compress",
    "douglas_peucker", "segment_distances",
    "compression_ratio", "dtw",
]
