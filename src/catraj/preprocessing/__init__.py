"""Preprocessing: cleaning raw positions before they are annotated or compressed.

:mod:`.filtering` implements the moving-average step of [Mehri2021] section
III-B, as a centred, time-windowed median or mean over positions and chosen
numeric columns, to remove GPS noise from AIS data.

The module's docstring carries the method and the decisions taken where the
paper leaves room.

Filtering is a default step of the package (decided with the author,
2026-11-24): every processing function -- :func:`catraj.compression.compress`,
:func:`~catraj.compression.cpls_compress`,
:func:`~catraj.compression.two_stage_cpls_compress`,
:func:`~catraj.compression.two_stage_cpls_select` (its speed series) and
:func:`catraj.annotation.annotate_points` (the station series) -- median-filters
its inputs over a 30-second window unless ``filter_window=None``.  Call
:func:`denoise` yourself to filter with another window, the mean, or once for
several steps.
"""

from .filtering import denoise, rolling_filter
from .mmsi import ITU_MIDS, mmsi_class, mmsi_classes

__all__ = ["denoise", "rolling_filter", "ITU_MIDS", "mmsi_class", "mmsi_classes"]
