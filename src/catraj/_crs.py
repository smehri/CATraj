"""Coordinate-system checks shared by compression and preprocessing."""

from __future__ import annotations

__all__ = ["_check_geographic", "_check_crs"]


def _check_geographic(df):
    """Refuse a geographic CRS: thresholds and windows are in metres."""
    crs = getattr(df, "crs", None)
    if crs is not None and crs.is_geographic:
        raise ValueError(
            "the trajectory is in a geographic CRS; epsilon is a distance, so "
            "reproject to a projected CRS in metres first (e.g. .to_crs(...))"
        )


def _check_crs(df, barriers):
    """Refuse a geographic CRS, and a CRS the barriers do not share."""
    _check_geographic(df)
    crs = getattr(df, "crs", None)
    if crs is not None and barriers.crs is not None:
        from pyproj import CRS

        if CRS.from_user_input(barriers.crs) != CRS.from_user_input(crs):
            raise ValueError(
                "the barriers and the trajectory are in different CRS; "
                "reproject the barriers with .to_crs(...) to the trajectory's"
            )
