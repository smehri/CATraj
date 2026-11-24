"""Splitting a vessel's track into trips.

An AIS archive holds one long track per vessel: a ferry or a harbour tug
over a month is dozens of trips joined end to end, with the time spent
alongside in between.  Algorithms that look at the whole track at once (the
self-intersection and inversion relations of CPLS, for instance) then treat
the passage from one trip to the next as part of a single movement, and
their cost grows with the length of the whole track.  Splitting it first
gives each trip to them on its own -- and the trips of one vessel can then
be shared out between processes like separate vessels.

A new trip starts

* after a **gap**: two consecutive positions more than ``max_gap`` seconds
  apart (the vessel was out of range or its transponder off);
* where a **stay** begins or ends: the vessel stays within ``stay_radius`` of
  one spot for at least ``stay_duration`` seconds (the movement stays of
  :func:`catraj.compression.stay_points`, looked for between gaps only, so
  that two fixes either side of a gap do not make one), or, when ``status``
  is given, its AIS navigational status says at anchor or moored.

Every position belongs to exactly one piece, and each stay is a piece of its
own, flagged in the ``stay`` column, so nothing is dropped: a compression run
per piece keeps each piece's first and last point, so the moments a trip
starts and ends survive it.

The navigational status is not used by default: it is set by hand on board
and is often left at "moored" or "under way" long after it stopped being
true; the movement test needs nothing but the positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._time import times_of

__all__ = ["trip_ids", "split_trips"]


def trip_ids(x, y, t, *, max_gap=3600.0, stay_radius=300.0, stay_duration=1800.0,
             status=None, stay_status=(1, 5)):
    """The trip of every position of one vessel, numbered from 0, and its stays.

    Parameters
    ----------
    x, y : array-like
        Positions in projected units (metres), in time order.
    t : array-like
        Times in seconds, increasing.
    max_gap : float or None
        A gap longer than this, in seconds, starts a new trip; ``None`` for no
        gap rule.
    stay_radius, stay_duration : float or None
        Metres and seconds of the movement-stay test; both ``None`` for none.
    status : array-like, optional
        AIS navigational status per position; codes in ``stay_status`` are
        stays as well.

    Returns
    -------
    trip : ndarray of int
        Non-decreasing, starting at 0.
    stay : ndarray of bool
        Whether the position is in a stay; each stay is a trip number of its own.
    """
    from ..compression.two_stage import stay_points  # here: compression imports this package

    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    y = np.asarray(y, dtype=float).ravel()
    gaps = np.diff(t) > max_gap if max_gap is not None else np.zeros(x.size - 1, dtype=bool)
    stay = np.zeros(x.size, dtype=bool)
    if status is not None:
        stay |= stay_points(x, y, t, status=status, stay_status=stay_status)
    if stay_radius is not None:
        # Within each stretch between gaps: across a gap, two fixes hours apart but
        # close together would otherwise pass for a stay.
        bounds = np.concatenate([[0], np.flatnonzero(gaps) + 1, [x.size]])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            stay[lo:hi] |= stay_points(x[lo:hi], y[lo:hi], t[lo:hi], stay_radius=stay_radius,
                                       stay_duration=stay_duration)
    starts = gaps | (np.diff(stay.astype(np.int8)) != 0)
    return np.concatenate([[0], np.cumsum(starts)]).astype(np.int64), stay


def split_trips(frame, *, by=None, time_column=None, max_gap=3600.0, stay_radius=300.0,
                stay_duration=1800.0, status_column=None, stay_status=(1, 5),
                trip_column="trip", stay_column="stay", n_jobs=1):
    """``frame`` with a trip number (and a stay flag) on every position.

    Parameters
    ----------
    frame : GeoDataFrame
        Positions in a projected CRS, in time order within each vessel.
    by : str, optional
        The column naming each vessel; trips are numbered from 0 within each.
        ``None`` for one vessel.
    time_column : str, optional
        Numeric time column (UNIX seconds); the index otherwise, as in
        :func:`catraj.compression.compress`.
    max_gap, stay_radius, stay_duration, stay_status
        As in :func:`trip_ids`.
    status_column : str, optional
        AIS navigational status, to count at-anchor and moored as stays too.
    trip_column, stay_column : str or None
        Names of the new columns; ``None`` for no stay column.
    n_jobs : int or None
        Processes over which the vessels are shared out (:func:`catraj.parallel_map`).

    Returns
    -------
    A copy of ``frame`` with the new columns.  Group by ``[by, trip_column]``
    to work on the trips.
    """
    from .._parallel import map_groups

    if not frame.index.is_unique:
        raise ValueError("split_trips needs a unique index; reset_index() first")

    def one(df):
        t = df[time_column].to_numpy(dtype=float) if time_column else times_of(df)
        status = df[status_column].to_numpy() if status_column else None
        trip, stay = trip_ids(df.geometry.x.to_numpy(), df.geometry.y.to_numpy(), t,
                              max_gap=max_gap, stay_radius=stay_radius,
                              stay_duration=stay_duration, status=status,
                              stay_status=stay_status)
        return pd.DataFrame({"trip": trip, "stay": stay}, index=df.index)

    if by is None:
        labels = one(frame)
    else:
        parts = map_groups(one, frame, by, n_jobs)
        labels = pd.concat(parts) if parts else pd.DataFrame({"trip": [], "stay": []})
    out = frame.copy()
    out[trip_column] = labels["trip"].reindex(frame.index).to_numpy()
    if stay_column is not None:
        out[stay_column] = labels["stay"].reindex(frame.index).to_numpy()
    return out
