"""GPS-noise filter -- a time-windowed median or moving average.

Raw AIS positions are noisy.  [Mehri2021] section III-B: "The AIS raw data
often can have noisy data due to the nature of signal acquisition [53].
Anomaly detection in preprocessing refers to using methods to detect and clean
anomalies in trajectory data [54], [55]. To correct the data outliers, the
paper applies the moving average method [55] to the raw AIS data."  [55] is
Chen et al. (2020).  Messages arrive "at frequent intervals of approximately
2s-6min" [57], so a window counted in points would span anything from seconds
to hours; this one is counted in time.  The filter belongs before compression:
filter, then compress.

Filtering is a default step of every processing function of the package
(decided with the author, 2026-11-24): :func:`catraj.compression.compress`,
:func:`~catraj.compression.cpls_compress`,
:func:`~catraj.compression.two_stage_cpls_compress` and
:func:`catraj.annotation.annotate_points` apply this median, with a 30 second
window, to their inputs unless called with ``filter_window=None``;
``two_stage_cpls_select`` filters its speed series with ``speed_window``.
:func:`denoise` is the same step on its own.  The array-level algorithms
(``douglas_peucker``, ``cadp_select``, ``cpls_select``) filter nothing.  Time
comes from ``time_column`` or a DatetimeIndex; any other index raises a
``ValueError``, because a window in seconds on a row-number index would
silently filter by row number.  (:func:`rolling_filter` itself takes any
numeric times.)

Decisions taken with the author (2026-11-24)
--------------------------------------------

* **F1. Two statistics, median by default.**  ``method="median"`` or
  ``method="mean"``.  The mean is the paper's moving average [55].  The median
  is added because a single position spike moves a mean over every neighbour
  in the window, but leaves a median unchanged.
* **F2. A centred time window.**  Point ``i`` is replaced by the statistic of
  all points ``j`` with ``|t[j] - t[i]| <= window / 2``, itself included, in
  the trajectory's time units (seconds for UNIX times).  The paper gives no
  window; the default is 30 seconds (decided with the author, 2026-11-24).
  Because the window is in time, a gap in transmission
  longer than ``window / 2`` keeps the points either side of it apart, and
  irregular sampling is handled.  At the ends the window is truncated (fewer
  points), not padded.
* **F3. Positions and speed.**  x and y (projected coordinates) are always
  filtered, each on its own: a component-wise statistic, not the geometric
  median.  Further numeric columns are filtered when named in ``columns=``;
  this is meant for speed over ground, whose name varies between datasets, so
  it must be named.  Bearings in degrees (course, heading) are named in
  ``circular=`` as well, and filtered through their unit vector: the
  statistic of the sine and of the cosine, then ``atan2``, in [0, 360).  A
  window holding one valid value returns it unchanged, and where both
  statistics are exactly zero the direction is undefined and comes back as 0.

Decided here
------------

* **F4. Missing values** (NaN) in a named column are ignored inside a window;
  a missing value itself stays missing (no imputation).  Positions must be
  finite (else ``ValueError``).
* **F5. Times** must not decrease within a trajectory (``ValueError``);
  repeated times are allowed and fall in each other's window.
* **F6. Rows are never dropped or reordered.**  The result has the same rows,
  index and columns; only the geometry and the named columns change (a named
  integer column comes back as float).  Other columns that also encode
  position, such as ``lon`` and ``lat``, are not touched and become stale.
  With ``by=`` each trajectory is filtered on its own and the rows keep their
  input order, unlike :func:`.compress`, which returns them grouped.
* **F7.** A geographic CRS is refused, as in compression, by the same check.

What it costs, and what it does
-------------------------------

The statistics are computed on padded blocks of windows, found with
``np.searchsorted``, so the cost is the number of points times the points per
window.  On the 24-hour sample (106,894 points, 136 vessels, positions and
speed over ground filtered, 8,602 "not available" speeds set to NaN first)
filtering takes about 0.25 seconds.

Compressed afterwards with 2stage-CPLS (``epsilon`` 1000 m, ``speed_epsilon``
1.54 m/s, status, 100 m / 600 s stays, all five relations).  The positions
are filtered as in the first two columns; the third is the number of points
kept when the speed series is *not* filtered (``speed_window=None``), the
fourth when it is filtered with the same window, as the default does (P2 of
:mod:`..compression.two_stage`):

======  ======  =========  =========  =============  ========================
filter  window  kept,      kept,      repeated       displacement from
                speed raw  speed      positions      raw, m (median / 95th
                           filtered                  percentile / maximum)
======  ======  =========  =========  =============  ========================
none            5,228      5,228      9,200          0
median  30      12,006     4,759      16,171         0.0 / 21.8 / 155
median  60      18,239     4,756      20,764         0.5 / 44.7 / 303
median  120     16,382     3,341      20,916         0.7 / 75.6 / 648
mean    30      11,220     4,048      7,200          0.5 / 18.2 / 129
mean    60      17,135     3,894      13,780         1.6 / 38.8 / 231
mean    120     15,794     2,795      8,648          2.2 / 63.8 / 509
======  ======  =========  =========  =============  ========================

"Repeated positions" are consecutive points of one vessel at exactly the same
position.

Why filtering the positions alone raises the number of speed breakpoints:
consecutive points whose windows hold the same points get the same filtered
position (the sample has repeated times and irregular sampling), so repeated
positions rise from 9,200 raw to 20,764 (median, 60 s) and 13,780 (mean, 60
s), and the central-difference speed then swings between zero and double.
Douglas--Peucker on that series keeps most of its points.  That is why
2stage-CPLS also filters the speed series, with the same window: the default
(median 30 s on both) keeps 4,759 points, fewer than the 5,228 of the
unfiltered run.  The mean's smaller count of repeated positions at 30 s
(7,200) shows the effect depends on the statistic and the window; the table is
as measured, not tuned for.  On the default track plain CPLS keeps 35,020
points (barrier relations alone 604), against 43,074 (607) on the raw one.

Hourly series are not changed: a 30 second window holds one observation of an
hourly station (annotation filters station series with the same step; on the
sample's 16 stations, 400 observations, none of the wind speed, temperature,
pressure, humidity or compass-coded wind direction values changes at 30 s, nor
at one hour).

Corner cutting
--------------

A centred filter averages across a bend, so it can cut a corner round a
headland and take a track onto land.  Measured on the sample, against the
sample coastline (175 land polygons): the raw track has 10 segments that
intersect land and 2 points on land.  After filtering, no segment crosses land
that the raw segment between the same two rows did not (0 for the median at 30
and 60 seconds and the mean at 30 seconds, re-measured with the defaults), and
no point is moved onto land.  The sample
has few headlands close to the tracks, so this is no guarantee: measure
it on yours with :func:`..compression.segment_crossings` on the raw and the
filtered coordinates.

References
----------
[Mehri2021] Mehri, S., Alesheikh, A. A., Basiri, A. (2021).  A Contextual
    Hybrid Model for Vessel Movement Prediction.  IEEE Access 9, 45600-45613.
[55] Chen, X., Ling, J., Yang, Y., Zheng, H., Xiong, P., Postolache, O.,
    Xiong, Y. (2020).  Ship trajectory reconstruction from AIS sensory data
    via data quality control and prediction.  Mathematical Problems in
    Engineering 2020, 7191296.
[57] Xiao, F., Ligteringen, H., van Gulijk, C., Ale, B. (2015).  Comparison
    study on AIS data of ship traffic behavior.  Ocean Engineering 95, 84-93.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._crs import _check_geographic
from .._time import times_of

__all__ = ["rolling_filter", "denoise"]

_METHODS = ("median", "mean")

# Most values held in one padded block; bounds memory for a dense, long window.
_BLOCK_VALUES = 2_000_000


def _statistic(block, valid, method):
    """Median or mean over axis 1 of ``block`` (c, w, k), ignoring ``~valid``."""
    block = np.where(valid, block, np.nan)
    count = np.sum(~np.isnan(block), axis=1)
    empty = count == 0
    safe = np.maximum(count, 1)
    if method == "mean":
        out = np.nansum(block, axis=1) / safe
    else:
        ordered = np.sort(block, axis=1)  # NaN sorts last
        low = np.take_along_axis(ordered, ((safe - 1) // 2)[:, None, :], axis=1)
        high = np.take_along_axis(ordered, (safe // 2)[:, None, :], axis=1)
        out = ((low + high) / 2.0)[:, 0, :]
    out[empty] = np.nan
    return out


def rolling_filter(t, values, window, *, method="median", circular=False):
    """Centred, time-windowed median or mean of one or more series.

    Point ``i`` is replaced by the statistic of every point ``j`` with
    ``|t[j] - t[i]| <= window / 2``, itself included (decision F2 of
    :mod:`.filtering`).  Each column of ``values`` is filtered on its own;
    missing values are ignored inside a window and stay missing (F4).

    Parameters
    ----------
    t : array-like, shape (n,)
        Times, finite and non-decreasing, in one unit.
    values : array-like, shape (n,) or (n, k)
        The series to filter.  NaN marks a missing value.
    window : float
        The window's full width, in the unit of ``t``; positive and finite.
    method : {"median", "mean"}
    circular : bool or sequence of bool
        Which columns are bearings in degrees (one flag per column, or one
        flag for all).  A bearing is filtered through its unit vector: the
        statistic of its sine and of its cosine, then ``atan2``, in
        ``[0, 360)`` (decision F3).  A window with one valid value returns it
        unchanged; where both statistics are exactly zero the direction is
        undefined and comes back as 0.

    Returns
    -------
    ndarray of float
        The same shape as ``values``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be a positive, finite duration")
    t = np.asarray(t, dtype=float).ravel()
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("values must be 1-D (n,) or 2-D (n, k)")
    if values.shape[0] != t.size:
        raise ValueError(
            f"t and values must have equal length; got {t.size}, {values.shape[0]}"
        )
    if not np.isfinite(t).all():
        raise ValueError("times must be finite")
    if (np.diff(t) < 0).any():
        raise ValueError("times must not decrease along the trajectory")
    if np.isinf(values).any():
        raise ValueError("values must be finite or NaN (missing)")
    flat = values.reshape(t.size, 1 if values.ndim == 1 else values.shape[1])
    n, k = flat.shape
    bearing = np.broadcast_to(np.asarray(circular, dtype=bool), (k,))
    half = window / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    if bearing.any():
        angle = np.radians(flat[:, bearing])
        done = rolling_filter(
            t, np.column_stack([flat[:, ~bearing], np.sin(angle), np.cos(angle)]),
            window, method=method,
        )
        out = np.empty((n, k))
        out[:, ~bearing] = done[:, :k - angle.shape[1]]
        sin, cos = np.split(done[:, k - angle.shape[1]:], 2, axis=1)
        degrees = np.mod(np.degrees(np.arctan2(sin, cos)), 360.0)
        out[:, bearing] = np.where(degrees >= 360.0, 0.0, degrees)
        # A window holding one valid value returns it as it is: the unit
        # vector would give it back only up to rounding.
        seen = np.vstack([np.zeros((1, k), dtype=int),
                          np.cumsum(~np.isnan(flat), axis=0)])
        alone = (seen[hi] - seen[lo]) == 1
        out[alone & bearing] = flat[alone & bearing]
        out[np.isnan(flat)] = np.nan
        return out.reshape(values.shape)
    out = np.empty((n, k))
    width = hi - lo
    start = 0
    while start < n:
        step = min(n - start, 4096)
        while step > 1 and step * width[start:start + step].max() * k > _BLOCK_VALUES:
            step //= 2
        stop = start + step
        span = int(width[start:stop].max())
        cols = lo[start:stop, None] + np.arange(span)[None, :]
        valid = cols < hi[start:stop, None]
        block = flat[np.minimum(cols, n - 1)]  # (c, span, k)
        out[start:stop] = _statistic(block, valid[:, :, None], method)
        start = stop
    out[np.isnan(flat)] = np.nan
    return out.reshape(values.shape)


def _frame(df, window, method, columns, circular, time_column, by):
    _check_geographic(df)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in the trajectory: {missing}")
    if time_column is not None and time_column in columns:
        raise ValueError("the time column cannot be filtered")
    stray = [c for c in circular if c not in columns]
    if stray:
        raise ValueError(f"circular columns must also be in columns: {stray}")
    if time_column is None and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "filtering needs times, and the index is not a DatetimeIndex: pass "
            "time_column=, or filter_window=None to switch filtering off"
        )
    geom = df.geometry
    x, y = geom.x.to_numpy(dtype=float), geom.y.to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("positions must be finite; drop missing points first")
    t = df[time_column].to_numpy(dtype=float) if time_column else times_of(df)
    series = [x, y] + [df[c].to_numpy(dtype=float, na_value=np.nan) for c in columns]
    values = np.column_stack(series)
    bearing = [False, False] + [c in circular for c in columns]
    if by is None:
        done = rolling_filter(t, values, window, method=method, circular=bearing)
    else:
        done = np.empty_like(values)
        codes, _ = pd.factorize(df[by], use_na_sentinel=False)
        order = np.argsort(codes, kind="stable")  # keeps each track's own order
        cuts = np.flatnonzero(np.diff(codes[order])) + 1
        for rows in np.split(order, cuts):
            done[rows] = rolling_filter(
                t[rows], values[rows], window, method=method, circular=bearing
            )
    import geopandas as gpd

    out = df.copy()
    name = geom.name
    out[name] = gpd.GeoSeries(
        gpd.points_from_xy(done[:, 0], done[:, 1]), index=df.index, crs=df.crs
    )
    for k, column in enumerate(columns, start=2):
        out[column] = done[:, k]
    return out


def denoise(
    traj, window=30, *, method="median", columns=(), circular=(),
    time_column=None, by=None,
):
    """Filter GPS noise from trajectories with a time-windowed median or mean.

    The result has the same rows, index and columns as the input; only the
    geometry and the named ``columns`` change (F6).  Other columns that also
    encode position, such as ``lon`` and ``lat``, are not touched and become
    stale.

    Time comes from ``time_column`` or from a DatetimeIndex (MovingPandas); a
    window in seconds on any other index would silently filter by row number,
    so that raises a ``ValueError``.

    Parameters
    ----------
    traj : movingpandas.Trajectory, TrajectoryCollection or GeoDataFrame
        Positions in a projected CRS, in trajectory order.
    window : float
        Full width of the centred window, in the unit of the times (seconds
        for UNIX times).  Default 30 seconds, decided with the author
        (2026-11-24) when filtering became a default step of every
        processing function.
    method : {"median", "mean"}
    columns : iterable of str
        Further numeric columns to filter, such as speed over ground.
        Filtered columns come back as float.
    circular : iterable of str
        Those of ``columns`` that are bearings in degrees (course, heading).
        A bearing is filtered through its unit vector -- the statistic of its
        sine and of its cosine, then ``atan2`` -- and returned in
        ``[0, 360)``, so 359 and 1 give 0, not 180.
    time_column : str, optional
        Numeric time column (UNIX seconds); otherwise the DatetimeIndex.
    by : str, optional
        For a GeoDataFrame holding several trajectories: the column naming
        each one.  Each is filtered on its own, and the rows keep their order.

    Returns
    -------
    The same kind of object.
    """
    if isinstance(columns, str):
        columns = (columns,)
    columns = tuple(dict.fromkeys(columns))
    if isinstance(circular, str):
        circular = (circular,)
    circular = tuple(dict.fromkeys(circular))
    options = dict(
        method=method, columns=columns, circular=circular, time_column=time_column
    )

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        import movingpandas as mpd

        return mpd.TrajectoryCollection(
            [denoise(tr, window, **options) for tr in traj.trajectories]
        )

    if hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        import movingpandas as mpd

        return mpd.Trajectory(
            _frame(traj.df, window, by=None, **options), traj.id
        )

    return _frame(traj, window, by=by, **options)
