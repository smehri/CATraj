"""Attaching context to trajectories -- composing the two stages.

Spatial interpolation gives a value at the moving object's position for every
epoch; temporal interpolation fits a curve through that series and reads it at
that object's exact timestamp.  This module joins them and writes the result onto
the trajectory.

No model selection happens here.  The caller supplies the methods -- chosen by
:mod:`.spatial_evaluation` and :mod:`.annotation_evaluation`, or simply known
in advance -- and annotation applies them.
"""

from __future__ import annotations

import copy

import numpy as np

from .._parallel import run_parallel
from .._time import times_of
from .interpolators import DLT
from .spatial import IDW

__all__ = ["build_series", "annotate_points", "annotate"]


def build_series(stations, x, y, spatial_method=None, circular=False):
    """Stage 1 -- estimate a context series at one position.

    Parameters
    ----------
    stations : DataFrame
        Station observations with columns ``x``, ``y``, ``t``, ``value``.
        Rows sharing a ``t`` form one epoch and are fitted together.
    x, y : float
        Query position, in the stations' CRS (projected metres).
    spatial_method : SpatialInterpolator, optional
        An *unfitted* instance, refitted per epoch.  Defaults to ``IDW``.
    circular : bool
        Treat values as bearings in degrees.

    Returns
    -------
    (times, values)
        One value per epoch.  Epochs the method cannot reach yield NaN.
    """
    method = spatial_method if spatial_method is not None else IDW(circular=circular)
    epochs = np.sort(stations["t"].unique())

    values = np.empty(epochs.size, dtype=float)
    for i, epoch in enumerate(epochs):
        rows = stations[stations["t"] == epoch]
        fit = method.fit(rows["x"], rows["y"], rows["value"])
        values[i] = fit(x, y)
    return epochs.astype(float), values


#: Above this many (epoch x point) cells, ``chunk_epochs="auto"`` processes
#: the points in time blocks instead of all at once.
_AUTO_CHUNK_CELLS = 20_000_000
#: Epochs per block when ``chunk_epochs="auto"`` decides to chunk.
_AUTO_CHUNK_EPOCHS = 24
#: Extra epochs either side of a block, so a curve fitted within it is not
#: cut off right at the block's first or last query.
_CHUNK_PAD = 3


def annotate_points(
    stations,
    x,
    y,
    t,
    spatial_method=None,
    temporal_method=None,
    circular=False,
    group_positions=True,
    chunk_epochs="auto",
    n_jobs=1,
):
    """Estimate context at scattered ``(x, y, t)`` points.

    The two stages run per position: spatial across stations at each epoch,
    then temporal through the resulting series.

    **Each epoch's spatial interpolator is fitted once and evaluated at every
    position at once.**  The station observations do not depend on where the
    query is, so refitting per position would repeat identical work -- with
    thousands of positions and dozens of epochs that difference is hours
    against seconds.

    Positions that coincide -- as every point of a stationary object does --
    share one temporal fit.  Pass ``group_positions=False`` to fit each point
    separately.

    **Large inputs are processed in time blocks.**  Stage 1 holds one value per
    epoch per position, which for months of data and millions of positions
    would not fit in memory.  With ``chunk_epochs`` the points are grouped by
    time, ``chunk_epochs`` epochs at a time, and each group sees only the
    epochs around it (plus a margin of a few epochs either side).  For DLT the
    result is identical to processing everything at once, since DLT only ever
    uses the two epochs bracketing a point.  The spline and piecewise
    polynomial are then fitted to each block's epochs rather than to the whole
    series, which can change values slightly; the margin keeps a point's
    nearest epochs in its fit.  ``"auto"`` (the default) chunks only when the
    input is large; ``None`` never chunks.

    **DLT is vectorised.**  For positions whose series has no gaps, DLT is
    evaluated for all points at once rather than fitted position by position;
    the result is the same.

    **Time blocks can run in parallel.**  They are independent, so with
    ``n_jobs`` they are spread over processes: ``1`` (the default) runs
    serially, ``-1`` uses every core.  The result is the same either way.
    Without chunking there is a single block, and ``n_jobs`` has no effect.

    Returns
    -------
    ndarray
        One context value per input point; NaN where it could not be estimated.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if not (x.size == y.size == t.size):
        raise ValueError("x, y and t must have equal length")

    temporal = temporal_method if temporal_method is not None else DLT(circular=circular)
    spatial = spatial_method if spatial_method is not None else IDW(circular=circular)
    out = np.full(x.size, np.nan)
    if x.size == 0:
        return out

    epochs = np.sort(stations["t"].unique()).astype(float)
    chunk = _chunk_size(chunk_epochs, epochs.size, x.size)
    if chunk is None or epochs.size <= chunk:
        return _annotate_block(stations, epochs, x, y, t, spatial, temporal,
                               group_positions)

    # Group the points by the epoch interval they fall in, `chunk` epochs to a
    # group; each group gets the station rows of its epochs plus a margin.
    position = np.searchsorted(epochs, t, side="right") - 1
    block = np.clip(position, 0, epochs.size - 1) // chunk
    order = np.argsort(block, kind="stable")
    bounds = np.flatnonzero(np.diff(block[order])) + 1
    station_t = stations["t"].to_numpy(dtype=float)
    pieces, tasks = np.split(order, bounds), []
    for idx in pieces:
        b = block[idx[0]]
        first = max(b * chunk - _CHUNK_PAD, 0)
        last = min((b + 1) * chunk + _CHUNK_PAD, epochs.size - 1)
        rows = (station_t >= epochs[first]) & (station_t <= epochs[last])
        tasks.append((stations[rows], epochs[first:last + 1], x[idx], y[idx], t[idx]))
    values = run_parallel(_annotate_block, tasks, n_jobs, spatial=spatial,
                          temporal=temporal, group_positions=group_positions)
    for idx, block_values in zip(pieces, values):
        out[idx] = block_values
    return out


def _chunk_size(chunk_epochs, n_epochs, n_points):
    if chunk_epochs is None:
        return None
    if isinstance(chunk_epochs, str):
        if chunk_epochs != "auto":
            raise ValueError("chunk_epochs must be 'auto', None or a positive integer")
        if n_epochs * n_points <= _AUTO_CHUNK_CELLS:
            return None
        return _AUTO_CHUNK_EPOCHS
    chunk = int(chunk_epochs)
    if chunk < 1:
        raise ValueError("chunk_epochs must be 'auto', None or a positive integer")
    return chunk


def _annotate_block(stations, epochs, x, y, t, spatial, temporal, group_positions):
    """Both stages for one set of points against one set of epochs."""
    out = np.full(x.size, np.nan)

    # Distinct positions: a trajectory at anchor repeats one position many
    # times, and the spatial estimate there is identical every time.
    if group_positions:
        keys, inverse = np.unique(
            np.column_stack([x, y]), axis=0, return_inverse=True
        )
        inverse = np.asarray(inverse).ravel()
    else:
        keys = np.column_stack([x, y])
        inverse = np.arange(x.size)

    # --- stage 1, once per epoch, evaluated at every distinct position -----
    series = np.empty((epochs.size, len(keys)), dtype=float)
    station_t = stations["t"].to_numpy(dtype=float)
    for i, epoch in enumerate(epochs):
        rows = stations[station_t == epoch]
        fitted = _refit(spatial, rows["x"], rows["y"], rows["value"])
        series[i] = fitted(keys[:, 0], keys[:, 1])

    # --- stage 2 -------------------------------------------------------------
    todo = np.ones(len(keys), dtype=bool)
    if type(temporal) is DLT and epochs.size >= 2:
        complete = np.isfinite(series).all(axis=0)
        points = np.flatnonzero(complete[inverse])
        out[points] = _dlt_vectorised(
            epochs, series, inverse[points], t[points], temporal.circular
        )
        todo = ~complete

    # Everything else -- other methods, and series with gaps -- position by
    # position, which drops the missing epochs from each fit.
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(keys)))
    ends = np.r_[starts[1:], order.size]
    for k in np.flatnonzero(todo):
        values = series[:, k]
        ok = np.isfinite(values)
        if ok.sum() < 2:
            continue  # nothing to fit a curve through
        idx = order[starts[k]:ends[k]]
        out[idx] = _refit(temporal, epochs[ok], values[ok])(t[idx])

    return out


def _dlt_vectorised(epochs, series, keys, t, circular):
    """DLT for many points at once, each on its own position's series.

    The same straight line between the two bracketing epochs that
    :class:`DLT` draws, with the end values held beyond the observed span;
    bearings are interpolated on their unit vectors.
    """
    j = np.clip(np.searchsorted(epochs, t, side="right") - 1, 0, epochs.size - 2)
    frac = np.clip((t - epochs[j]) / (epochs[j + 1] - epochs[j]), 0.0, 1.0)
    lo, hi = series[j, keys], series[j + 1, keys]
    if not circular:
        return lo + (hi - lo) * frac
    lo, hi = np.radians(lo), np.radians(hi)
    u = np.cos(lo) + (np.cos(hi) - np.cos(lo)) * frac
    v = np.sin(lo) + (np.sin(hi) - np.sin(lo)) * frac
    return np.mod(np.degrees(np.arctan2(v, u)), 360.0)


def annotate(
    traj,
    stations,
    column,
    spatial_method=None,
    temporal_method=None,
    circular=False,
    chunk_epochs="auto",
    n_jobs=1,
):
    """Annotate a MovingPandas trajectory, or a GeoDataFrame, with context.

    Context is written as an ordinary column on the trajectory's dataframe, so
    it survives the ``.iloc`` slicing that trajectory compression performs --
    a compressed trajectory keeps the context of the points it retains.

    Parameters
    ----------
    traj : movingpandas.Trajectory or GeoDataFrame
        Modified in place; also returned, so calls can be chained.
    stations : DataFrame
        Columns ``x``, ``y``, ``t``, ``value``.  Positions in the trajectory's
        CRS; ``t`` in the same units as the trajectory's index.
    column : str
        Name for the new column.
    spatial_method, temporal_method : optional
        Unfitted instances.  Default to ``IDW`` and ``DLT``.
    circular : bool
        Treat values as bearings in degrees.
    chunk_epochs : "auto", int or None
        Process long inputs in time blocks; see :func:`annotate_points`.
    n_jobs : int
        Processes for those blocks; ``-1`` uses every core.

    Returns
    -------
    The trajectory, with ``column`` added.
    """
    df = getattr(traj, "df", traj)

    geom = df.geometry
    t = _times_of(df)
    values = annotate_points(
        stations,
        geom.x.to_numpy(),
        geom.y.to_numpy(),
        t,
        spatial_method=spatial_method,
        temporal_method=temporal_method,
        circular=circular,
        chunk_epochs=chunk_epochs,
        n_jobs=n_jobs,
    )
    df[column] = values
    return traj


def _refit(prototype, *fit_args):
    """A fresh interpolator with the prototype's settings, fitted to new data.

    The caller's instance is a template, never refitted in place: one epoch's
    or one position's fit would otherwise leak into the next.  Spatial
    interpolators take ``(x, y, values)`` and temporal ones ``(t, values)``, so
    the fitting arguments are passed straight through.
    """
    fresh = copy.copy(prototype)
    fresh._fitted = False
    return fresh.fit(*fit_args)


_times_of = times_of  # kept under its old name; the logic lives in catraj._time
