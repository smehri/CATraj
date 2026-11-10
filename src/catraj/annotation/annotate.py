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


def annotate_points(
    stations,
    x,
    y,
    t,
    spatial_method=None,
    temporal_method=None,
    circular=False,
    group_positions=True,
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
    epochs = np.sort(stations["t"].unique()).astype(float)
    series = np.empty((epochs.size, len(keys)), dtype=float)
    for i, epoch in enumerate(epochs):
        rows = stations[stations["t"] == epoch]
        fitted = _refit(spatial, rows["x"], rows["y"], rows["value"])
        series[i] = fitted(keys[:, 0], keys[:, 1])

    # --- stage 2, once per distinct position -------------------------------
    for k in range(len(keys)):
        values = series[:, k]
        ok = np.isfinite(values)
        if ok.sum() < 2:
            continue  # nothing to fit a curve through
        idx = np.flatnonzero(inverse == k)
        out[idx] = _refit(temporal, epochs[ok], values[ok])(t[idx])

    return out


def annotate(
    traj,
    stations,
    column,
    spatial_method=None,
    temporal_method=None,
    circular=False,
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
