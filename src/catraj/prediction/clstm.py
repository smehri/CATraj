"""CLSTM, Context-aware LSTM -- [Mehri2021] sections III-A, IV-B to IV-E, V-B.

The second half of the hybrid model (Fig. 5).  The curvilinear model of
:mod:`.curvilinear` predicts a vessel's next three positions, T', from its
speed and course; a Long Short-Term Memory network, **one per vessel type**,
takes those three points and returns the three corrected positions, T.  The
ordinary LSTM it is compared with is the same network trained once on every
vessel type together.  The context matters because the deviation
``delta_k = (x_k - x'_k) i + (y_k - y'_k) j`` (Eq. 2) differs by type, and
pooling the types hides it (the Simpson's paradox argument of section IV-E).

This module holds the samples (Eqs. 21 and 22), the split by trajectory and
the error measures (Eqs. 23-25).  Torch is needed only by the networks, which
import it inside the functions that use it, so importing this module does not.

Decided with the author (2026-11-24)
------------------------------------

* **C1. Input and output exactly as Eqs. 21 and 22.**  A sample is a sequence
  of three steps; step j is ``(x'_{t+j}, y'_{t+j}, tau_j)``.  The output is the
  three absolute positions ``(x_{t+j}, y_{t+j})``.  No observed history, no
  relative coordinates.
* **C2. One network per vessel type**, plus the ordinary LSTM as the same
  class with no context.
* **C3. PyTorch**, as the optional extra ``catraj[predict]``.
* **C4. Data.**  Tests and the tutorial run on the committed sample; accuracy
  results come from the full Brest archive, through the benchmark script.

Decided here
------------

* **C5.** Constant speed and course (see :mod:`.curvilinear`).
* **C6. Speed and course come from the AIS fields** SOG and COG of the row at
  t.  SOG is in knots unless ``speed_unit="native"`` (position units per
  second); ``1 knot = 1852/3600 m/s``.  COG is in degrees clockwise from north.
* **C7. Unavailable values drop the sample.**  SOG ``>= 102.3`` (AIS "not
  available"), COG ``>= 360`` (also read on the raw column when
  ``speed_unit="native"``), or NaN in either.  The row still serves as a
  vertex for the targets of other samples.
* **C8. Grid north is taken as north** (see :mod:`.curvilinear`).
* **C9. Delta t per trajectory, by Eq. 12**, over the trajectory as given:
  ``dt = (t_last - t_first) / (n - 1)``.
* **C10. Every vertex is a prediction origin**, and the actual positions at
  ``t + j * dt`` are linearly interpolated in time along the same trajectory.
  A sample whose ``t + steps * dt`` lies past the last vertex is dropped.
* **C11. tau_j is the time offset** ``j * dt`` in seconds, not a timestamp.
* **C12. Vessel type groups.**  A null, empty or missing type is ``"Other"``
  (the paper's definition).  A group with fewer than ``min_samples`` training
  samples gets no network and is listed in ``dropped_groups_``; its
  predictions are NaN.
* **C13. Split by trajectory**, never by sample, seeded
  (:func:`split_by_trajectory`).
* **C14. A trajectory is what the caller passes**: a ``Trajectory``, each
  member of a ``TrajectoryCollection``, or each ``by`` group of a
  ``GeoDataFrame``.  There is no gap splitting here.
* **H1. Architecture.**  ``LSTM(3, 64, 1, batch_first=True)`` over the three
  steps, then ``Linear(64, 2)`` on each step's hidden state.  Zero initial
  states; PyTorch's default weight initialisation.
* **H2. Scaling.**  Per network, every input feature and every output
  coordinate is standardised with the mean and standard deviation of that
  network's training samples, stored and inverted on prediction.
* **H3. Loss.**  Mean squared error on the standardised outputs.
* **H4. Optimiser.**  Adam, learning rate 1e-3, batch size 256, at most 200
  epochs.
* **H5. Early stopping.**  10 % of the training trajectories (by C13) are held
  out; training stops after 10 epochs without improvement and the best
  weights are restored.  With fewer than 2 training trajectories there is no
  validation and ``max_epochs`` epochs are run.
* **H6. Determinism.**  ``seed`` seeds numpy and torch.  On CPU two fits with
  the same seed give identical predictions.

Details of the sample builder (also decided here)
-------------------------------------------------

* Time is read as :func:`catraj.compression.two_stage_cpls_compress` reads it:
  from ``time_column`` or from a DatetimeIndex, in seconds.  Times must not
  decrease; a repeated time keeps its first row (``np.interp`` needs strictly
  increasing times).
* The positions must be in a projected CRS; a geographic one raises.
* A trajectory with fewer than 2 vertices, or no samples that fit, yields
  none; that is not an error.
* ``context_column`` is converted to ``str`` after nulls are filled.  If the
  type is not constant within a trajectory, the most frequent non-null value
  (the first met on a tie) is used for all its samples.
* The trajectory id is the ``by`` value, the movingpandas id, or ``0`` for a
  single ``GeoDataFrame`` without ``by``.  Without ``context_column`` the
  group of every sample is ``"all"``.
* Everything stays float64; a network casts to float32 at its boundary only.
* :func:`prediction_errors`: ``rmse`` is Eq. 25 read as the root of the mean of
  ``dx^2 + dy^2`` (the combination of the RMSEs of Eqs. 23 and 24 as
  ``sqrt(rmse_x^2 + rmse_y^2)``); the point-wise error is the distance between
  the predicted and the actual vertex.  A sample with a NaN prediction counts
  in ``missing`` and not in ``samples`` or any statistic.

References
----------
[Mehri2021] Mehri, S., Alesheikh, A. A., Basiri, A. (2021).  A Contextual
    Hybrid Model for Vessel Movement Prediction.  IEEE Access 9, 45600-45613.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._crs import _check_geographic
from .._time import times_of
from .curvilinear import KNOT, curvilinear_predict

__all__ = [
    "PredictionSamples",
    "make_samples",
    "split_by_trajectory",
    "prediction_errors",
]

_SOG_UNAVAILABLE = 102.3  # AIS "speed not available"
_COG_UNAVAILABLE = 360.0  # AIS "course not available"
_OTHER = "Other"


# ---------------------------------------------------------------- samples


@dataclass
class PredictionSamples:
    """Prediction samples: Eq. 21 inputs and Eq. 22 targets, with their keys.

    Attributes
    ----------
    inputs : ndarray, shape (m, steps, 3)
        ``(x', y', tau)`` per step: the curvilinear position and the time
        offset in seconds (C11).
    targets : ndarray, shape (m, steps, 2)
        The actual ``(x, y)`` per step, interpolated in time (C10).
    origin : ndarray, shape (m, 2)
        The position at t.
    time : ndarray, shape (m,)
        t of the origin, in seconds.
    dt : ndarray, shape (m,)
        The trajectory's time step (Eq. 12).
    group : ndarray of object, shape (m,)
        The vessel type (null is ``"Other"``), or ``"all"`` with no context.
    trajectory : ndarray of object, shape (m,)
        The trajectory id.
    """

    inputs: np.ndarray
    targets: np.ndarray
    origin: np.ndarray
    time: np.ndarray
    dt: np.ndarray
    group: np.ndarray
    trajectory: np.ndarray

    def __len__(self):
        return int(self.inputs.shape[0])

    @property
    def curvilinear(self):
        """The curvilinear prediction T', ``inputs[..., :2]``."""
        return self.inputs[..., :2]

    def subset(self, mask_or_indices):
        """The samples selected by a boolean mask or an array of indices."""
        sel = np.asarray(mask_or_indices)
        if sel.dtype != bool:
            sel = sel.astype(np.intp)
        return PredictionSamples(
            *(getattr(self, name)[sel] for name in _FIELDS)
        )

    @classmethod
    def concat(cls, samples):
        """Join several :class:`PredictionSamples` end to end."""
        samples = list(samples)
        if not samples:
            raise ValueError("nothing to concatenate")
        return cls(*(
            np.concatenate([getattr(s, name) for s in samples])
            for name in _FIELDS
        ))


_FIELDS = ("inputs", "targets", "origin", "time", "dt", "group", "trajectory")


def _empty(steps):
    return PredictionSamples(
        inputs=np.empty((0, steps, 3)), targets=np.empty((0, steps, 2)),
        origin=np.empty((0, 2)), time=np.empty(0), dt=np.empty(0),
        group=np.empty(0, dtype=object), trajectory=np.empty(0, dtype=object),
    )


def _require(df, *columns):
    for name in columns:
        if name is not None and name not in df.columns:
            raise KeyError(f"column {name!r} not found in the trajectory data")


def _vessel_type(df, context_column):
    """The trajectory's type: the most frequent non-null value, else Other."""
    if context_column is None:
        return "all"
    values = df[context_column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(object)
    values = values[values.notna()].astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return _OTHER
    return str(values.value_counts(sort=True).index[0])


def _trajectory_samples(df, ident, speed_column, course_column, context_column,
                        time_column, steps, speed_unit):
    _check_geographic(df)
    _require(df, speed_column, course_column, context_column, time_column)
    group = _vessel_type(df, context_column)
    t = df[time_column].to_numpy(dtype=float) if time_column else times_of(df)
    if (np.diff(t) < 0).any():
        raise ValueError("times must not decrease along the trajectory")
    geom = df.geometry
    x, y = geom.x.to_numpy(float), geom.y.to_numpy(float)
    sog = df[speed_column].to_numpy(dtype=float)
    cog = df[course_column].to_numpy(dtype=float)
    first = np.concatenate([[True], np.diff(t) > 0]) if t.size else np.empty(0, bool)
    t, x, y, sog, cog = t[first], x[first], y[first], sog[first], cog[first]
    n = t.size
    if n < 2:
        return _empty(steps)
    dt = (t[-1] - t[0]) / (n - 1)  # Eq. 12
    fits = t + steps * dt <= t[-1]
    with np.errstate(invalid="ignore"):
        usable = (sog < _SOG_UNAVAILABLE) & (cog < _COG_UNAVAILABLE)
    k = np.flatnonzero(fits & usable & np.isfinite(sog) & np.isfinite(cog))
    if k.size == 0:
        return _empty(steps)
    speed = sog[k] * KNOT if speed_unit == "knots" else sog[k]
    px, py = curvilinear_predict(x[k], y[k], speed, cog[k], dt, steps)
    offset = dt * np.arange(1, steps + 1)
    epochs = t[k][:, None] + offset[None, :]
    inputs = np.stack([px, py, np.broadcast_to(offset, px.shape)], axis=-1)
    targets = np.stack(
        [np.interp(epochs, t, x), np.interp(epochs, t, y)], axis=-1
    )
    return PredictionSamples(
        inputs=inputs, targets=targets,
        origin=np.column_stack([x[k], y[k]]), time=t[k],
        dt=np.full(k.size, dt),
        group=np.full(k.size, group, dtype=object),
        trajectory=np.full(k.size, ident, dtype=object),
    )


def make_samples(
    traj, *, speed_column, course_column, context_column=None,
    time_column=None, by=None, steps=3, speed_unit="knots",
):
    """Build the prediction samples of Eqs. 21 and 22 from trajectories.

    Every vertex of a trajectory is a prediction origin (C10).  The input is
    the curvilinear prediction of the next ``steps`` positions from the
    origin's speed and course (C5, C6), with the offsets ``j * dt`` (C11); the
    target is where the trajectory actually is at those times, interpolated
    linearly in time.

    Parameters
    ----------
    traj : movingpandas.TrajectoryCollection, Trajectory or GeoDataFrame
        Points in trajectory order, in a projected CRS.  Normally the output
        of :func:`catraj.compression.two_stage_cpls_compress`.
    speed_column, course_column : str
        AIS speed over ground (knots) and course over ground (degrees).
    context_column : str, optional
        The vessel type.  A null or empty type is ``"Other"``; a type that is
        not constant within a trajectory is replaced by its most frequent
        non-null value.  Without it every sample is in the group ``"all"``.
    time_column : str, optional
        Numeric time column (UNIX seconds); the index by default.
    by : str, optional
        For a GeoDataFrame holding several trajectories: the column naming
        each one.  It becomes the trajectory id; otherwise the id is the
        movingpandas id.
    steps : int
        Number of predicted positions per sample.
    speed_unit : {"knots", "native"}
        ``"native"``: the speed column is already in position units per second.

    Returns
    -------
    PredictionSamples

    Raises
    ------
    KeyError
        If a named column is missing.
    ValueError
        For a geographic CRS, an unknown ``speed_unit``, or times that
        decrease.
    """
    if speed_unit not in ("knots", "native"):
        raise ValueError("speed_unit must be 'knots' or 'native'")
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")

    def one(df, ident):
        return _trajectory_samples(
            df, ident, speed_column, course_column, context_column,
            time_column, steps, speed_unit,
        )

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        parts = [one(tr.df, tr.id) for tr in traj.trajectories]
    elif hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        parts = [one(traj.df, traj.id)]
    elif by is not None:
        _require(traj, by)
        parts = [one(g, key) for key, g in traj.groupby(by, sort=False)]
    else:
        parts = [one(traj, 0)]
    parts = [p for p in parts if len(p)]
    return PredictionSamples.concat(parts) if parts else _empty(steps)


def split_by_trajectory(samples, test_size=0.2, seed=0):
    """Split samples into train and test, keeping every trajectory whole (C13).

    The unique trajectory ids, in order of first appearance, are shuffled with
    ``numpy.random.default_rng(seed)``; the first
    ``round(test_size * n_trajectories)`` of them, at least 1 and at most
    ``n - 1``, are the test set.

    Returns
    -------
    train, test : PredictionSamples

    Raises
    ------
    ValueError
        With fewer than 2 trajectories.
    """
    ids = list(dict.fromkeys(samples.trajectory.tolist()))
    n = len(ids)
    if n < 2:
        raise ValueError("splitting needs at least 2 trajectories")
    order = np.arange(n)
    np.random.default_rng(seed).shuffle(order)
    n_test = min(max(int(round(test_size * n)), 1), n - 1)
    in_test = {ids[i] for i in order[:n_test]}
    mask = np.fromiter(
        (ident in in_test for ident in samples.trajectory.tolist()),
        dtype=bool, count=len(samples),
    )
    return samples.subset(~mask), samples.subset(mask)


# ----------------------------------------------------------------- errors


def prediction_errors(predicted, samples):
    """Eqs. 23-25 and the point-wise horizontal error, per group and overall.

    Parameters
    ----------
    predicted : array-like, shape (m, steps, 2)
        Predicted ``(x, y)`` per step: a predictor's output, or
        ``samples.curvilinear`` to score the physical model alone.  A sample
        with a NaN is left out of the statistics and counted in ``missing``.
    samples : PredictionSamples
        Supplies the actual positions and the groups.

    Returns
    -------
    pandas.DataFrame
        One row per group (sorted), then ``"all"``.  Columns: ``samples``
        (scored), ``missing``, ``rmse`` (Eq. 25, over all steps),
        ``rmse_t+1`` ... ``rmse_t+steps``, ``mean_error`` and ``median_error``
        (the point-wise distance, in position units).
    """
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != samples.targets.shape:
        raise ValueError(
            f"predicted has shape {predicted.shape}; expected "
            f"{samples.targets.shape}, that of the samples' targets"
        )
    steps = predicted.shape[1]
    delta = predicted - samples.targets
    missing = np.isnan(delta).any(axis=(1, 2))
    dist = np.hypot(delta[..., 0], delta[..., 1])  # (m, steps)
    columns = (["samples", "missing", "rmse"]
               + [f"rmse_t+{j}" for j in range(1, steps + 1)]
               + ["mean_error", "median_error"])
    groups = samples.group.astype(str)

    def row(select):
        total = int(select.sum())
        ok = select & ~missing
        d = dist[ok]
        out = {"samples": int(ok.sum()), "missing": total - int(ok.sum())}
        if d.size == 0:
            out.update({c: np.nan for c in columns[2:]})
            return out
        out["rmse"] = float(np.sqrt(np.mean(d ** 2)))
        for j in range(steps):
            out[f"rmse_t+{j + 1}"] = float(np.sqrt(np.mean(d[:, j] ** 2)))
        out["mean_error"] = float(d.mean())
        out["median_error"] = float(np.median(d))
        return out

    names = sorted(set(groups.tolist()))
    rows = {name: row(groups == name) for name in names}
    rows["all"] = row(np.ones(len(samples), dtype=bool))
    return pd.DataFrame.from_dict(rows, orient="index")[columns]
