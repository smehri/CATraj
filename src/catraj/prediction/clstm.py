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

* **C1. Input exactly as Eq. 21; what the network learns changed on
  2026-11-25.**  A sample is a sequence of three steps; step j is
  ``(x'_{t+j}, y'_{t+j}, tau_j)``; no observed history.  The final output of
  :meth:`ClstmPredictor.predict` is always the three absolute positions
  ``(x_{t+j}, y_{t+j})``.  As first decided, the network regressed those
  absolute positions (``target="absolute"``).  Changed with the author on
  2026-11-25 to ``target="deviation"`` (the default): the network learns the
  deviation ``delta = actual - T'`` of Eq. 2 and ``predict`` returns
  ``T' + delta_hat``.  Reason: absolute UTM coordinates standardised over the
  whole area have a standard deviation of tens of kilometres, so a network
  that reaches a standardised RMSE of 1.7e-3 is still about 70 m off, more
  than the curvilinear error over a horizon of seconds.  Measured on the full
  archive (clean Cargo, 1.03 million training samples, 30 epochs, mean /
  median point-wise error on held-out trips): curvilinear 59.2 / 15.6 m,
  absolute 95.5 / 52.2 m, deviation 60.8 / 18.4 m.
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
* **H2. Scaling.**  Per network, every input feature and every learned
  target coordinate (the absolute position, or with ``target="deviation"``
  the deviation ``delta``) is standardised with the mean and standard
  deviation of that network's training samples, stored and inverted on
  prediction.
* **H3. Loss.**  Mean squared error on the standardised outputs.
* **H4. Optimiser.**  Adam, learning rate 1e-3, batch size 256, at most 200
  epochs.
* **H5. Early stopping.**  10 % of the training trajectories (by C13) are held
  out; training stops after 10 epochs without improvement and the best
  weights are restored.  With fewer than 2 training trajectories there is no
  validation and ``max_epochs`` epochs are run.
* **H6. Determinism.**  ``seed`` seeds numpy and torch.  On CPU two fits with
  the same seed give identical predictions.

Details of the networks (also decided here)
-------------------------------------------

* The scalers of H2 are computed on the network's training trajectories only
  (the validation trajectories of H5 excluded); each input feature over all
  samples and steps.  A zero standard deviation becomes 1.
* Every network starts from the same seed, so a group's network does not depend
  on the other groups.  Batches are drawn with a seeded ``torch.randperm``.
* The training loss of an epoch is the mean over its samples; the validation
  loss is the MSE on the standardised outputs of the held-out trajectories.
  ``validation_size=0`` switches validation off.
* ``context=False`` trains one network on every sample, whatever its group; it
  is stored under the name ``"all"``.  A group met at ``predict`` time that has
  no network gives NaN rows.
* :meth:`ClstmPredictor.save` writes one ``torch.save`` file of plain types and
  tensors only, so :meth:`ClstmPredictor.load` reads it with
  ``weights_only=True``, on any device.

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

import copy
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
    "ClstmPredictor",
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


# --------------------------------------------------------------- networks

_INSTALL_HINT = (
    "CLSTM needs PyTorch: pip install catraj[predict]  (for an NVIDIA GPU, "
    "install the CUDA build of torch first, from https://pytorch.org)"
)
_PREDICT_BATCH = 65_536


def _torch():
    try:
        import torch
    except ImportError:
        raise ImportError(_INSTALL_HINT) from None
    return torch


def _network(torch, hidden_size, num_layers):
    """H1: an LSTM over the three steps, then a linear layer on every step."""

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(3, hidden_size, num_layers, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, 2)

        def forward(self, x):
            out, _ = self.lstm(x)  # zero initial states
            return self.head(out)

    return _Net()


def _scaler(inputs, targets):
    """H2: mean and standard deviation of each input feature and coordinate."""
    def stats(a):
        flat = a.reshape(-1, a.shape[-1])
        std = flat.std(axis=0)
        return flat.mean(axis=0), np.where(std > 0, std, 1.0)

    in_mean, in_std = stats(inputs)
    out_mean, out_std = stats(targets)
    return {"in_mean": in_mean, "in_std": in_std,
            "out_mean": out_mean, "out_std": out_std}


class ClstmPredictor:
    """CLSTM: one LSTM network per vessel type (C2), or one for all (the LSTM).

    Parameters
    ----------
    context : bool
        ``True``: a network per ``samples.group`` value (CLSTM).  ``False``: one
        network for every sample, ignoring the group (the ordinary LSTM).
    hidden_size, num_layers : int
        The LSTM of H1.
    lr, batch_size, max_epochs : float, int, int
        Adam and its schedule (H4).
    target : {"deviation", "absolute"}
        What the network learns (C1): ``"deviation"`` (default) the difference
        between the actual positions and the curvilinear ones, to which
        ``predict`` adds the curvilinear positions back; ``"absolute"`` the
        absolute positions.  Either way ``predict`` returns absolute positions.
    patience, validation_size : int, float
        Early stopping (H5): the epochs without improvement that end the
        training, and the share of the training trajectories held out.
    min_samples : int
        A group with fewer samples gets no network (C12).
    seed : int
        Seeds numpy and torch (H6).
    device : str
        ``"auto"`` is ``"cuda"`` when available, else ``"cpu"``; or any string
        :class:`torch.device` accepts.

    Attributes
    ----------
    groups_ : list of str
        The groups with a network, sorted.
    dropped_groups_ : list of str
        The groups with fewer than ``min_samples`` samples, sorted.
    history_ : dict
        Group -> list of ``(epoch, train_loss, validation_loss)``; the
        validation loss is NaN when there was none.

    Raises
    ------
    ImportError
        On construction, if PyTorch is not installed.
    """

    _CONFIG = ("context", "hidden_size", "num_layers", "lr", "batch_size",
               "max_epochs", "patience", "validation_size", "min_samples", "seed",
               "target")

    def __init__(self, *, context=True, hidden_size=64, num_layers=1, lr=1e-3,
                 batch_size=256, max_epochs=200, patience=10,
                 validation_size=0.1, min_samples=200, seed=0, device="auto",
                 target="deviation"):
        torch = _torch()
        if target not in ("deviation", "absolute"):
            raise ValueError("target must be 'deviation' or 'absolute'")
        self.target = target
        self.context = bool(context)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.validation_size = float(validation_size)
        self.min_samples = int(min_samples)
        self.seed = int(seed)
        if min(self.batch_size, self.max_epochs, self.hidden_size) < 1:
            raise ValueError("batch_size, max_epochs and hidden_size must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        torch.device(self.device)  # refuse an unknown device now
        self._models = {}
        self._info = {}
        self.groups_ = []
        self.dropped_groups_ = []
        self.history_ = {}
        self._fitted = False

    # -------------------------------------------------------------- fit

    def _keys(self, samples):
        if not self.context:
            return np.full(len(samples), "all", dtype=object)
        return samples.group.astype(str).astype(object)

    def fit(self, samples):
        """Train one network per group (H1-H6); returns ``self``."""
        torch = _torch()
        if len(samples) == 0:
            raise ValueError("no samples to fit")
        keys = self._keys(samples)
        self._models, self._info, self.history_ = {}, {}, {}
        dropped = []
        for name in sorted(set(keys.tolist())):
            part = samples.subset(keys == name)
            if len(part) < self.min_samples:
                dropped.append(name)
                continue
            self._train_group(torch, name, part)
        self.groups_ = sorted(self._models)
        self.dropped_groups_ = dropped
        self._fitted = True
        return self

    def _train_group(self, torch, name, part):
        device = torch.device(self.device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        val_idx = None
        train_idx = np.arange(len(part))
        if len(set(part.trajectory.tolist())) >= 2 and self.validation_size > 0:
            _, held = split_by_trajectory(part, self.validation_size, self.seed)
            in_val = np.isin(part.trajectory.astype(str), held.trajectory.astype(str))
            train_idx = np.flatnonzero(~in_val)
            val_idx = torch.as_tensor(np.flatnonzero(in_val), device=device)
        # H2: the scalers come from the training trajectories only.
        learned = part.targets
        if self.target == "deviation":  # Eq. 2: delta = actual - T'
            learned = part.targets - part.curvilinear
        scaler = _scaler(part.inputs[train_idx], learned[train_idx])
        x = (part.inputs - scaler["in_mean"]) / scaler["in_std"]
        y = (learned - scaler["out_mean"]) / scaler["out_std"]
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        train_idx = torch.as_tensor(train_idx, device=device)

        net = _network(torch, self.hidden_size, self.num_layers).to(device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        history, best, best_state, stale = [], np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            order = train_idx[torch.randperm(len(train_idx), generator=generator)
                              .to(device)]
            # Summed on the device: a float() per batch would wait for the GPU.
            total = torch.zeros((), device=device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(net(x[batch]), y[batch])
                loss.backward()
                optimiser.step()
                total += loss.detach() * len(batch)
            total = float(total)
            val_loss = np.nan
            if val_idx is not None:
                net.eval()
                with torch.no_grad():
                    # In chunks: the whole validation set at once would not fit
                    # on a GPU with millions of samples.
                    sq = torch.zeros((), dtype=torch.float64, device=device)
                    for v in range(0, len(val_idx), _PREDICT_BATCH):
                        rows = val_idx[v:v + _PREDICT_BATCH]
                        sq += ((net(x[rows]) - y[rows]) ** 2).sum(dtype=torch.float64)
                    val_loss = float(sq) / y[val_idx[:1]].numel() / len(val_idx)
                if val_loss < best:
                    best, stale = val_loss, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    stale += 1
            history.append((epoch, total / len(order), val_loss))
            if val_idx is not None and stale >= self.patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._models[name] = (net, scaler)
        self._info[name] = {
            "samples": int(len(part)), "epochs": len(history),
            "best_validation_loss": float(best) if val_idx is not None else np.nan,
        }
        self.history_[name] = history

    # ---------------------------------------------------------- predict

    def predict(self, samples):
        """Absolute predicted positions, shape ``(m, steps, 2)``.

        Rows of a group with no network are NaN (C12).
        """
        self._check_fitted()
        torch = _torch()
        device = torch.device(self.device)
        out = np.full(samples.targets.shape, np.nan)
        keys = self._keys(samples)
        for name, (net, scaler) in self._models.items():
            rows = np.flatnonzero(keys == name)
            for start in range(0, rows.size, _PREDICT_BATCH):
                sel = rows[start:start + _PREDICT_BATCH]
                x = (samples.inputs[sel] - scaler["in_mean"]) / scaler["in_std"]
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
                with torch.no_grad():
                    z = net(x).cpu().numpy().astype(np.float64)
                z = z * scaler["out_std"] + scaler["out_mean"]
                if self.target == "deviation":  # T' + delta_hat
                    z = z + samples.inputs[sel][..., :2]
                out[sel] = z
        return out

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("this ClstmPredictor is not fitted: call fit first")

    # ---------------------------------------------------- save and load

    def save(self, path):
        """Write the configuration, the networks and their scalers to one file."""
        self._check_fitted()
        torch = _torch()
        payload = {
            "config": {name: getattr(self, name) for name in self._CONFIG},
            "dropped": list(self.dropped_groups_),
            "history": {k: [tuple(r) for r in v] for k, v in self.history_.items()},
            "info": self._info,
            "networks": {
                name: {
                    "state_dict": {k: v.detach().cpu()
                                   for k, v in net.state_dict().items()},
                    "scaler": {k: torch.as_tensor(v) for k, v in scaler.items()},
                }
                for name, (net, scaler) in self._models.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="auto"):
        """Read a predictor written by :meth:`save`, onto ``device``."""
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = {"target": "absolute", **payload["config"]}  # files before 2026-11-25
        self = cls(device=device, **config)
        for name, entry in payload["networks"].items():
            net = _network(torch, self.hidden_size, self.num_layers)
            net.load_state_dict(entry["state_dict"])
            net.to(torch.device(self.device)).eval()
            scaler = {k: v.numpy().astype(np.float64)
                      for k, v in entry["scaler"].items()}
            self._models[name] = (net, scaler)
        self._info = payload["info"]
        self.groups_ = sorted(self._models)
        self.dropped_groups_ = list(payload["dropped"])
        self.history_ = {k: [tuple(r) for r in v]
                         for k, v in payload["history"].items()}
        self._fitted = True
        return self

    # ----------------------------------------------------------- report

    def summary(self):
        """Per group: training samples, epochs run, best validation loss."""
        if not self._fitted:
            return "ClstmPredictor (not fitted)"
        kind = "CLSTM, one network per group" if self.context else "LSTM, one network"
        lines = [f"{kind}; learns the {self.target}; device {self.device}"]
        for name in self.groups_:
            info = self._info[name]
            best = info["best_validation_loss"]
            best = "none" if np.isnan(best) else f"{best:.5f}"
            lines.append(
                f"  {name}: {info['samples']} samples, {info['epochs']} epochs, "
                f"best validation loss {best}"
            )
        dropped = ", ".join(self.dropped_groups_) or "none"
        lines.append(f"dropped (fewer than {self.min_samples} samples): {dropped}")
        return "\n".join(lines)
