"""Context-Aware Douglas--Peucker (CADP) -- [Mehri2025] section 2.3.

Douglas--Peucker keeps the points that carry *spatial* information.  A point
can be spatially redundant -- it lies on a straight leg -- and still mark the
moment the sea got rough, the water shoaled, or the vessel slowed down.  CADP
keeps those points too.

The method, as Algorithm 2 states it and as confirmed with the author:

1. **Spatial.**  Simplify the raw trajectory with plain Douglas--Peucker at
   ``epsilon`` (:func:`.douglas_peucker`).
2. **Context.**  For every raw point, measure the *contextual displacement*
   that removing it causes, against the DP-simplified trajectory.  A point
   ``i`` lying between consecutive retained points ``a`` and ``b`` would, once
   removed, be represented by the context the simplified trajectory implies at
   its time -- linear interpolation between the two::

       c_hat_i = c_a + (c_b - c_a) * (t_i - t_a) / (t_b - t_a)
       e_i     = | c_i - c_hat_i |

   If ``e_i`` exceeds that variable's threshold, ``i`` is added.
3. **Union.**  The result is the DP points plus every point added in step 2.

This is the spatial error of Eq. 1 carried over to context: the projection of a
point onto the simplifying segment becomes the interpolation of its context
along that segment.  Interpolating in *time* is what Eq. 8's division by
``t_i - t_j`` is for -- a context change is judged against how long the
segment took, so irregular sampling does not distort it.  The error is in the
variable's own units, which is why the paper's thresholds read 0.15 m of wave
height, 1.5 degrees of wave direction and 80 m of depth.

Decisions where the paper leaves room (confirmed with the author, 2026-11-10):

* **Step 2 runs after DP, once.**  Every error is measured against the
  DP-simplified trajectory, not against one that already includes points
  added for context, and adding points never removes a DP point.  CADP's
  output is therefore always a superset of DP's, and its compression ratio
  never higher.  (Folding the context test into the DP recursion would be a
  different algorithm: an added point would split segments and change which
  spatial points DP keeps.)
* **Several variables, one rule.**  Each numeric context variable -- whether
  environmental (waves, depth, wind) or internal (speed, acceleration) -- has
  its own threshold and is tested on its own.  A point is added if *any*
  variable exceeds its threshold.  The domains differ in where the data come
  from, not in how they are tested.
* **Bearings** are differenced and interpolated on the circle: 350 to 10
  degrees is a 20 degree change through north.
* **Nominal (categorical) context** -- navigational status, activity -- cannot
  be interpolated.  For now both points either side of every change of
  category are kept, so the compressed trajectory still brackets the moment of
  the change.  A dedicated method for nominal context is future work.

Two modes for step 2
--------------------

``mode="post"`` (the default) is the method as published and described above:
one pass, every error against the DP trajectory, every point over its
threshold added at once.

``mode="iterative"`` is an alternative, added for comparison.  It repeats step
2 against the trajectory as it grows: each round, in every segment where some
variable exceeds its threshold, only the *worst* point is added -- the one
whose error is the largest multiple of its variable's threshold -- and errors
are measured again.  It stops when no point exceeds any threshold.  This is
Douglas--Peucker's own logic applied to context, within the DP segments.

The two differ most where context changes smoothly along a long DP segment.
The curve then bows away from the straight interpolation, and in ``post`` mode
every point on the bow that clears the threshold is kept, while
``iterative`` keeps the apex, re-measures, and usually finds the rest now
within tolerance.  ``iterative`` also *guarantees* the bound on the result:
every dropped point is within threshold of the interpolation between its
final neighbours.  ``post`` measures against the DP neighbours only, and once
points are added those are no longer the neighbours, so the final error is not
bounded by the threshold.

Edge cases, decided here:

* A missing context value at a point, or at either segment end, gives no
  error: the point is neither added nor excluded on that variable's account.
  DP and the other variables still decide.
* Points between two retained points that share one timestamp are compared
  with the midpoint of the two values (on the circle for bearings), since a
  time fraction is undefined there.

Filtering first (decided with the author, 2026-11-24)
-----------------------------------------------------

Raw AIS is noisy, and a noisy speed or position is compared against a
threshold as if it were the truth.  So :func:`compress`, the dataframe-level
step, first median-filters its inputs with :func:`catraj.preprocessing.denoise`:
the geometry and every numeric threshold column, over a centred time window of
``filter_window`` seconds, 30 by default; ``None`` switches it off and gives
the unfiltered result exactly.  Bearings listed in ``circular`` are filtered
through their unit vector; nominal columns are not filtered.  The compressed
rows carry the filtered geometry and the filtered columns: preprocess, then
compress.  Time comes from ``time_column`` or a DatetimeIndex; any other index
raises a ``ValueError``, since a window in seconds on row numbers would be
wrong.  The array-level :func:`cadp_select` is the published algorithm on the
arrays it is given and filters nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._time import times_of
from .._crs import _check_geographic
from ..annotation._circular import angular_residual
from ..preprocessing.filtering import denoise
from .._parallel import map_groups, parallel_map
from .douglas_peucker import douglas_peucker

__all__ = [
    "Context",
    "Selection",
    "context_errors",
    "nominal_boundaries",
    "cadp_select",
    "compress",
]


@dataclass(frozen=True)
class Context:
    """One numeric context variable and the threshold it is tested against.

    Parameters
    ----------
    values : array-like
        One value per trajectory point, in trajectory order.
    threshold : float
        Largest contextual displacement a dropped point may cause, in the
        variable's own units.  A point is added when its error is strictly
        greater.
    circular : bool
        Treat values as bearings in degrees.
    name : str, optional
        A label for reporting.
    """

    values: object
    threshold: float
    circular: bool = False
    name: str | None = None


@dataclass
class Selection:
    """Which points CADP keeps, and why.

    ``dp`` is the spatial stage's choice.  ``nominal`` maps each categorical
    variable to its boundary points.  ``context`` maps each numeric variable
    to the points it accounts for: in ``post`` mode every point over its
    threshold (before the union, so a point DP already kept may appear too);
    in ``iterative`` mode the points added on its account.  ``mask`` is the
    final answer.
    """

    dp: np.ndarray
    context: dict = field(default_factory=dict)
    nominal: dict = field(default_factory=dict)
    mode: str = "post"
    rounds: int = 0

    @property
    def mask(self):
        out = self.dp.copy()
        for flagged in (*self.context.values(), *self.nominal.values()):
            out |= flagged
        return out

    @property
    def added(self):
        """Points kept for context that DP alone would have dropped."""
        return self.mask & ~self.dp

    def summary(self):
        """Point counts, for reading at a glance."""
        n = int(self.dp.size)
        kept = int(self.mask.sum())
        rounds = f", {self.rounds} rounds" if self.mode == "iterative" else ""
        lines = [
            f"{n} points -> {kept} kept "
            f"({100.0 * (1 - kept / n) if n else 0.0:.2f}% compression; "
            f"{self.mode} mode{rounds})",
            f"  Douglas-Peucker      {int(self.dp.sum())}",
        ]
        for label, group in (("context", self.context), ("nominal", self.nominal)):
            for name, flagged in group.items():
                extra = int((flagged & ~self.dp).sum())
                lines.append(f"  + {label} {name:<12} {extra}")
        return "\n".join(lines)


def _check_times(t):
    t = np.asarray(t, dtype=float).ravel()
    if not np.isfinite(t).all():
        raise ValueError("timestamps must be finite")
    if np.any(np.diff(t) < 0):
        raise ValueError("timestamps must be in trajectory order (non-decreasing)")
    return t


def context_errors(t, values, keep, circular=False):
    """Contextual displacement caused by removing each point.

    Each point is measured against the retained points either side of it:
    the context those two imply at the point's time, by linear interpolation,
    compared with the context actually observed there.  Retained points have
    zero error.

    Parameters
    ----------
    t : array-like
        Timestamps in trajectory order.
    values : array-like
        The context variable, one value per point.
    keep : array-like of bool
        The retained points.  The first and last must be among them.
    circular : bool
        Treat values as bearings in degrees.

    Returns
    -------
    ndarray
        ``|c_i - c_hat_i|`` per point, in the variable's units; NaN where a
        value needed for it is missing.
    """
    t = _check_times(t)
    c = np.asarray(values, dtype=float).ravel()
    keep = np.asarray(keep, dtype=bool).ravel()
    if not (t.size == c.size == keep.size):
        raise ValueError("t, values and keep must have equal length")
    n = t.size
    if n == 0:
        return np.empty(0)
    if not (keep[0] and keep[-1]):
        raise ValueError("the first and last points must be retained")

    retained = np.flatnonzero(keep)
    slot = np.searchsorted(retained, np.arange(n), side="right") - 1
    a = retained[slot]
    b = retained[np.minimum(slot + 1, retained.size - 1)]

    span = t[b] - t[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(span > 0, (t - t[a]) / np.where(span > 0, span, 1.0), 0.5)

    if circular:
        c_hat = c[a] + angular_residual(c[b], c[a]) * frac
        err = np.abs(angular_residual(c, c_hat))
    else:
        c_hat = c[a] + (c[b] - c[a]) * frac
        err = np.abs(c - c_hat)

    err[keep] = 0.0
    return err


def nominal_boundaries(values):
    """Points either side of every change in a categorical context.

    Missing values form a category of their own, so a gap in the record is a
    change too.
    """
    codes, _ = pd.factorize(pd.Series(np.asarray(values, dtype=object).ravel()))
    flagged = np.zeros(codes.size, dtype=bool)
    if codes.size < 2:
        return flagged
    change = codes[1:] != codes[:-1]
    flagged[1:] |= change
    flagged[:-1] |= change
    return flagged


MODES = ("post", "iterative")


def _refine(t, contexts, names, keep):
    """``iterative`` mode: add each segment's worst point until none exceeds.

    Returns ``(keep, added_by, rounds)`` where ``added_by`` maps each variable
    name to the points added on its account.
    """
    keep = keep.copy()
    n = t.size
    added_by = {name: np.zeros(n, dtype=bool) for name in names}
    rounds = 0
    while True:
        worst = np.zeros(n)              # largest error / threshold so far
        culprit = np.full(n, -1)
        exceeds = np.zeros(n, dtype=bool)
        for k, ctx in enumerate(contexts):
            err = np.nan_to_num(
                context_errors(t, ctx.values, keep, circular=ctx.circular), nan=0.0
            )
            over = err > ctx.threshold
            exceeds |= over
            if ctx.threshold > 0:
                ratio = err / ctx.threshold
            else:
                ratio = np.where(err > 0, np.inf, 0.0)
            better = over & (ratio > worst)
            worst[better] = ratio[better]
            culprit[better] = k
        candidates = np.flatnonzero(exceeds)
        if candidates.size == 0:
            return keep, added_by, rounds
        rounds += 1

        # One point per segment: the candidate with the largest ratio.
        segment = (np.cumsum(keep) - 1)[candidates]
        order = np.lexsort((-worst[candidates], segment))
        seg_sorted = segment[order]
        first = np.r_[True, seg_sorted[1:] != seg_sorted[:-1]]
        chosen = candidates[order[first]]
        keep[chosen] = True
        for k, name in enumerate(names):
            added_by[name][chosen[culprit[chosen] == k]] = True


def cadp_select(x, y, t, epsilon, contexts=(), nominal=(), mode="post"):
    """Run CADP on arrays and report which points are kept, and why.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    t : array-like
        Timestamps in the same order -- UNIX seconds, or any consistent scale.
    epsilon : float
        The spatial threshold for Douglas--Peucker, in the positions' units.
    contexts : sequence of Context
        Numeric context variables, each with its own threshold.
    nominal : mapping or sequence
        Categorical context variables.  A mapping ``{name: values}``, or a
        sequence of value arrays.
    mode : {"post", "iterative"}
        ``post`` is the published method; ``iterative`` re-measures after
        each addition.  See the module docstring.

    Returns
    -------
    Selection
        ``selection.mask`` is the CADP result.  With no context given it is
        exactly Douglas--Peucker.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    x = np.asarray(x, dtype=float).ravel()
    t = _check_times(t)
    if x.size != t.size:
        raise ValueError("positions and timestamps must have equal length")

    dp = douglas_peucker(x, y, epsilon)
    selection = Selection(dp=dp, mode=mode)
    if x.size <= 2:
        return selection

    contexts = list(contexts)
    names = []
    for k, ctx in enumerate(contexts):
        if not np.isfinite(ctx.threshold) or ctx.threshold < 0:
            raise ValueError("context thresholds must be finite and non-negative")
        names.append(ctx.name or f"context_{k}")

    items = nominal.items() if isinstance(nominal, dict) else (
        (f"nominal_{k}", v) for k, v in enumerate(nominal)
    )
    for name, values in items:
        flagged = nominal_boundaries(values)
        if flagged.size != x.size:
            raise ValueError(f"nominal variable {name!r} has the wrong length")
        selection.nominal[name] = flagged

    if mode == "post":
        for name, ctx in zip(names, contexts):
            err = context_errors(t, ctx.values, dp, circular=ctx.circular)
            with np.errstate(invalid="ignore"):
                selection.context[name] = err > ctx.threshold  # NaN -> False
    else:
        # Nominal boundaries are fixed points of the result, so numeric
        # context is measured against the trajectory that already has them.
        base = dp.copy()
        for flagged in selection.nominal.values():
            base |= flagged
        _, selection.context, selection.rounds = _refine(t, contexts, names, base)

    return selection


# ------------------------------------------------------------ dataframes


def _select_frame(df, epsilon, thresholds, circular, nominal, time_column, mode):
    _check_geographic(df)
    t = (
        df[time_column].to_numpy(dtype=float)
        if time_column is not None
        else times_of(df)
    )
    geom = df.geometry
    contexts = [
        Context(df[col].to_numpy(dtype=float), thr, circular=col in circular, name=col)
        for col, thr in (thresholds or {}).items()
    ]
    nominal_values = {col: df[col].to_numpy() for col in nominal}
    return cadp_select(
        geom.x.to_numpy(), geom.y.to_numpy(), t, epsilon,
        contexts=contexts, nominal=nominal_values, mode=mode,
    )


def compress(
    traj,
    epsilon,
    thresholds=None,
    circular=(),
    nominal=(),
    time_column=None,
    by=None,
    mode="post",
    filter_window=30,
    n_jobs=1,
):
    """Compress trajectories with CADP.

    Context lives in ordinary columns -- written there by
    :func:`catraj.annotation.annotate`, or recorded with the positions, like
    speed -- and the result is a row subset of the input, so every column of
    every retained point survives.

    Parameters
    ----------
    traj : movingpandas.Trajectory, TrajectoryCollection or GeoDataFrame
        Positions in a projected CRS, in trajectory order.
    epsilon : float
        Spatial threshold for Douglas--Peucker, in the CRS's units (metres).
    thresholds : dict, optional
        ``{column: threshold}`` for each numeric context variable.  Omit for
        plain Douglas--Peucker.
    circular : iterable of str
        Which of those columns are bearings in degrees.
    nominal : iterable of str
        Categorical context columns; both points either side of each change
        are kept.
    time_column : str, optional
        Numeric time column (UNIX seconds).  Defaults to the index, which
        MovingPandas keeps as a DatetimeIndex.
    by : str, optional
        For a GeoDataFrame holding several trajectories: the column naming
        each one.  Each is compressed on its own.
    mode : {"post", "iterative"}
        ``post`` is the published method; ``iterative`` re-measures after
        each addition.  See the module docstring.
    filter_window : float or None
        Width, in seconds (the unit of the times), of the time-windowed median
        applied before selection: ``30`` by default, ``None`` for no
        filtering.  See "Filtering first" in the module docstring.
    n_jobs : int or None
        Processes over which the trajectories are shared out: ``1`` (the
        default) for none, ``-1`` for every logical CPU.  Each trajectory is
        still compressed on its own, so the result is the same.  See
        :func:`catraj.parallel_map`.

    Returns
    -------
    The same kind of object, holding only the retained points; with
    filtering on, their geometry and threshold columns are the filtered ones.
    """
    circular = set(circular)
    nominal = list(nominal)
    unknown = circular - set(thresholds or {})
    if unknown:
        raise ValueError(f"circular columns without a threshold: {sorted(unknown)}")
    args = (epsilon, thresholds, circular, nominal, time_column, mode)

    def one(df):
        if filter_window is not None:
            df = denoise(
                df, filter_window, columns=list(thresholds or {}),
                circular=circular, time_column=time_column,
            )
        return df.iloc[_select_frame(df, *args).mask]

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        import movingpandas as mpd

        def again(tr):
            return compress(tr, epsilon, thresholds, circular, nominal, time_column,
                            mode=mode, filter_window=filter_window)

        return mpd.TrajectoryCollection(parallel_map(
            again, traj.trajectories, n_jobs, [len(tr.df) for tr in traj.trajectories]))

    if hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        import movingpandas as mpd

        return mpd.Trajectory(one(traj.df).copy(), traj.id)

    if by is not None:
        parts = map_groups(one, traj, by, n_jobs)
        return pd.concat(parts) if parts else traj.iloc[:0]

    return one(traj).copy()
