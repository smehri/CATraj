"""Choosing a spatial interpolation method -- leave-one-station-out.

Hold out one station, predict its value from the remaining ones, compare
against what it actually observed.  Repeat for every station, at every epoch.
This is the spatial analogue of the temporal stage's k-fold, with stations
rather than time as the folds.

Two things are reported that a bare error figure would hide:

* **unreachable predictions.**  A held-out station may fall outside the convex
  hull of the others -- corner and coastal stations often do -- and the
  triangulation methods cannot reach it.  Those cases are counted rather than
  quietly dropped, because a method that fails at edge stations is worse in a
  way its error alone does not show.
* **station count.**  Every fit uses one fewer station than exist.  With a
  sparse network that is thin for kriging, and the reader should know.

The methods themselves are in :mod:`.spatial`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._metrics import Metrics, score
from .spatial import SPATIAL_METHODS

__all__ = ["SpatialResult", "SpatialTable", "evaluate_spatial", "leave_one_station_out"]


@dataclass(frozen=True)
class SpatialResult:
    """One method's score at one epoch."""

    method: str
    epoch: object
    metrics: Metrics
    unreachable: int
    n_stations: int

    def __str__(self):
        tail = f"  unreachable={self.unreachable}" if self.unreachable else ""
        return f"{self.method:<14} {self.metrics}{tail}"


def _make(method, circular, **kwargs):
    if isinstance(method, str):
        try:
            cls = SPATIAL_METHODS[method]
        except KeyError:
            raise ValueError(
                f"unknown spatial method {method!r}; "
                f"choose from {sorted(SPATIAL_METHODS)}"
            ) from None
        return cls(circular=circular, **kwargs)
    return method


def leave_one_station_out(x, y, values, method, circular=False, **kwargs):
    """Score one method at one epoch.

    Returns ``(Metrics, unreachable, n_stations)``.  Needs at least three
    stations -- with two, dropping one leaves nothing to interpolate between.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[keep], y[keep], v[keep]
    m = v.size

    if m < 3:
        return score([], [], circular=circular), 0, m

    predicted = np.empty(m, dtype=float)
    for i in range(m):
        hold = np.ones(m, dtype=bool)
        hold[i] = False
        fit = _make(method, circular, **kwargs).fit(x[hold], y[hold], v[hold])
        predicted[i] = fit(x[i], y[i])

    unreachable = int(np.count_nonzero(~np.isfinite(predicted)))
    return score(predicted, v, circular=circular), unreachable, m


class SpatialTable:
    """Leave-one-station-out results for every method at every epoch.

    ``select`` reduces the table to a choice.  The default rule minimises RMSE;
    ``per_epoch`` decides whether that choice is made once or hour by hour.
    """

    def __init__(self, results, circular=False):
        self.results = list(results)
        self.circular = circular

    @property
    def methods(self):
        return sorted({r.method for r in self.results})

    @property
    def epochs(self):
        return sorted({r.epoch for r in self.results})

    def aggregate(self, rule="rmse"):
        """Mean of ``rule`` over all epochs, per method.  Lower is better."""
        out = {}
        for name in self.methods:
            vals = [
                getattr(r.metrics, rule)
                for r in self.results
                if r.method == name and np.isfinite(getattr(r.metrics, rule))
            ]
            out[name] = float(np.mean(vals)) if vals else np.inf
        return out

    def select(self, rule="rmse", per_epoch=False):
        """Choose a method.

        With ``per_epoch=False`` (the default) one method is chosen across all
        epochs: the surface keeps a consistent character through time and hours
        stay comparable.

        With ``per_epoch=True`` the winner is chosen independently at each
        epoch.  Spatial structure genuinely varies -- a calm field and a storm
        front are not best served by the same interpolator -- but the method
        can then switch between consecutive hours, and a value that moves
        because the *interpolator* changed rather than the weather is an
        artefact.  It also hands the temporal stage a series assembled from
        different estimators, whose extra roughness that stage will faithfully
        reproduce.
        """
        if rule == "r2":
            raise ValueError("select() minimises its rule; R2 is better when larger")

        if not per_epoch:
            agg = self.aggregate(rule)
            best = min(agg, key=agg.get)
            if not np.isfinite(agg[best]):
                raise RuntimeError("no method produced a usable score")
            return best

        chosen = {}
        for epoch in self.epochs:
            rows = [
                r
                for r in self.results
                if r.epoch == epoch and np.isfinite(getattr(r.metrics, rule))
            ]
            if rows:
                chosen[epoch] = min(rows, key=lambda r: getattr(r.metrics, rule)).method
        if not chosen:
            raise RuntimeError("no method produced a usable score at any epoch")
        return chosen

    def to_frame(self):
        """Results as a pandas DataFrame, one row per method and epoch."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "method": r.method,
                    "epoch": r.epoch,
                    "n_stations": r.n_stations,
                    "unreachable": r.unreachable,
                    **r.metrics.as_dict(),
                }
                for r in self.results
            ]
        )

    def summary(self, rule="rmse"):
        """One line per method, averaged over epochs -- for reading at a glance."""
        agg = self.aggregate(rule)
        unreach = {
            name: sum(r.unreachable for r in self.results if r.method == name)
            for name in self.methods
        }
        stations = {r.n_stations for r in self.results} or {0}
        lines = [
            f"leave-one-station-out over {len(self.epochs)} epoch(s), "
            f"{min(stations)}-{max(stations)} stations"
        ]
        for name in sorted(agg, key=agg.get):
            note = f"   unreachable={unreach[name]}" if unreach[name] else ""
            lines.append(f"  {name:<14} mean {rule}={agg[name]:.4g}{note}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"<SpatialTable {len(self.methods)} methods x "
            f"{len(self.epochs)} epochs>"
        )


def evaluate_spatial(
    x, y, values, epochs, methods=None, circular=False, method_kwargs=None
):
    """Score every candidate method at every epoch.

    Parameters
    ----------
    x, y : array-like
        Station positions in projected metres, repeated per epoch.
    values : array-like
        Observed context values.
    epochs : array-like
        Epoch label for each observation; rows sharing one are fitted together.
    methods : sequence, optional
        Names from ``SPATIAL_METHODS`` or interpolator instances.  Defaults to
        all of them.
    circular : bool
        Treat values as bearings in degrees.
    method_kwargs : dict, optional
        Extra keyword arguments per method name, e.g. ``{"idw": {"power": 3}}``.

    Returns
    -------
    SpatialTable
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    ep = np.asarray(epochs).ravel()
    if not (x.size == y.size == v.size == ep.size):
        raise ValueError("x, y, values and epochs must have equal length")

    names = list(methods) if methods is not None else list(SPATIAL_METHODS)
    kwargs = method_kwargs or {}

    results = []
    for epoch in sorted(set(ep.tolist())):
        sel = ep == epoch
        for name in names:
            label = name if isinstance(name, str) else type(name).__name__
            metrics, unreachable, n = leave_one_station_out(
                x[sel],
                y[sel],
                v[sel],
                name,
                circular=circular,
                **(kwargs.get(label, {}) if isinstance(name, str) else {}),
            )
            results.append(SpatialResult(label, epoch, metrics, unreachable, n))

    return SpatialTable(results, circular=circular)
