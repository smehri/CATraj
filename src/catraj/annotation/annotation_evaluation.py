"""Choosing a temporal interpolation method -- k-fold cross-validation.

Observations are partitioned randomly into k sets of roughly equal size.  Each
fold is held out in turn, the rest fits the model, the held-out fold is
predicted and scored; reported metrics are the mean over all k folds.

``k`` does double duty.  Training size grows with k, so sweeping k = 2..10
varies both the fold structure and the number of fitting points -- it selects a
method *and* a suitable number of fitting points.

**The first and last observation are always kept in training.**  Random
partitioning could otherwise hold one out, leaving the training set unable to
span it, and the model would extrapolate.  Context observations bracket the
trajectory by design, so extrapolation is out of scope and is prevented
structurally rather than merely scored.

Selection is deliberately a separate step: ``evaluate_temporal`` returns the
table, and the analyst calls :meth:`TemporalTable.select` on it.  The published
study shows the decision is not purely mechanical -- for mean wind speed, DLT
won on MAE while the spline won on RMSE, and the spline was chosen because RMSE
weights the large errors that matter.

The methods themselves are in :mod:`.interpolators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._metrics import Metrics, score
from .interpolators import DLT, PiecewisePolynomial, Spline

__all__ = [
    "TemporalResult",
    "TemporalTable",
    "evaluate_temporal",
    "k_fold",
    "default_candidates",
]


def default_candidates(degrees=range(2, 8)):
    """The paper's candidate set: DLT, piecewise polynomials 2-7, spline."""
    out = [("dlt", DLT, {})]
    out += [
        (f"pp_deg{d}", PiecewisePolynomial, {"degree": d}) for d in degrees
    ]
    out.append(("spline", Spline, {}))
    return out


@dataclass(frozen=True)
class TemporalResult:
    """One method's score at one fold count."""

    method: str
    k: int
    metrics: Metrics
    n_train: int

    def __str__(self):
        return f"{self.method:<12} k={self.k:<3} {self.metrics}"


def k_fold(t, values, factory, k, circular=False, seed=0, **kwargs):
    """Score one method at one fold count.

    Returns ``(Metrics, n_train)`` where ``n_train`` is the training size of
    the last fold -- the "number of fitting points" the paper reports.

    The endpoints are removed from the pool before partitioning and added back
    to every training set, so no fold ever asks a method to extrapolate.
    """
    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]

    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    m = t.size
    if m < 4 or k < 2:
        return score([], [], circular=circular), 0

    endpoints = np.array([0, m - 1])
    interior = np.arange(1, m - 1)
    if interior.size < k:
        return score([], [], circular=circular), 0

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(interior)
    folds = np.array_split(shuffled, k)

    per_fold, n_train = [], 0
    for fold in folds:
        if fold.size == 0:
            continue
        mask = np.ones(m, dtype=bool)
        mask[fold] = False
        mask[endpoints] = True  # endpoints never held out

        model = factory(circular=circular, **kwargs).fit(t[mask], v[mask])
        predicted = model(t[fold])
        per_fold.append(score(predicted, v[fold], circular=circular))
        n_train = int(mask.sum())

    if not per_fold:
        return score([], [], circular=circular), 0

    # Mean over all k folds, ignoring any fold that produced nothing scorable.
    def avg(attr):
        vals = [getattr(fm, attr) for fm in per_fold if np.isfinite(getattr(fm, attr))]
        return float(np.mean(vals)) if vals else np.nan

    return (
        Metrics(
            rmse=avg("rmse"),
            mae=avg("mae"),
            me=avg("me"),
            max_ae=avg("max_ae"),
            r2=avg("r2"),
            n=int(sum(fm.n for fm in per_fold)),
        ),
        n_train,
    )


class TemporalTable:
    """k-fold results for every method at every fold count.

    ``select`` reduces the table to a fitted, ready-to-use interpolator.
    """

    def __init__(self, results, t, values, circular=False, candidates=None):
        self.results = list(results)
        self._t = np.asarray(t, dtype=float)
        self._v = np.asarray(values, dtype=float)
        self.circular = circular
        self._candidates = {
            name: (factory, kwargs)
            for name, factory, kwargs in (candidates or default_candidates())
        }

    @property
    def methods(self):
        seen = []
        for r in self.results:
            if r.method not in seen:
                seen.append(r.method)
        return seen

    @property
    def ks(self):
        return sorted({r.k for r in self.results})

    def best_row(self, rule="rmse"):
        """The (method, k) row minimising ``rule``."""
        rows = [r for r in self.results if np.isfinite(getattr(r.metrics, rule))]
        if not rows:
            raise RuntimeError("no method produced a usable score")
        return min(rows, key=lambda r: getattr(r.metrics, rule))

    def select(self, rule="rmse", method=None):
        """Choose a method and return it **fitted to every observation**.

        The default rule minimises RMSE.  Pass ``method`` to override the
        choice -- when MAE and RMSE disagree, which method wins is a judgement
        about whether rare large errors matter more than typical ones, and that
        judgement belongs to the analyst, not to a default.
        """
        if rule == "r2":
            raise ValueError("select() minimises its rule; R2 is better when larger")

        name = method or self.best_row(rule).method
        try:
            factory, kwargs = self._candidates[name]
        except KeyError:
            raise ValueError(
                f"unknown method {name!r}; choose from {list(self._candidates)}"
            ) from None
        return factory(circular=self.circular, **kwargs).fit(self._t, self._v)

    def to_frame(self):
        """Results as a pandas DataFrame, one row per method and k."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "method": r.method,
                    "k": r.k,
                    "n_train": r.n_train,
                    **r.metrics.as_dict(),
                }
                for r in self.results
            ]
        )

    def summary(self, rule="rmse"):
        """Best fold count per method, ordered by score -- for reading quickly."""
        lines = [f"k-fold over k={min(self.ks)}..{max(self.ks)}, {self._t.size} observations"]
        best = {}
        for name in self.methods:
            rows = [
                r
                for r in self.results
                if r.method == name and np.isfinite(getattr(r.metrics, rule))
            ]
            if rows:
                best[name] = min(rows, key=lambda r: getattr(r.metrics, rule))
        for name, r in sorted(best.items(), key=lambda kv: getattr(kv[1].metrics, rule)):
            lines.append(
                f"  {name:<12} best k={r.k:<3} {rule}={getattr(r.metrics, rule):.4g}"
                f"   MAE={r.metrics.mae:.4g}  n_fit={r.n_train}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"<TemporalTable {len(self.methods)} methods x {len(self.ks)} k values>"


def evaluate_temporal(
    t, values, candidates=None, ks=range(2, 11), circular=False, seed=0
):
    """Score every candidate method at every fold count.

    Parameters
    ----------
    t : array-like
        Observation times -- UNIX seconds, or any consistent numeric scale.
    values : array-like
        Observed context values at one location.
    candidates : sequence of ``(name, factory, kwargs)``, optional
        Defaults to DLT, piecewise polynomials of degree 2 to 7, and the spline.
    ks : iterable of int
        Fold counts to sweep.  Default 2 to 10, following the paper.
    circular : bool
        Treat values as bearings in degrees.
    seed : int
        Seed for the random partition, so a run is reproducible.

    Returns
    -------
    TemporalTable
    """
    cands = list(candidates) if candidates is not None else default_candidates()
    results = []
    for name, factory, kwargs in cands:
        for k in ks:
            metrics, n_train = k_fold(
                t, values, factory, k, circular=circular, seed=seed, **kwargs
            )
            results.append(TemporalResult(name, int(k), metrics, n_train))
    return TemporalTable(results, t, values, circular=circular, candidates=cands)
