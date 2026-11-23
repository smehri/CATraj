"""Error metrics shared by the spatial and temporal stages.

Five measures are reported together because no one of them is sufficient:

``RMSE``   overall error, weighting large errors heavily
``MAE``    overall error, weighting all errors equally
``ME``     signed -- the only one that reveals systematic bias
``MaxAE``  the worst single point, which RMSE can hide
``R2``     variance explained

RMSE and MAE are both magnitude-only: a model that consistently under-estimates
wave height by 2 m scores identically to one scattering plus or minus 2 m
symmetrically.  ME separates them.  The gap between RMSE and MAE is itself
diagnostic -- a large gap means a few big errors, a small gap means they are
spread evenly.

Percentage metrics (MAPE, SMAPE) are deliberately absent: wind direction
crosses zero, so percentage errors explode.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ._circular import angular_residual

__all__ = ["Metrics", "residuals", "score"]


@dataclass(frozen=True)
class Metrics:
    """Error measures over one set of held-out predictions."""

    rmse: float
    mae: float
    me: float
    max_ae: float
    r2: float
    n: int

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self):
        return (
            f"RMSE={self.rmse:.4g}  MAE={self.mae:.4g}  ME={self.me:+.4g}  "
            f"Max|e|={self.max_ae:.4g}  R2={self.r2:.4g}  (n={self.n})"
        )


def residuals(predicted, observed, circular=False):
    """Prediction minus observation, angle-aware when ``circular``."""
    if circular:
        return angular_residual(predicted, observed)
    return np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)


def score(predicted, observed, circular=False):
    """Compute all five metrics, ignoring pairs where either value is missing.

    Returns ``Metrics`` with ``n`` counting the pairs actually scored.  If
    nothing is scorable every measure is NaN rather than raising, so a method
    that fails everywhere still produces a comparable row.
    """
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    keep = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[keep], obs[keep]

    n = int(pred.size)
    if n == 0:
        return Metrics(np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    err = residuals(pred, obs, circular=circular)

    # R^2 against the mean of the observations.  For a circular variable the
    # "mean" is the circular mean, so that total variance is measured on the
    # circle too rather than across the 0/360 seam.
    if circular:
        centre = np.degrees(
            np.arctan2(
                np.mean(np.sin(np.radians(obs))), np.mean(np.cos(np.radians(obs)))
            )
        )
        total = angular_residual(obs, centre)
    else:
        total = obs - obs.mean()

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum(total**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return Metrics(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        me=float(np.mean(err)),
        max_ae=float(np.max(np.abs(err))),
        r2=r2,
        n=n,
    )
