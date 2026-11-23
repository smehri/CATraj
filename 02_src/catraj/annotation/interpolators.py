"""Temporal interpolation of context -- stage 2 of annotation.

Context is observed far less often than position, so most trajectory points
fall between two context epochs with no value of their own.  These estimators
reconstruct the variable's behaviour through time at one location, so a value
can be read off at any timestamp.

Three candidates, following [Mehri2023] section 4.1:

``DLT``                  a straight line between consecutive epochs (Eq. 2)
``PiecewisePolynomial``  contiguous least-squares windows (Eq. 3)
``Spline``               one cubic curve through every observation (Eq. 4)

Which performs best is an empirical question -- see
:mod:`.annotation_evaluation`.

A fitted interpolator belongs to the context variable and location, not to a
trajectory: fit it once, then annotate every object that crosses that area in
that period.

Bearings are interpolated on the unit circle (see :mod:`._circular`).  The
caller declares which variables are circular; nothing is inferred from a name.

Methods are compared in :mod:`.annotation_evaluation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline

from ._circular import from_components, to_components

__all__ = [
    "TemporalInterpolator",
    "DLT",
    "PiecewisePolynomial",
    "Spline",
    "TEMPORAL_METHODS",
]


class TemporalInterpolator(ABC):
    """Estimate a context variable at arbitrary times from epoch observations.

    Subclasses implement :meth:`_fit_component` and :meth:`_eval_component` for
    a single scalar series; circular handling, sorting, missing-value filtering
    and the public API live here so every method behaves identically.

    Parameters
    ----------
    circular : bool
        Treat values as bearings in degrees.  The caller decides.
    """

    def __init__(self, circular=False):
        self.circular = bool(circular)
        self._fitted = False

    # -- to implement in subclasses ------------------------------------

    @abstractmethod
    def _fit_component(self, t, values):
        """Return an opaque state for one scalar series."""

    @abstractmethod
    def _eval_component(self, state, t):
        """Evaluate a fitted component at times ``t``."""

    # -- public API ----------------------------------------------------

    def fit(self, t, values):
        """Fit to a context series at one location.

        Observations are sorted by time; those with a missing time or value are
        dropped.  Duplicate timestamps are collapsed to their mean, since an
        interpolator cannot pass through two values at one instant.
        """
        t = np.asarray(t, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if t.size != v.size:
            raise ValueError(
                f"t and values must have equal length; got {t.size}, {v.size}"
            )

        keep = np.isfinite(t) & np.isfinite(v)
        t, v = t[keep], v[keep]

        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]

        if t.size and np.any(np.diff(t) == 0):
            t, v = self._collapse_duplicates(t, v, self.circular)

        self.t_min = float(t[0]) if t.size else np.nan
        self.t_max = float(t[-1]) if t.size else np.nan
        self.n_observations = int(t.size)
        self._empty = t.size < self._min_observations

        if not self._empty:
            if self.circular:
                u, w = to_components(v)
                self._state = (
                    self._fit_component(t, u),
                    self._fit_component(t, w),
                )
            else:
                self._state = self._fit_component(t, v)

        self._fitted = True
        return self

    #: Fewest observations the method can be fitted to.
    _min_observations = 2

    @staticmethod
    def _collapse_duplicates(t, v, circular):
        uniq = np.unique(t)
        out = np.empty(uniq.size, dtype=float)
        for i, ti in enumerate(uniq):
            vals = v[t == ti]
            if circular:
                out[i] = np.degrees(
                    np.arctan2(
                        np.mean(np.sin(np.radians(vals))),
                        np.mean(np.cos(np.radians(vals))),
                    )
                ) % 360.0
            else:
                out[i] = float(np.mean(vals))
        return uniq, out

    def __call__(self, t):
        """Estimate at the given times."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")

        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        flat = np.atleast_1d(t).ravel().astype(float)

        if self._empty:
            out = np.full(flat.size, np.nan)
        elif self.circular:
            su, sw = self._state
            out = from_components(
                self._eval_component(su, flat), self._eval_component(sw, flat)
            )
        else:
            out = np.asarray(self._eval_component(self._state, flat), dtype=float)

        out = np.asarray(out, dtype=float)
        return float(out[0]) if scalar else out.reshape(np.atleast_1d(t).shape)

    def __repr__(self):
        return f"{type(self).__name__}(circular={self.circular})"


class DLT(TemporalInterpolator):
    """Dynamic Linear Trajectory annotation -- Eq. 2.

    A straight line between each consecutive pair of observed values::

        v(t) = m * t + b,    m = (v2 - v1) / (t2 - t1),   b = v1 - m * t1

    Its assumption is that the variable changes linearly between epochs.  The
    paper notes this "does not seem valid, especially in the case of low
    temporal measurements resolution" -- which motivates the two nonlinear
    methods.  Exact at the observations.
    """

    def _fit_component(self, t, values):
        return t, np.asarray(values, dtype=float)

    def _eval_component(self, state, t):
        tt, vv = state
        # np.interp holds the end values beyond the range rather than shooting
        # off; annotation never asks outside it, and holding is the safer edge.
        return np.interp(t, tt, vv)


class PiecewisePolynomial(TemporalInterpolator):
    """Piecewise polynomial over contiguous windows -- Eq. 3.

    Eq. 3 partitions the interval into subintervals that tile the domain, with
    one sub-polynomial fitted on each.  Windows are therefore contiguous, not
    overlapping, and consecutive windows share a single breakpoint::

        window = degree + 2      observations per sub-polynomial
        step   = window - 1      so neighbours share one breakpoint

    A degree-*d* polynomial has *d+1* coefficients, so a window of exactly
    *d+1* points would be forced through every one of them -- zero residual,
    zero degrees of freedom, and every degree scoring identically.
    ``degree + 2`` is the smallest window that leaves the fit any freedom, so
    **each sub-polynomial is a least-squares fit and the curve passes near its
    observations rather than through them**.  That is what separates this
    method from :class:`Spline`.

    No continuity is imposed between subintervals: at a shared breakpoint the
    two neighbouring fits are anchored by the same observation but computed
    independently, so they agree closely without agreeing exactly.  In practice
    that shows as a change of slope, not a step.
    """

    def __init__(self, degree=2, window=None, step=None, circular=False):
        super().__init__(circular=circular)
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = int(degree)
        self.window = int(window) if window is not None else self.degree + 2
        if self.window < self.degree + 1:
            raise ValueError(
                f"window ({self.window}) must hold at least degree + 1 "
                f"({self.degree + 1}) observations"
            )
        self.step = int(step) if step is not None else self.window - 1
        if self.step < 1:
            raise ValueError("step must be at least 1")

    @property
    def _min_observations(self):
        return self.degree + 1

    def _fit_component(self, t, values):
        """Fit one sub-polynomial per window; keep each window's time span.

        Each window is fitted in its own coordinate, ``(t - lo) / (hi - lo)``,
        which runs from 0 to 1 across the window.  Raw UNIX seconds raised to
        the 7th power overflow the fit outright, and even times measured from
        the start of the series grow too large over a long record -- months
        of hourly data put the last windows ~1e7 s out, and the Vandermonde
        matrix becomes ill-conditioned.  The rescaling is affine, so each
        piece is the same least-squares polynomial; only its arithmetic is
        better behaved.
        """
        n = t.size
        pieces = []
        start = 0
        while start < n - 1:
            stop = min(start + self.window, n)
            ts, vs = t[start:stop], values[start:stop]
            lo, hi = float(ts[0]), float(ts[-1])
            scale = hi - lo if hi > lo else 1.0
            if ts.size >= self.degree + 1:
                coeffs = np.polyfit((ts - lo) / scale, vs, self.degree)
            elif ts.size >= 2:  # tail shorter than a window: drop the degree
                coeffs = np.polyfit((ts - lo) / scale, vs, ts.size - 1)
            else:
                break
            pieces.append((lo, hi, scale, coeffs))
            if stop >= n:
                break
            start += self.step
        return pieces

    def _eval_component(self, state, t):
        pieces = state
        out = np.full(t.size, np.nan)
        if not pieces:
            return out
        for lo, hi, scale, coeffs in pieces:
            # Later windows overwrite the shared breakpoint; the boundary
            # belongs to one piece, so p(x) stays single-valued.
            sel = (t >= lo) & (t <= hi)
            if np.any(sel):
                out[sel] = np.polyval(coeffs, (t[sel] - lo) / scale)
        # Outside the observed span, hold the nearest piece.
        first_lo, _, first_scale, first_c = pieces[0]
        last_lo, last_hi, last_scale, last_c = pieces[-1]
        below, above = t < first_lo, t > last_hi
        if np.any(below):
            out[below] = np.polyval(first_c, 0.0)
        if np.any(above):
            out[above] = np.polyval(last_c, (last_hi - last_lo) / last_scale)
        return out

    def __repr__(self):
        return (
            f"PiecewisePolynomial(degree={self.degree}, window={self.window}, "
            f"step={self.step}, circular={self.circular})"
        )


class Spline(TemporalInterpolator):
    """Interpolating cubic spline -- Eq. 4.

    A piecewise polynomial of degree *n* with *n-1* continuous derivatives;
    requiring the first and second derivatives to be continuous forces degree
    at least three.

    Unlike :class:`PiecewisePolynomial` this is **interpolating**: it passes
    through every observation exactly, and matches slope and curvature at every
    knot.  Boundary condition is not-a-knot.
    """

    _min_observations = 4  # not-a-knot needs four points

    def __init__(self, bc_type="not-a-knot", circular=False):
        super().__init__(circular=circular)
        self.bc_type = bc_type

    def _fit_component(self, t, values):
        bc = self.bc_type
        if t.size < 4 and bc == "not-a-knot":
            bc = "natural"
        return CubicSpline(t, values, bc_type=bc, extrapolate=True)

    def _eval_component(self, state, t):
        return state(t)

    def __repr__(self):
        return f"Spline(bc_type={self.bc_type!r}, circular={self.circular})"


#: Candidate methods, in the order they are reported.  Piecewise polynomials
#: are swept over degrees 2 to 7 by the evaluation stage.
TEMPORAL_METHODS = {
    "dlt": DLT,
    "piecewise_polynomial": PiecewisePolynomial,
    "spline": Spline,
}
