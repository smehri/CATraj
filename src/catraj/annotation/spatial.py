"""Spatial interpolation of context -- stage 1 of annotation.

Context is measured at scattered stations, never where the moving object is.
For each epoch these estimators interpolate the station values **directly to
the query position**; no raster is built.  A gridded surface is available from
:func:`to_surface` for viewing and export, but it is not the annotation path.

Interpolation runs in projected metres.  Degrees are not a metric space -- at
48 N a degree of longitude spans about 74 km against 111 km for a degree of
latitude -- so distance-weighted methods biased by that anisotropy would weight
east-west neighbours wrongly.

Bearings are interpolated on the unit circle (see :mod:`._circular`).  The
caller declares which variables are circular; nothing is inferred from a name.

Bearings are handled in :mod:`._circular`; methods are compared in
:mod:`.spatial_evaluation`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)
from scipy.spatial import cKDTree

from ._circular import from_components, to_components

__all__ = [
    "SpatialInterpolator",
    "Nearest",
    "IDW",
    "TIN",
    "CloughTocher",
    "RBF",
    "Kriging",
    "SPATIAL_METHODS",
    "to_surface",
]


class SpatialInterpolator(ABC):
    """Estimate a context variable at arbitrary positions from station values.

    Subclasses implement :meth:`_fit_component` and :meth:`_eval_component` for
    a single scalar field; circular handling, missing-value filtering and the
    public API live here so every method behaves identically.

    Parameters
    ----------
    circular : bool
        Treat values as bearings in degrees.  The caller decides; a direction
        column is not detected by name.
    """

    #: Whether the method can produce values outside the convex hull of the
    #: stations.  Triangulation-based methods cannot.
    extrapolates = True

    def __init__(self, circular=False):
        self.circular = bool(circular)
        self._fitted = False

    # -- to implement in subclasses ------------------------------------

    @abstractmethod
    def _fit_component(self, xy, values):
        """Return an opaque state for one scalar field."""

    @abstractmethod
    def _eval_component(self, state, xy):
        """Evaluate a fitted component at ``xy``; NaN where unreachable."""

    # -- public API ----------------------------------------------------

    def fit(self, x, y, values):
        """Fit to station observations at one epoch.

        Stations whose value or position is missing are dropped.  Fitting with
        no usable station is allowed -- evaluation then returns NaN -- so one
        empty epoch does not abort a whole run.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if not (x.size == y.size == v.size):
            raise ValueError(
                f"x, y and values must have equal length; "
                f"got {x.size}, {y.size}, {v.size}"
            )

        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        xy = np.column_stack([x[keep], y[keep]])
        v = v[keep]

        self.n_stations = int(v.size)
        self._empty = self.n_stations == 0

        if not self._empty:
            if self.circular:
                u, w = to_components(v)
                self._state = (
                    self._fit_component(xy, u),
                    self._fit_component(xy, w),
                )
            else:
                self._state = self._fit_component(xy, v)

        self._fitted = True
        return self

    def __call__(self, x, y):
        """Estimate at the given positions.  Unreachable points return NaN."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        scalar = x.ndim == 0
        xy = np.column_stack([np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()])

        if self._empty:
            out = np.full(len(xy), np.nan)
        elif self.circular:
            su, sw = self._state
            out = from_components(
                self._eval_component(su, xy), self._eval_component(sw, xy)
            )
        else:
            out = self._eval_component(self._state, xy)

        out = np.asarray(out, dtype=float)
        return float(out[0]) if scalar else out.reshape(np.atleast_1d(x).shape)

    def __repr__(self):
        return f"{type(self).__name__}(circular={self.circular})"


class Nearest(SpatialInterpolator):
    """Nearest-neighbour. Piecewise constant, discontinuous -- an error floor."""

    def _fit_component(self, xy, values):
        return NearestNDInterpolator(xy, values)

    def _eval_component(self, state, xy):
        return state(xy)


class IDW(SpatialInterpolator):
    """Inverse distance weighting, ``w_i = 1 / d_i**power``.

    Exact at the stations: a query landing on one returns that station's value
    rather than dividing by zero.  ``k`` limits the fit to the nearest few
    stations, which keeps distant, weakly related observations from dragging
    the estimate.
    """

    def __init__(self, power=2.0, k=None, circular=False):
        super().__init__(circular=circular)
        self.power = float(power)
        self.k = k

    def _fit_component(self, xy, values):
        return cKDTree(xy), np.asarray(values, dtype=float)

    def _eval_component(self, state, xy):
        tree, values = state
        k = min(self.k or len(values), len(values))
        dist, idx = tree.query(xy, k=k)
        dist = np.atleast_2d(dist.T).T if k > 1 else dist[:, None]
        idx = np.atleast_2d(idx.T).T if k > 1 else idx[:, None]

        out = np.empty(len(xy), dtype=float)
        exact = dist[:, 0] == 0.0
        out[exact] = values[idx[exact, 0]]

        far = ~exact
        if np.any(far):
            w = 1.0 / dist[far] ** self.power
            out[far] = np.sum(w * values[idx[far]], axis=1) / np.sum(w, axis=1)
        return out

    def __repr__(self):
        return f"IDW(power={self.power:g}, k={self.k}, circular={self.circular})"


_QHULL_OK = None

_QHULL_PROBE = (
    "import numpy as np;"
    "from scipy.interpolate import LinearNDInterpolator as L;"
    "f=L(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),np.array([0.,1.,1.,2.]));"
    "assert np.isfinite(f(np.array([[.25,.25]]))).all()"
)


def _qhull_is_usable():
    """Whether SciPy's Qhull bindings work in this build.

    Some SciPy/NumPy combinations ship a Qhull extension that *aborts the
    interpreter* -- a native fault, not an exception -- the first time a
    triangulation is queried.  That cannot be caught in-process, so the probe
    runs in a subprocess: if it dies, this process survives and the
    triangulation methods simply report themselves unavailable.  The result is
    cached, so the cost is paid once.

    Set ``CATRAJ_ASSUME_QHULL=1`` to skip the probe (or ``=0`` to force the
    methods off) if the subprocess launch is unwanted.
    """
    global _QHULL_OK
    if _QHULL_OK is None:
        override = os.environ.get("CATRAJ_ASSUME_QHULL")
        if override is not None:
            _QHULL_OK = override not in ("0", "false", "False", "")
            return _QHULL_OK
        try:
            _QHULL_OK = (
                subprocess.run(
                    [sys.executable, "-c", _QHULL_PROBE],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                ).returncode
                == 0
            )
        except Exception:
            _QHULL_OK = False
        if not _QHULL_OK:
            warnings.warn(
                "SciPy's Qhull bindings are unusable in this environment, so the "
                "TIN and CloughTocher spatial methods are disabled and return "
                "NaN. The other methods are unaffected. Reinstalling SciPy "
                "usually fixes it (conda install -c conda-forge scipy=1.16).",
                RuntimeWarning,
                stacklevel=3,
            )
    return _QHULL_OK


class _TriangulationInterpolator(SpatialInterpolator):
    """Shared base for the Delaunay-based methods.

    Returns NaN outside the convex hull of the stations -- there is no basis
    for a value there, and inventing one would be worse than admitting the gap.
    """

    extrapolates = False
    _interpolator = None

    def _fit_component(self, xy, values):
        if len(values) < 3 or not _qhull_is_usable():
            return None
        return type(self)._interpolator(xy, values)

    def _eval_component(self, state, xy):
        if state is None:
            return np.full(len(xy), np.nan)
        return state(xy)


class TIN(_TriangulationInterpolator):
    """Linear interpolation on a Delaunay triangulation."""

    _interpolator = staticmethod(LinearNDInterpolator)


class CloughTocher(_TriangulationInterpolator):
    """Cubic C1 interpolation on a Delaunay triangulation.

    Smoother than :class:`TIN`; the same convex-hull limit applies.
    """

    _interpolator = staticmethod(CloughTocher2DInterpolator)


class RBF(SpatialInterpolator):
    """Radial basis function interpolation, thin-plate spline by default.

    Smooth, and unlike the triangulation methods it produces values beyond the
    station hull -- treat those with caution, as they are extrapolation.
    """

    def __init__(self, kernel="thin_plate_spline", smoothing=0.0, circular=False):
        super().__init__(circular=circular)
        self.kernel = kernel
        self.smoothing = float(smoothing)

    def _fit_component(self, xy, values):
        # A thin-plate spline needs enough points to resolve its polynomial
        # tail; fall back to a kernel with no such requirement when short.
        kernel = self.kernel
        if len(values) < 3 and kernel == "thin_plate_spline":
            kernel = "linear"
        return RBFInterpolator(
            xy, values, kernel=kernel, smoothing=self.smoothing
        )

    def _eval_component(self, state, xy):
        return state(xy)

    def __repr__(self):
        return f"RBF(kernel={self.kernel!r}, circular={self.circular})"


class Kriging(SpatialInterpolator):
    """Gaussian-process regression -- kriging in its machine-learning form.

    Unlike the deterministic methods this also yields a variance, so the
    uncertainty of each estimate is available via :meth:`predict_std`.

    ``nugget`` is the assumed observation noise.  The default, 1e-8, is
    effectively zero -- just enough to keep the covariance matrix invertible
    -- so the fit passes through the station values to within rounding.
    """

    def __init__(self, length_scale=None, nugget=1e-8, circular=False):
        super().__init__(circular=circular)
        self.length_scale = length_scale
        self.nugget = float(nugget)

    def _fit_component(self, xy, values):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        # Scale distances so the optimiser starts somewhere sane: station
        # spacing in metres is many orders away from sklearn's defaults.
        span = float(np.ptp(xy, axis=0).max()) if len(xy) > 1 else 1.0
        ls = self.length_scale or max(span / 3.0, 1.0)
        centre = float(np.mean(values))

        kernel = ConstantKernel(
            max(float(np.var(values)), 1e-8), (1e-10, 1e10)
        ) * Matern(length_scale=ls, length_scale_bounds=(ls * 1e-3, ls * 1e3), nu=1.5)

        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.nugget, normalize_y=False, n_restarts_optimizer=0
        )
        gp.fit(xy, values - centre)
        return gp, centre

    def _eval_component(self, state, xy):
        gp, centre = state
        return gp.predict(xy) + centre

    def predict_std(self, x, y):
        """Standard deviation of the estimate at each position.

        Not defined for circular variables, whose uncertainty lives on the
        circle rather than on a scalar axis.
        """
        if not self._fitted:
            raise RuntimeError("Kriging must be fitted before use")
        if self.circular:
            raise NotImplementedError(
                "predict_std is not defined for circular variables"
            )
        xy = np.column_stack(
            [np.atleast_1d(np.asarray(x, float)).ravel(),
             np.atleast_1d(np.asarray(y, float)).ravel()]
        )
        if self._empty:
            return np.full(len(xy), np.nan)
        gp, _ = self._state
        return gp.predict(xy, return_std=True)[1]


#: Candidate methods, in the order they are reported.
SPATIAL_METHODS = {
    "nearest": Nearest,
    "idw": IDW,
    "tin": TIN,
    "clough_tocher": CloughTocher,
    "rbf": RBF,
    "kriging": Kriging,
}


def to_surface(interpolator, bounds, resolution):
    """Evaluate a fitted interpolator over a regular grid.

    For visualisation, export and sanity-checking -- annotation samples at
    points and never needs this.

    Parameters
    ----------
    interpolator : SpatialInterpolator
        Already fitted to one epoch.
    bounds : tuple
        ``(x_min, y_min, x_max, y_max)`` in the stations' CRS.
    resolution : float
        Cell size, in the same units.

    Returns
    -------
    (values, x_coords, y_coords)
        ``values`` has shape ``(len(y_coords), len(x_coords))``, row 0 at
        ``y_min``.  Cells the method cannot reach are NaN.
    """
    x_min, y_min, x_max, y_max = map(float, bounds)
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounds must be (x_min, y_min, x_max, y_max)")

    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    gx, gy = np.meshgrid(xs, ys)
    return interpolator(gx, gy), xs, ys
