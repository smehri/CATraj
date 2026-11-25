"""Curvilinear (dead-reckoning) prediction -- [Mehri2021] section IV-C.

The first half of the hybrid model.  A vessel's next positions are predicted
from its speed and course alone; :mod:`.clstm` then learns the correction.

The method, as the paper states it.  With high-frequency AIS, the motion
between two messages is taken as linear: "both normal and tangential
accelerations can be set to zero".  Then::

    L_i  = v_i * dt_i                                  (8)
    dt_i = t_{i+1} - t_i                               (9)
    dx_i = L_i * sin(Phi_i),  dy_i = L_i * cos(Phi_i)  (10)
    (x'_{i+1}, y'_{i+1}) = (x_i, y_i) + (dx_i, dy_i)   (11)

and the time step of a trajectory is the mean interval between its vertices::

    dt = (sum of dt_i) / n                             (12)

The paper predicts three future positions, T' = [p'_{t+1}, p'_{t+2}, p'_{t+3}].

Decided here
------------

* **C5. Constant speed and course.**  T' holds ``v_t`` and ``Phi_t`` for all
  steps: ``p'_{t+j} = p_t + j * dt * v_t * (sin Phi_t, cos Phi_t)``.  The
  values at t-1 are not used: the paper sets both accelerations to zero, and
  those accelerations are the only way t-1 enters the model.
* **C8. Grid north is taken as north.**  The positions are projected (UTM)
  while the course is measured from true north.  Over the Brest area the
  meridian convergence is under 2 degrees; it is not corrected.  This is a
  limitation.
* **Units.**  ``speed`` is in position units per second and ``course`` in
  degrees clockwise from north.  :data:`KNOT` converts AIS knots to metres
  per second (C6, applied by :func:`catraj.prediction.clstm.make_samples`).
"""

from __future__ import annotations

import numpy as np

__all__ = ["KNOT", "mean_interval", "curvilinear_predict"]

KNOT = 1852 / 3600  # one knot, in metres per second


def mean_interval(t):
    """Mean interval between consecutive vertices, Eq. 12.

    Parameters
    ----------
    t : array-like
        Times in seconds, strictly increasing.

    Returns
    -------
    float
        ``(t[-1] - t[0]) / (n - 1)``.

    Raises
    ------
    ValueError
        With fewer than two times, or times that do not increase.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("mean_interval needs at least two times")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return float((t[-1] - t[0]) / (t.size - 1))


def curvilinear_predict(x, y, speed, course, dt, steps=3):
    """Predict the next ``steps`` positions at constant speed and course.

    Eqs. 8--11 with the accelerations at zero (C5).

    Parameters
    ----------
    x, y : array-like, length m
        Positions at t, in a projected CRS.
    speed : array-like, length m
        Speed in position units per second.
    course : array-like, length m
        Course in degrees clockwise from north.
    dt : float or array-like of length m
        Time step in seconds.
    steps : int
        Number of future positions.

    Returns
    -------
    px, py : ndarray, shape (m, steps)
        ``p'_{t+j}`` for j = 1..steps.

    Raises
    ------
    ValueError
        If the arrays differ in length or ``dt`` is negative.
    """
    x, y, speed, course = (
        np.asarray(a, dtype=float).reshape(-1) for a in (x, y, speed, course)
    )
    m = x.size
    if not (y.size == speed.size == course.size == m):
        raise ValueError("x, y, speed and course must have the same length")
    dt = np.asarray(dt, dtype=float)
    if dt.ndim == 0:
        dt = np.full(m, float(dt))
    elif dt.shape != (m,):
        raise ValueError("dt must be a scalar or have the same length as x")
    if np.any(dt < 0):
        raise ValueError("dt must not be negative")
    phi = np.radians(course)
    reach = np.arange(1, steps + 1)[None, :] * (dt * speed)[:, None]
    return (x[:, None] + reach * np.sin(phi)[:, None],
            y[:, None] + reach * np.cos(phi)[:, None])
