"""Circular (bearing) helpers.

Bearings live on a circle: 0 degrees and 360 degrees are the same direction.
Interpolating them as ordinary numbers is wrong -- 350 followed by 10 passes
through 180, the opposite heading, instead of taking the 20 degree arc.

Every circular quantity in this package is handled the same way: decompose the
bearing into a unit vector, interpolate the two components independently, then
recover the angle with ``atan2``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_components",
    "from_components",
    "angular_residual",
    "decode_compass",
    "COMPASS_STEP",
]

#: Degrees per point on a 16-point compass rose.
COMPASS_STEP = 22.5


def to_components(degrees):
    """Split bearings into ``(cos, sin)`` unit-vector components.

    NaN bearings propagate as NaN in both components.
    """
    theta = np.radians(np.asarray(degrees, dtype=float))
    return np.cos(theta), np.sin(theta)


def from_components(u, v):
    """Recover bearings in ``[0, 360)`` from unit-vector components.

    The vector length is not required to be one: interpolation shortens it
    wherever neighbouring bearings disagree, which is itself a signal that the
    direction is poorly determined there.  Only the angle is returned.
    """
    deg = np.degrees(np.arctan2(np.asarray(v, dtype=float), np.asarray(u, dtype=float)))
    return np.mod(deg, 360.0)


def angular_residual(predicted, observed):
    """Signed angular difference wrapped into ``[-180, 180]``.

    So a 359 -> 1 degree error reads as 2 degrees, not 358.
    """
    diff = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return (diff + 180.0) % 360.0 - 180.0


def decode_compass(codes, step=COMPASS_STEP, north=1):
    """Decode compass-point codes to degrees.

    Direction is often recorded as a code on a 16-point rose rather than in
    degrees.  Interpolating the codes is meaningless -- they are categorical
    labels, and a step from code 5 to code 13 is not a rotation -- so they must
    be decoded first.

    ``degrees = (code - north) * step``.  The default is the table of the
    Zenodo archive (``E2_Weather_Conditions/table_windDirection.csv``, record
    1167595): codes 1 to 16 are N, NNE, ..., NNW, from 0 to 337.5 degrees, so
    code 1 is north; 17 is "calm", 18 "variable" and 19 "no data".  Those are
    not directions.  Pass ``north=0`` for a rose whose first code is 0.

    A code outside ``north .. north + 360 / step - 1`` is off the rose and is
    returned as NaN, as is a missing code.
    """
    code = np.asarray(codes, dtype=float)
    last = north + round(360.0 / step) - 1
    on_rose = (code >= north) & (code <= last)
    return np.where(on_rose, (code - north) * step, np.nan)
