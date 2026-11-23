"""Time handling shared by every algorithm in the package."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["times_of"]


def times_of(df):
    """Numeric timestamps, in UNIX seconds, from a trajectory dataframe.

    MovingPandas keeps time in a DatetimeIndex; context epochs are usually
    UNIX seconds, so the index is converted to match.  A non-datetime index is
    taken as already numeric.

    The conversion goes through a Timedelta rather than the raw integer
    representation: pandas 3 infers the index's resolution (``s``, ``ms``,
    ``us`` or ``ns``), so its integers are no longer always nanoseconds, and
    dividing by 1e9 regardless would put every timestamp off by up to nine
    orders of magnitude.  Timezone-aware indexes are taken in UTC.
    """
    index = df.index
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        elapsed = index - pd.Timestamp("1970-01-01")
        return (elapsed / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    return np.asarray(index, dtype=float)
