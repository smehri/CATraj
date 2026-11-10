"""Shared fixtures.

Everything here is synthetic: a field whose true value is known everywhere, so
an interpolator's answer can be checked exactly rather than eyeballed.  Tests
that need the real sample data carry the ``data`` marker and skip when
``datasets/`` is absent.
"""

import os

# Must precede the first NumPy import; see catraj/__init__.py.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


def linear_field(x, y, t=0.0):
    """A plane drifting linearly in time: every method should reproduce it."""
    return 2.0 + 0.001 * x - 0.0005 * y + 0.01 * t


@pytest.fixture
def grid_stations():
    """A 4x4 station grid, 10 km apart, in projected metres."""
    xs, ys = np.meshgrid(np.arange(4) * 10_000.0, np.arange(4) * 10_000.0)
    return xs.ravel(), ys.ravel()


@pytest.fixture
def station_frame(grid_stations):
    """Hourly observations of :func:`linear_field` at every grid station."""
    x, y = grid_stations
    epochs = np.arange(6) * 3600.0
    rows = [
        pd.DataFrame({"x": x, "y": y, "t": e, "value": linear_field(x, y, e)})
        for e in epochs
    ]
    return pd.concat(rows, ignore_index=True)


def pytest_collection_modifyitems(config, items):
    if DATASETS.exists():
        return
    skip = pytest.mark.skip(reason="sample datasets/ not present")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
