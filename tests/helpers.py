"""Plain helpers shared by the tests.

The tests live in ``tests/``, which -- starting with a digit -- cannot be a
Python package, so they import these by name (``from helpers import ...``)
rather than relatively. pytest puts this folder on ``sys.path``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "preprocessed"


def linear_field(x, y, t=0.0):
    """A plane drifting linearly in time: every method should reproduce it."""
    return 2.0 + 0.001 * x - 0.0005 * y + 0.01 * t
