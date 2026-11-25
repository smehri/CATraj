"""catraj -- context-aware trajectory processing.

Reference implementations of the context-aware annotation, compression and
prediction algorithms developed in the doctoral research of Saeed Mehri,
compatible with MovingPandas.
"""

__version__ = "0.1.0"

# Some Intel MKL builds (2026.1.0 among them) abort the process with a stack
# buffer overrun the first time a LAPACK routine runs, when their threading
# layer is resolved dynamically.  Selecting the sequential layer avoids it, and
# must happen before NumPy loads MKL.  Set MKL_THREADING_LAYER yourself to
# override; nothing is changed if you have.
import os as _os

_os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from . import annotation, compression, datasets, preprocessing
from ._parallel import parallel_map

__all__ = [
    "annotation", "compression", "datasets", "preprocessing", "parallel_map",
    "__version__",
]
