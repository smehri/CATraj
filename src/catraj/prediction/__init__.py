"""Context-aware trajectory prediction.

The prediction half of the hybrid model of [Mehri2021] (Fig. 5): a curvilinear
(dead-reckoning) model predicts a vessel's next three positions from its speed
and course, and a Long Short-Term Memory network, one per vessel type, corrects
them.  :mod:`.curvilinear` holds the physical model; :mod:`.clstm` holds the
samples, the split by trajectory, the networks and the error measures.  Only
the networks need PyTorch (``pip install catraj[predict]``); importing this
package does not.
"""

from .clstm import (
    PredictionSamples,
    make_samples,
    prediction_errors,
    split_by_trajectory,
)
from .curvilinear import KNOT, curvilinear_predict, mean_interval

__all__ = [
    "KNOT",
    "mean_interval",
    "curvilinear_predict",
    "PredictionSamples",
    "make_samples",
    "split_by_trajectory",
    "prediction_errors",
]
