"""Trajectory annotation -- linking trajectories to their environmental context.

Movement data and context data are observed at different resolutions, and
movement is far denser.  Annotation supplies a context value for every
trajectory point, in two stages.

The method is not specific to any domain: vessels, aircraft, vehicles, wildlife
and pedestrians all produce trajectories, and any of them can be annotated with
whatever context was measured around them.

1. **Spatial** (:mod:`.spatial`) -- context is measured at scattered stations,
   never where the moving object is.  Station values are interpolated to its
   position, one epoch at a time.
2. **Temporal** (:mod:`.interpolators`) -- a curve is fitted through the
   resulting series and read at the object's exact timestamp.

Each stage has a companion module that compares candidate methods by
cross-validation: :mod:`.spatial_evaluation` and :mod:`.annotation_evaluation`.

The temporal stage follows [Mehri2023] section 4.1.  The spatial stage extends
it -- that paper generated per-epoch surfaces but placed the study of spatial
interpolation outside its scope.

Each module's docstring carries the part of the method it implements.
"""

from ._circular import COMPASS_STEP, angular_residual, decode_compass
from ._descriptions import (
    FieldDescription,
    describe_columns,
    parse_data_dictionary,
)
from ._metrics import Metrics, residuals, score
from .annotate import annotate, annotate_points, build_series
from .annotation_evaluation import (
    TemporalTable,
    default_candidates,
    evaluate_temporal,
    k_fold,
)
from .interpolators import (
    DLT,
    TEMPORAL_METHODS,
    PiecewisePolynomial,
    Spline,
    TemporalInterpolator,
)
from .spatial import (
    IDW,
    RBF,
    SPATIAL_METHODS,
    CloughTocher,
    Kriging,
    Nearest,
    SpatialInterpolator,
    TIN,
    to_surface,
)
from .spatial_evaluation import (
    SpatialTable,
    evaluate_spatial,
    leave_one_station_out,
)

__all__ = [
    # spatial stage
    "SpatialInterpolator", "Nearest", "IDW", "TIN", "CloughTocher", "RBF",
    "Kriging", "SPATIAL_METHODS", "to_surface",
    "evaluate_spatial", "leave_one_station_out", "SpatialTable",
    # temporal stage
    "TemporalInterpolator", "DLT", "PiecewisePolynomial", "Spline",
    "TEMPORAL_METHODS",
    "evaluate_temporal", "k_fold", "default_candidates", "TemporalTable",
    # composing them
    "annotate", "annotate_points", "build_series",
    # shared
    "Metrics", "score", "residuals",
    "decode_compass", "angular_residual", "COMPASS_STEP",
    "parse_data_dictionary", "describe_columns", "FieldDescription",
]
