"""End-to-end on the bundled sample data: the path the tutorial and app take."""

import numpy as np
import pandas as pd
import pytest

from catraj.annotation import IDW, Spline, annotate_points, evaluate_spatial

from helpers import DATASETS

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def weather():
    gpd = pytest.importorskip("geopandas")
    return gpd.read_file(DATASETS / "context" / "Weather" / "Weather_Observation_UTM.shp")


@pytest.fixture(scope="module")
def stations(weather):
    return pd.DataFrame({
        "x": weather.geometry.x, "y": weather.geometry.y,
        "t": weather.local_time.astype(float), "value": weather.Ff.astype(float),
    })


def test_sample_weather_shape(weather):
    assert weather.id_station.nunique() == 16
    assert weather.local_time.nunique() == 25


def test_spatial_comparison_runs(stations):
    s = stations
    table = evaluate_spatial(s.x, s.y, s.value, s.t, methods=["nearest", "idw", "tin"])
    assert table.select() in {"nearest", "idw", "tin"}


def test_one_vessel_is_fully_annotated(stations):
    gpd = pytest.importorskip("geopandas")
    ais = gpd.read_file(DATASETS / "AIS" / "AIS_DLT_UTM.shp")
    track = ais[ais.sourcemmsi == ais.sourcemmsi.value_counts().index[0]]
    out = annotate_points(
        stations, track.geometry.x, track.geometry.y, track.t.astype(float),
        spatial_method=IDW(), temporal_method=Spline(),
    )
    assert np.isfinite(out).all()
    lo, hi = stations.value.min(), stations.value.max()
    assert ((out > lo - 5) & (out < hi + 5)).all()  # plausible wind speeds
