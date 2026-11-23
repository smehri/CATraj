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
    return gpd.read_file(DATASETS / "Weather" / "Weather_Observation_UTM.shp")


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


def test_sample_coastline_fits_the_ais():
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    land = gpd.read_file(DATASETS / "Coastline" / "Coastline_UTM.shp")
    ais = gpd.read_file(DATASETS / "AIS" / "AIS_DLT_UTM.shp")
    assert land.crs.to_epsg() == 32630
    assert len(land) > 0 and not land.geometry.is_empty.any()
    assert (land.geom_type == "Polygon").all()

    lx0, ly0, lx1, ly1 = land.total_bounds
    ax0, ay0, ax1, ay1 = ais.total_bounds
    assert lx0 < ax1 and lx1 > ax0 and ly0 < ay1 and ly1 > ay0   # they overlap
    margin = 5_000 + 1e-6
    assert lx0 >= ax0 - margin and ly0 >= ay0 - margin
    assert lx1 <= ax1 + margin and ly1 <= ay1 + margin

    # The positions are at sea; a handful sit on the mapped land.
    on_land = shapely.intersects(
        shapely.union_all(land.geometry.values), ais.geometry.values).mean()
    assert on_land <= 0.001
