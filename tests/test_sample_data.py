"""End-to-end on the bundled sample data: the path the tutorial and app take."""

import numpy as np
import pandas as pd
import pytest

from catraj.annotation import IDW, Spline, annotate_points, evaluate_spatial
from catraj.datasets import load_ais, load_coastline, load_weather

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def weather():
    pytest.importorskip("geopandas")
    return load_weather()


@pytest.fixture(scope="module")
def stations(weather):
    return pd.DataFrame({
        "x": weather.geometry.x, "y": weather.geometry.y,
        "t": weather.unix_time.astype(float),
        "value": weather.wind_speed.astype(float),
    })


def test_sample_weather_shape(weather):
    # Measured on the 2015-11-09/10 sample: 16 stations, hourly for two days.
    assert weather.station_id.nunique() == 16
    assert weather.unix_time.nunique() == 48


def test_spatial_comparison_runs(stations):
    s = stations
    table = evaluate_spatial(s.x, s.y, s.value, s.t, methods=["nearest", "idw", "tin"])
    assert table.select() in {"nearest", "idw", "tin"}


def test_one_vessel_is_fully_annotated(stations):
    pytest.importorskip("geopandas")
    ais = load_ais()
    track = ais[ais.mmsi == ais.mmsi.value_counts().index[0]]
    out = annotate_points(
        stations, track.geometry.x, track.geometry.y, track.unix_time.astype(float),
        spatial_method=IDW(), temporal_method=Spline(),
    )
    assert np.isfinite(out).all()
    lo, hi = stations.value.min(), stations.value.max()
    assert ((out > lo - 5) & (out < hi + 5)).all()  # plausible wind speeds


def test_sample_coastline_fits_the_ais():
    pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    land = load_coastline()
    ais = load_ais()
    assert land.crs.to_epsg() == 32630
    assert len(land) > 0 and not land.geometry.is_empty.any()
    assert set(land.geom_type) <= {"Polygon", "MultiPolygon"}

    lx0, ly0, lx1, ly1 = land.total_bounds
    ax0, ay0, ax1, ay1 = ais.total_bounds
    assert lx0 < ax1 and lx1 > ax0 and ly0 < ay1 and ly1 > ay0   # they overlap
    # make_sample.py cuts the coastline to the AIS box plus 0.2 degrees (in the
    # coastline's own CRS), so in degrees the land stays inside that box.
    lon0, lat0, lon1, lat1 = land.to_crs(4326).total_bounds
    assert lon0 >= ais.lon.min() - 0.2 - 1e-3 and lat0 >= ais.lat.min() - 0.2 - 1e-3
    assert lon1 <= ais.lon.max() + 0.2 + 1e-3 and lat1 <= ais.lat.max() + 0.2 + 1e-3

    # Positions on the mapped land are moored vessels in ports the generalised
    # coastline closes off.  The old 24 h sample had 0.002 % of positions on
    # land; here, measured on 2015-11-09/10, 9.5 % are, 34 vessels, 96 % of
    # those rows stationary (ferries at their berths).  So: positions at all
    # at most 10 %, and moving ones (>= 1 knot) at most 0.5 % of all (measured
    # 0.38 %, 12 vessels).
    union = shapely.union_all(land.geometry.values)
    shapely.prepare(union)
    on_land = shapely.intersects(union, ais.geometry.values)
    assert on_land.mean() <= 0.10
    assert (on_land & (ais.speed_over_ground >= 1).to_numpy()).mean() <= 0.005
