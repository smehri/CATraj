"""The loaders for the preprocessed data."""

import geopandas as gpd
import pandas as pd
import pytest

from catraj import datasets
from catraj.datasets import (
    load_ais, load_coastline, load_ocean, load_weather, read_points, repo_root,
    full_root, sample_root,
)

pytestmark = pytest.mark.data


def test_roots_are_found():
    assert (repo_root() / "data" / "ZenodoData" / "sample").is_dir()
    assert sample_root() == repo_root() / "data" / "ZenodoData" / "sample" / "preprocessed"


def test_repo_root_is_found_from_the_environment(tmp_path, monkeypatch):
    (tmp_path / "data" / "ZenodoData" / "sample").mkdir(parents=True)
    monkeypatch.setenv("CATRAJ_ROOT", str(tmp_path))
    assert repo_root() == tmp_path


def test_repo_root_errors_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("CATRAJ_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(datasets, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    with pytest.raises(FileNotFoundError, match="data/ZenodoData/sample"):
        repo_root()


def test_load_ais():
    ais = load_ais()
    assert isinstance(ais, gpd.GeoDataFrame) and ais.crs.to_epsg() == 32630
    assert pd.api.types.is_integer_dtype(ais.unix_time)
    assert (ais.geometry.x == ais.x).all() and (ais.geometry.y == ais.y).all()
    key = list(zip(ais.mmsi, ais.unix_time))
    assert key == sorted(key)
    assert {"mmsi", "speed_over_ground", "vessel_type"} <= set(ais.columns)


def test_load_ais_columns():
    ais = load_ais(columns=["mmsi", "unix_time"])
    assert set(ais.columns) == {"mmsi", "unix_time", "x", "y", "geometry"}


def test_load_weather():
    w = load_weather()
    assert w.crs.to_epsg() == 32630 and w.station_id.nunique() == 16
    assert pd.api.types.is_integer_dtype(w.unix_time)


def test_load_ocean_and_coastline():
    assert {"wave_height", "unix_time"} <= set(load_ocean().columns)
    assert len(load_ocean(month=None)) == len(load_ocean())
    land = load_coastline()
    assert land.crs.to_epsg() == 32630 and len(land) > 0


def test_an_explicit_root_is_used(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_zenodo"):
        load_ais(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_ocean(tmp_path, month=None)
    assert len(load_ais(sample_root())) == len(load_ais())


def test_read_points_reads_parquet_and_other_files(tmp_path):
    path = sample_root() / "Weather" / "weather.parquet"
    assert len(read_points(path)) == len(load_weather())
    shp = tmp_path / "p.shp"
    gpd.GeoDataFrame({"a": [1]}, geometry=gpd.points_from_xy([1.0], [2.0]),
                     crs=32630).to_file(shp)
    assert read_points(shp).crs.to_epsg() == 32630


def test_dataset_argument():
    assert sample_root("zenodo") == sample_root()
    with pytest.raises(ValueError, match="dataset"):
        sample_root("mars")
    with pytest.raises(ValueError, match="dataset"):
        full_root("mars")


# ----------------------------------------------------------------- US data


def _us_root(tmp_path):
    ais = tmp_path / "AIS"
    ais.mkdir()
    for month, mmsi, t in (("2017-11", [2, 1], [5, 9]), ("2017-12", [1, 2], [3, 4])):
        pd.DataFrame({"mmsi": mmsi, "unix_time": t, "x": [1.0, 2.0], "y": [3.0, 4.0],
                      "vessel_group": ["Tug", "Cargo"]}).to_parquet(ais / f"ais_{month}.parquet")
    return tmp_path


def test_load_ais_us_concatenates_the_months(tmp_path):
    ais = load_ais(_us_root(tmp_path), dataset="us")
    assert ais.crs.to_epsg() == 32618 and len(ais) == 4
    assert list(zip(ais.mmsi, ais.unix_time)) == [(1, 3), (1, 9), (2, 4), (2, 5)]
    assert (ais.geometry.x == ais.x).all()


def test_load_ais_us_months_and_columns(tmp_path):
    root = _us_root(tmp_path)
    nov = load_ais(root, dataset="us", months=["2017-11"], columns=["mmsi"])
    assert len(nov) == 2 and set(nov.columns) == {"mmsi", "x", "y", "geometry"}
    with pytest.raises(FileNotFoundError, match="2018-01"):
        load_ais(root, dataset="us", months=["2018-01"])
    with pytest.raises(ValueError, match="months"):
        load_ais(root, months=["2017-11"])
    with pytest.raises(FileNotFoundError):
        load_ais(tmp_path / "empty", dataset="us")


def test_us_coastline_name(tmp_path):
    (tmp_path / "Coastline").mkdir()
    gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,1 0,1 1,0 0))"]),
                     crs=32618).to_parquet(tmp_path / "Coastline" / "shoreline.parquet")
    assert load_coastline(tmp_path, dataset="us").crs.to_epsg() == 32618
    with pytest.raises(FileNotFoundError):
        load_coastline(tmp_path)  # Zenodo expects coastline.parquet


def test_us_sample_data():
    assert sample_root("us") == repo_root() / "data" / "UsAIS" / "sample" / "preprocessed"
    ais = load_ais(dataset="us")
    assert ais.crs.to_epsg() == 32618 and len(ais) > 0
    assert {"mmsi", "unix_time", "course_over_ground_grid", "vessel_group"} <= set(ais.columns)
    assert load_coastline(dataset="us").crs.to_epsg() == 32618
