"""Read the preprocessed data the same way everywhere.

Tests, the annotation app and the notebooks all read the committed sample, or
the full archive once it has been preprocessed, through these functions. A
*root* is a folder written by :func:`catraj.preprocessing.preprocess_archive`:

* ``data/sample/preprocessed`` (committed; the default), or
* ``data/zenodo_1167595/preprocessed`` (the whole archive, not committed).

Reading Parquet needs ``pyarrow`` (``pip install catraj[data]``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

__all__ = [
    "repo_root", "sample_root", "load_ais", "load_weather", "load_ocean",
    "load_coastline", "read_points", "CRS",
]

#: The projected CRS the preprocessed ``x`` and ``y`` are in (UTM zone 30N).
CRS = "EPSG:32630"

_SAMPLE = Path("data") / "sample" / "preprocessed"


def repo_root() -> Path:
    """The repository root: the folder that holds ``data/sample``.

    Looked up from the package (which sits in ``<root>/src`` when installed
    editable), then upwards from the working directory. Set ``CATRAJ_ROOT``
    to override. Raises ``FileNotFoundError`` when none holds the data folder,
    for instance with a regular (non-editable) install.
    """
    candidates = []
    if os.environ.get("CATRAJ_ROOT"):
        candidates.append(Path(os.environ["CATRAJ_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[2])
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    for folder in candidates:
        if (folder / "data" / "sample").is_dir():
            return folder
    raise FileNotFoundError(
        "Cannot find the repository's data/sample folder. Run from a clone of "
        "the repository, install it editable (pip install -e .), or set the "
        "CATRAJ_ROOT environment variable."
    )


def sample_root() -> Path:
    """The committed preprocessed sample, ``data/sample/preprocessed``."""
    root = repo_root() / _SAMPLE
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. Rebuild it with scripts/prepare_zenodo.py "
            "--data data/sample/raw --out data/sample/preprocessed.")
    return root


def _file(root, *parts) -> Path:
    base = sample_root() if root is None else Path(root)
    path = base.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. {base} should be a folder written by "
            "catraj.preprocessing.preprocess_archive (for the full archive, "
            "run scripts/prepare_zenodo.py first).")
    return path


def _points(df: pd.DataFrame, crs=CRS):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)


def read_points(path, crs=CRS):
    """Read a point table as a GeoDataFrame.

    A ``.parquet`` file (preprocessed AIS or weather) is built into points from
    its ``x`` and ``y`` columns, in ``crs``; anything else is read by
    ``geopandas.read_file`` and must carry its own CRS.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return _points(pd.read_parquet(path), crs)
    import geopandas as gpd

    return gpd.read_file(path)


def load_ais(root=None, columns=None):
    """The preprocessed AIS as a GeoDataFrame in EPSG:32630.

    Columns are those of ``ais.parquet`` (``mmsi``, ``unix_time`` in seconds
    since 1970 as an integer, ``speed_over_ground``, ``course_over_ground``,
    ``true_heading``, ``navigational_status``, ``rate_of_turn``, ``lon``,
    ``lat``, ``x``, ``y``, ``ship_type_code``, ``vessel_type``); the geometry
    is ``(x, y)``. Rows are sorted by ``(mmsi, unix_time)``, with a fresh
    index. ``columns`` limits the columns read (``x`` and ``y`` are kept).
    """
    path = _file(root, "AIS", "ais.parquet")
    if columns is not None:
        columns = list(dict.fromkeys([*columns, "x", "y"]))
    return _points(pd.read_parquet(path, columns=columns))


def load_weather(root=None):
    """The preprocessed hourly weather as a GeoDataFrame in EPSG:32630.

    One row per station and hour (``station_id``, ``unix_time``,
    ``air_temperature``, ``wind_speed``, ... ``x``, ``y``).
    """
    return _points(pd.read_parquet(_file(root, "Weather", "weather.parquet")))


def load_ocean(root=None, month="november"):
    """The preprocessed sea-state forecast as a DataFrame (``lon``, ``lat``).

    ``month`` names the file ``Ocean/ocean_<month>.parquet``; ``None`` reads
    every month found and concatenates them.
    """
    if month is not None:
        return pd.read_parquet(_file(root, "Ocean", f"ocean_{month}.parquet"))
    base = sample_root() if root is None else Path(root)
    files = sorted((base / "Ocean").glob("ocean_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Ocean/ocean_*.parquet under {base}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_coastline(root=None):
    """The land polygons as a GeoDataFrame in EPSG:32630 (GeoParquet)."""
    import geopandas as gpd

    return gpd.read_parquet(_file(root, "Coastline", "coastline.parquet"))
