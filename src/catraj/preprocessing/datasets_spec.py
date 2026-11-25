"""What differs between the two datasets, in one place: ``get_dataset("zenodo" | "us")``.

The Zenodo archive (:mod:`.archive`) and the US MarineCadastre data
(:mod:`.marinecadastre`) are preprocessed by the *same sequence of steps* with
the *same function names and signatures*; a notebook (or script) sets
``DATASET`` and calls the steps through the spec::

    spec = get_dataset(DATASET)
    steps = spec.steps                     # the module: archive or marinecadastre
    ais = steps.read_ais_raw(spec.ais_path(spec.sample_raw), report)
    ais = steps.drop_invalid_mmsi(ais, report)
    ais = steps.drop_invalid_time(ais, report, period=spec.period)
    ais = steps.drop_invalid_position(ais, spec.area, report)
    ais = steps.unavailable_to_nan(ais, report)
    ais = steps.check_ranges(ais, report)
    ais = steps.drop_duplicates_and_sort(ais, report)
    ais = steps.drop_position_jumps(ais, 50, report)
    ais = spec.attach_vessel_types(ais, spec.sample_raw, report)
    ais = steps.project(ais, crs=spec.crs)
    ais = steps.denoise_ais(ais, 30, report)
    if "coastline" in spec.has:
        coast = steps.read_coastline(spec.coastline_path(spec.sample_raw), spec.area, report)
        coast = steps.project_coastline(coast, spec.crs, report)
        coast = steps.clip_coastline(coast, spec.area, report)

``spec.has`` lists the parts the dataset has (``"ais"``, ``"coastline"``,
``"weather"``, ``"ocean"``): the US data has no weather and no sea state, so a
notebook skips those sections when they are absent.  ``spec.preprocess`` runs the
whole preprocessing and writes ``<out>/AIS/...``, ``<out>/Coastline/...`` and
``preprocessing_report.csv`` / ``.txt`` (the Zenodo run also ``Weather/`` and
``Ocean/``).

Differences a caller still sees:

* the AIS file name: ``AIS/ais.parquet`` (one file) for Zenodo, one file per
  month, ``AIS/ais_2017-11.parquet``, for the US (``spec.ais_outputs(out)``);
* the raw reader's ``path``: a CSV file for Zenodo; a daily ``.csv.zst`` or the
  folder of them for the US (the reader also clips to the study area, see its
  docstring);
* ``attach_vessel_types``: Zenodo reads its static messages and type table from
  the raw folder; the US data carries the codes (``spec.attach_vessel_types``
  hides that);
* the US output has the extra columns ``course_over_ground_grid``,
  ``true_heading_grid`` (course on the map, for the curvilinear model),
  ``vessel_group``, ``length`` and ``width``; Zenodo's has ``rate_of_turn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import archive, marinecadastre

__all__ = ["DatasetSpec", "DATASETS", "get_dataset"]

#: The repository root when run from a source checkout (``src/catraj/...``).
ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's paths, CRS, area, period, steps and entry point."""

    name: str
    title: str
    steps: Any                       # module with the step functions
    crs: int
    area: Any                        # what ``drop_invalid_position`` and the coastline take
    period: tuple                    # (first, last) unix seconds, both included
    raw: Path
    preprocessed: Path
    sample_raw: Path
    sample_preprocessed: Path
    has: tuple = ("ais",)
    _ais_path: Callable = field(default=None, repr=False)
    _coastline_path: Callable = field(default=None, repr=False)
    _attach: Callable = field(default=None, repr=False)
    _preprocess: Callable = field(default=None, repr=False)
    _ais_outputs: Callable = field(default=None, repr=False)

    def ais_path(self, raw_root):
        """The raw AIS file (Zenodo) or folder of daily files (US) under ``raw_root``."""
        return self._ais_path(Path(raw_root))

    def coastline_path(self, raw_root):
        """The raw coastline / shoreline shapefile under ``raw_root``."""
        return self._coastline_path(Path(raw_root))

    def attach_vessel_types(self, ais, raw_root, report=None):
        """Add the vessel type columns (see the module docstring)."""
        return self._attach(ais, Path(raw_root), report)

    def ais_outputs(self, out_root):
        """The AIS Parquet files ``preprocess`` writes (or wrote) under ``out_root``."""
        return self._ais_outputs(Path(out_root))

    def preprocess(self, raw_root=None, out_root=None, *, n_jobs=1, force=False, **kw):
        """Run the whole preprocessing; returns its :class:`PreprocessingReport`.

        Defaults: the full data to its ``preprocessed`` folder.  Further keyword
        arguments go to :func:`~archive.preprocess_archive` or
        :func:`~marinecadastre.preprocess_us`.
        """
        raw_root = self.raw if raw_root is None else Path(raw_root)
        out_root = self.preprocessed if out_root is None else Path(out_root)
        return self._preprocess(raw_root, out_root, n_jobs=n_jobs, force=force, **kw)


def _zenodo_attach(ais, raw_root, report=None):
    return archive.attach_vessel_types(
        ais, raw_root / "P1_AIS_Data" / "nari_static.csv",
        raw_root / "P1_AIS_Codes" / "Ship Types List.csv", report)


def _us_preprocess(raw_root, out_root, *, n_jobs=1, force=False, **kw):
    return marinecadastre.preprocess_us(raw_root, out_root, n_jobs=n_jobs, force=force, **kw)


def _us_shoreline(raw_root):
    for folder in (raw_root / "shoreline", raw_root.parent / "shoreline"):
        hits = sorted(folder.glob("*.shp")) if folder.is_dir() else []
        if hits:
            return hits[0]
    return raw_root / "shoreline" / "us_medium_shoreline.shp"


_ZENODO = ROOT / "data" / "ZenodoData"
_US = ROOT / "data" / "UsAIS"

DATASETS = {
    "zenodo": DatasetSpec(
        name="zenodo", title="Zenodo archive (Brittany, 2015-2016)", steps=archive,
        crs=archive.CRS, area=archive.AREA, period=archive.ARCHIVE_PERIOD,
        raw=_ZENODO, preprocessed=_ZENODO / "preprocessed",
        sample_raw=_ZENODO / "sample" / "raw",
        sample_preprocessed=_ZENODO / "sample" / "preprocessed",
        has=("ais", "coastline", "weather", "ocean"),
        _ais_path=lambda r: r / "P1_AIS_Data" / "nari_dynamic.csv",
        _coastline_path=lambda r: r / "C2_European_Coastline" / "Europe Coastline (Polygone).shp",
        _attach=_zenodo_attach,
        _preprocess=archive.preprocess_archive,
        _ais_outputs=lambda o: [o / "AIS" / "ais.parquet"]),
    "us": DatasetSpec(
        name="us", title="US MarineCadastre AIS, east coast (Nov-Dec 2017)",
        steps=marinecadastre, crs=marinecadastre.US_CRS, area=marinecadastre.AREA,
        period=marinecadastre.US_PERIOD,
        raw=_US / "csv2017", preprocessed=_US / "preprocessed",
        sample_raw=_US / "sample" / "raw",
        sample_preprocessed=_US / "sample" / "preprocessed",
        has=("ais", "coastline"),
        _ais_path=lambda r: r,
        _coastline_path=_us_shoreline,
        _attach=lambda ais, raw_root, report=None: marinecadastre.attach_vessel_types(ais, report),
        _preprocess=_us_preprocess,
        _ais_outputs=lambda o: sorted((o / "AIS").glob("ais_*.parquet"))),
}


def get_dataset(name) -> DatasetSpec:
    """The spec of ``"zenodo"`` or ``"us"`` (a spec passes through)."""
    if isinstance(name, DatasetSpec):
        return name
    try:
        return DATASETS[str(name).lower()]
    except KeyError:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}; got {name!r}") from None
