# data — all data

All the data the project reads lives here, in three subfolders. Only `sample/`
is committed. git ignores the other two because they are large and carry their
own licences.

Two forms of the data exist. *Raw* is the Zenodo archive exactly as published
(CSV files with coded column names and "not available" codes, and a
shapefile). *Preprocessed* is what `catraj.preprocessing.preprocess_archive`
makes of it: Parquet files with descriptive names, the codes as NaN, values
range-checked, positions projected to EPSG:32630 and noise removed by a 30 s
median filter. Code, tests, the app and the notebooks read the preprocessed
form, through `catraj.datasets`.

```
data/
├── README.md                       # this file
├── LICENSE.md                      # data licences (the code's BSD licence does not cover data)
├── sample/                         # committed: two days, 2015-11-09 00:00 to 2015-11-11 00:00 UTC (60 MB raw, 9 MB preprocessed)
│   ├── raw/                        # the archive's own layout, cut to the window and the AIS area (scripts/make_sample.py)
│   │   ├── P1_AIS_Data/            # nari_dynamic.csv, nari_static.csv, README_parameters.txt
│   │   ├── P1_AIS_Codes/           # code tables, whole
│   │   ├── E2_Weather_Conditions/  # observations in the window, stations whole
│   │   ├── E1_ocean/               # oc_november.csv: the window plus one 3 h step each side
│   │   └── C2_European_Coastline/  # Europe Coastline (Polygone).shp (+ sidecars), clipped
│   └── preprocessed/               # preprocess_archive(raw, ...) over the folder above
│       ├── AIS/                    # ais.parquet, mmsi_check.csv
│       ├── Weather/                # weather.parquet
│       ├── Ocean/                  # ocean_november.parquet
│       ├── Coastline/              # coastline.parquet (GeoParquet, EPSG:32630)
│       └── preprocessing_report.csv, .txt   # rows in and out of every step
├── zenodo_1167595/                 # not committed: the full archive, doi:10.5281/zenodo.1167595 (Ray et al., 2019)
│   ├── P1_AIS_Data/                # nari_dynamic.csv, nari_static.csv, README_parameters.txt
│   ├── P1_AIS_Codes/               # Ship Types List.csv, MMSI Country Codes.csv, ...
│   ├── E1_ocean/                   # oc_<month>.csv (sea state), README_parameters.txt
│   ├── E2_Weather_Conditions/      # weather observations and stations
│   ├── C2_European_Coastline/      # Europe Coastline (Polygone).shp (+ sidecars): the full EEA coastline
│   └── preprocessed/               # not committed: scripts/prepare_zenodo.py writes AIS/ais.parquet, Weather/, Ocean/, Coastline/
└── cache/                          # not committed: files the scripts build; safe to delete
```

## `sample/`: works without a download

A two-day window, 2015-11-09 00:00 to 2015-11-11 00:00 UTC (start inclusive,
end exclusive), cut from the archive and kept in both forms.

| dataset | raw | preprocessed |
|---|---|---|
| AIS | 239,075 messages | 237,318 positions, 565 vessels, with vessel type |
| Weather | 16 stations, hourly | 768 observations (16 stations x 48 hours) |
| Sea state | `oc_november.csv`, the window plus one 3 h step each side, in the AIS box | 677,736 grid-cell rows |
| Coastline | the EEA polygons clipped to the AIS box plus 0.2 degrees | 729 land polygons, EPSG:32630 |

Size on disk: `raw/` 60 MB (the sea state is 44 MB of it), `preprocessed/` 9 MB.

How it was made: `python scripts/make_sample.py` cuts `raw/` from
`zenodo_1167595/` (it needs the full download), and
`python scripts/prepare_zenodo.py --data data/sample/raw --out data/sample/preprocessed`
(that is, `catraj.preprocessing.preprocess_archive`) makes `preprocessed/`. The
step-by-step account is in `preprocessed/preprocessing_report.txt`.

Read it with `catraj.datasets` (`load_ais()`, `load_weather()`, `load_ocean()`,
`load_coastline()`, all defaulting to `preprocessed/`):

```python
from catraj.datasets import load_ais, load_coastline
ais = load_ais()          # GeoDataFrame, EPSG:32630, unix_time in seconds
land = load_coastline()
```

| used by | needs |
|---|---|
| `notebooks/01_trajectory_annotation.ipynb` (`SOURCE = "sample"`) | `preprocessed/AIS`, `preprocessed/Weather` |
| `notebooks/03_cpls_compression.ipynb` | `preprocessed/AIS`, `preprocessed/Coastline` |
| `app/` (the annotation app) | `preprocessed/AIS`, `preprocessed/Weather` |
| `tests/` (tests marked `data`) | `preprocessed/` (all four), `raw/E2_Weather_Conditions/README_parameters.txt` |
| `tests/test_archive.py`, `scripts/prepare_zenodo.py` | `raw/` |

## `zenodo_1167595/`: download it first

To fill it, run this from the repository root:

```bash
python scripts/download_data.py          # or: annotation | cadp | cpls | --check
```

Then preprocess it (descriptive column names, "not available" codes as NaN,
ranges checked, projection to EPSG:32630, noise removed with a median filter),
which writes `zenodo_1167595/preprocessed/` (not committed; the whole archive's
AIS alone is about 18 million rows):

```bash
python scripts/prepare_zenodo.py         # needs pyarrow; --force rebuilds, --window sets seconds
```

or, from Python, `catraj.preprocessing.preprocess_archive(raw_root, out_root)`.
`catraj.datasets.load_ais("data/zenodo_1167595/preprocessed")` and the other
loaders read the result.

You can also set `DOWNLOAD = True` in a notebook's settings cell, or download
by hand. All three ways are explained step by step in
[`docs/getting_the_data.md`](../docs/getting_the_data.md).

| used by | needs |
|---|---|
| `scripts/prepare_zenodo.py` | `P1_AIS_Data`, `P1_AIS_Codes`, `E1_ocean`, `E2_Weather_Conditions` (writes `preprocessed/`) |
| `notebooks/01_trajectory_annotation.ipynb` (`SOURCE = "zenodo"`) | `preprocessed/AIS/ais.parquet`, `preprocessed/Weather/weather.parquet` |
| `notebooks/02_cadp_compression.ipynb` | `preprocessed/AIS/ais.parquet`, `preprocessed/Ocean/ocean_february.parquet` |
| `notebooks/03_cpls_compression.ipynb` (optional last section) | `C2_European_Coastline` |

## Licences

**AIS** (both the sample and the archive, raw and preprocessed): CC BY-NC-SA 4.0. You must attribute
NARI, Brest. Commercial use is not allowed, and data derived from it must be
shared under the same licence.

**Sea state** (SHOM – IFREMER, WAVEWATCH III): public, but you must cite
Boudière et al., 2013,
[doi:10.1016/j.ijome.2013.11.010](https://doi.org/10.1016/j.ijome.2013.11.010).

**Weather** (rp5.ua, Met Office / NOAA): public; cite the source.

**Coastline** (European Environment Agency, "EEA coastline for analysis"; the sample
and the archive): free to re-use, commercially or not, provided the source is
acknowledged. Copyright holder: Directorate-General for Environment (DG ENV),
[eea.europa.eu/legal/copyright](http://www.eea.europa.eu/legal/copyright).

[`LICENSE.md`](LICENSE.md) gives the details for the sample.
