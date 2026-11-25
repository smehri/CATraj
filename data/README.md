# data — all data

All the data the project reads lives here, in three subfolders. Only `sample/`
is committed. git ignores the other two because they are large and carry their
own licences.

```
data/
├── README.md                       # this file
├── LICENSE.md                      # data licences (the code's BSD licence does not cover data)
├── sample/                         # committed: 24-hour extract, 2016-02-05 01:00 to 2016-02-06 01:00 UTC
│   ├── AIS/                        # AIS_DLT_UTM.shp (+ sidecars), AIS_parameters.txt
│   ├── Weather/                    # Weather_Observation_UTM.shp (+ sidecars), Weather_parameters.txt
│   └── Coastline/                  # Coastline_UTM.shp (+ sidecars), Coastline_parameters.txt: land polygons, EEA
├── zenodo_1167595/                 # not committed: the full archive, doi:10.5281/zenodo.1167595 (Ray et al., 2019)
│   ├── P1_AIS_Data/                # nari_dynamic.csv, nari_static.csv, README_parameters.txt
│   ├── P1_AIS_Codes/               # Ship Types List.csv, MMSI Country Codes.csv, ...
│   ├── E1_ocean/                   # oc_february.csv (sea state), README_parameters.txt
│   ├── E2_Weather_Conditions/      # weather observations and stations
│   ├── C2_European_Coastline/      # Europe Coastline (Polygone).shp (+ sidecars): the full EEA coastline
│   └── clean/                      # scripts/prepare_zenodo.py: ais_dynamic, ocean_<month>, weather (.parquet)
└── cache/                          # not committed: files the scripts build; safe to delete
```

## `sample/`: works without a download

| used by | needs |
|---|---|
| `notebooks/01_trajectory_annotation.ipynb` (`SOURCE = "sample"`) | `sample/AIS`, `sample/Weather` |
| `notebooks/03_cpls_compression.ipynb` | `sample/AIS`, `sample/Coastline` |
| `app/` (the annotation app) | `sample/AIS`, `sample/Weather` |
| `tests/` (tests marked `data`) | `sample/AIS`, `sample/Weather`, `sample/Coastline` |

## `zenodo_1167595/`: download it first

To fill it, run this from the repository root:

```bash
python scripts/download_data.py          # or: annotation | cadp | cpls | --check
```

Then clean it (descriptive column names, "not available" codes as NaN, noise
removed with a median filter), which writes `zenodo_1167595/clean/`:

```bash
python scripts/prepare_zenodo.py         # needs pyarrow; --force rebuilds, --window sets seconds
```

You can also set `DOWNLOAD = True` in a notebook's settings cell, or download
by hand. All three ways are explained step by step in
[`docs/getting_the_data.md`](../docs/getting_the_data.md).

| used by | needs |
|---|---|
| `scripts/prepare_zenodo.py` | `P1_AIS_Data`, `P1_AIS_Codes`, `E1_ocean`, `E2_Weather_Conditions` (writes `clean/`) |
| `notebooks/01_trajectory_annotation.ipynb` (`SOURCE = "zenodo"`) | `clean/ais_dynamic.parquet`, `clean/weather.parquet`, the README files in `P1_AIS_Data`, `E2_Weather_Conditions` |
| `notebooks/02_cadp_compression.ipynb` | `clean/ais_dynamic.parquet`, `clean/ocean_february.parquet` |
| `notebooks/03_cpls_compression.ipynb` (optional last section) | `C2_European_Coastline` |

## Licences

**AIS** (both the sample and the archive): CC BY-NC-SA 4.0. You must attribute
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
