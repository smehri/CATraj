# catraj — context-aware trajectory processing

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?urlpath=lab/tree/notebooks)

Reference implementations of context-aware trajectory algorithms, compatible
with [MovingPandas](https://github.com/movingpandas/movingpandas).

Movement is embedded in context. A vessel avoids shallow water and heavy seas;
an aircraft rides the wind it is given; an animal follows the terrain it can
cross. `catraj` provides algorithms that carry environmental, internal and
semantic context through annotation, compression and prediction instead of
discarding it.

The algorithms apply to **any moving object** — vessels, aircraft, vehicles,
wildlife, pedestrians. The maritime data in the examples is one application,
not the scope.

**This release implements trajectory annotation, CADP compression, CPLS
compression, 2stage-CPLS compression and CLSTM prediction.** The preprocessing
of the public AIS archive they are demonstrated on is part of the package too.

---

## Algorithms

| Algorithm | What it does | Paper | Status |
|---|---|---|---|
| **Annotation** | Attaches context observed at coarse resolution to high-frequency trajectory points. Spatial interpolation across observation stations, then temporal interpolation through time. | 2023 | **implemented** |
| **CADP** | Context-Aware Douglas–Peucker. Extends DP with three context domains — environmental, internal, semantic — retaining points that carry contextual information a purely spatial criterion would discard. | 2025 | **implemented** |
| **CPLS** | Context-aware Piecewise Linear Segmentation. Guarantees logical consistency, e.g., a compressed (simplified) vessel trajectory may not cross the shoreline, and keeps its topological signature (barriers, direction, self-intersections). | 2021 | **implemented** |
| **2stage-CPLS** | CPLS plus stay-point preservation. | 2021 | **implemented** |
| **CLSTM** | Context-aware LSTM. Routes each trajectory to a per-type network, avoiding Simpson's paradox in a single global model. The choices made where the paper is open (including what the network learns) are documented in `catraj.prediction.clstm`. | 2021, 2023 | **implemented** |

---

## Papers

The two open-access papers are in [`docs/papers/`](docs/papers/), so each module can be
checked against the method it implements; the 2023 paper is available from its
publisher via the DOI. The implementation does not depend on having them.

| Year | Paper | | |
|---|---|---|---|
| 2025 | *A Novel Context-Aware Douglas–Peucker (CADP) Trajectory Compression Method.* ISPRS Int. J. Geo-Inf., 14(2), 58. | [pdf](docs/papers/2025_Mehri_IJGI_CADP.pdf) | [doi](https://doi.org/10.3390/ijgi14020058) |
| 2023 | *A context-aware approach for vessels' trajectory prediction.* Ocean Engineering, 282, 114916. | — | [doi](https://doi.org/10.1016/j.oceaneng.2023.114916) |
| 2021 | *A Contextual Hybrid Model for Vessel Movement Prediction.* IEEE Access, 9, 45600–45613. | [pdf](docs/papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf) | [doi](https://doi.org/10.1109/ACCESS.2021.3066463) |

---

## Repository layout

| folder | what it holds |
|---|---|
| [`src/catraj/`](src/catraj/) | the library: `annotation`, `preprocessing`, `compression`, `prediction`, `datasets` |
| [`tests/`](tests/) | the test suite (`pytest`) |
| [`notebooks/`](notebooks/) | the notebooks, in reading order |
| [`scripts/`](scripts/) | command-line scripts: the data downloader and whole-archive workflows |
| [`app/`](app/) | the interactive annotation app |
| [`data/`](data/) | all data: the two-day sample in `sample/` (raw and preprocessed), committed; the downloaded archive in `zenodo_1167595/` (and its `preprocessed/` folder), not committed |
| `results/` | everything the notebooks, scripts and app write, not committed ([`README`](results/README.md)) |
| [`docs/`](docs/) | the source papers (open-access ones) and [`getting_the_data.md`](docs/getting_the_data.md) |
| [`binder/`](binder/) | the environment Binder builds for the notebooks |
| [`CLSTM_GUIDE.md`](CLSTM_GUIDE.md), [`CPLS_PHASE1_GUIDE.md`](CPLS_PHASE1_GUIDE.md) | historical implementation briefs for CLSTM and CPLS, kept for reference; the docstrings and notebooks describe the code as it is |

---

## Notebooks

The notebooks run live on [Binder](https://mybinder.org) with no install and no account, on the committed sample (`data/sample/`). github.com does not render the interactive hvplot figures, so use Binder or nbviewer to see them.

| Notebook | Purpose | Run | View | Full-archive run |
|---|---|---|---|---|
| `01_preprocessing` | Rename, check and clean the AIS, weather, sea-state and coastline files step by step on the sample; downloads and preprocesses the full archive. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?labpath=notebooks%2F01_preprocessing.ipynb) | [nbviewer](https://nbviewer.org/github/smehri/CATraj/blob/main/notebooks/01_preprocessing.ipynb) | this notebook covers the full archive itself |
| `02_trajectory_annotation` | Annotate every vessel with weather and sea state, on the sample and on the full archive. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?labpath=notebooks%2F02_trajectory_annotation.ipynb) | [nbviewer](https://nbviewer.org/github/smehri/CATraj/blob/main/notebooks/02_trajectory_annotation.ipynb) | [`full_runs/02_trajectory_annotation.ipynb`](notebooks/full_runs/02_trajectory_annotation.ipynb) |
| `03_two_stage_cpls_compression` | Compress every trip of every vessel with 2stage-CPLS, keeping the coastline constraint. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?labpath=notebooks%2F03_two_stage_cpls_compression.ipynb) | [nbviewer](https://nbviewer.org/github/smehri/CATraj/blob/main/notebooks/03_two_stage_cpls_compression.ipynb) | [`full_runs/03_two_stage_cpls_compression.ipynb`](notebooks/full_runs/03_two_stage_cpls_compression.ipynb) |
| `04_cadp_compression` | CADP compression of every trip, driven by the sea state (depth, wave height, wave direction). | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?labpath=notebooks%2F04_cadp_compression.ipynb) | [nbviewer](https://nbviewer.org/github/smehri/CATraj/blob/main/notebooks/04_cadp_compression.ipynb) | [`full_runs/04_cadp_compression.ipynb`](notebooks/full_runs/04_cadp_compression.ipynb) |
| `05_clstm_prediction` | CLSTM trajectory prediction (PyTorch), trained on the clean trips and on the 2stage-CPLS output of notebook 03. | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/smehri/CATraj/main?labpath=notebooks%2F05_clstm_prediction.ipynb) | [nbviewer](https://nbviewer.org/github/smehri/CATraj/blob/main/notebooks/05_clstm_prediction.ipynb) | [`full_runs/05_clstm_prediction.ipynb`](notebooks/full_runs/05_clstm_prediction.ipynb) |

The notebooks in `notebooks/` run on the sample. [`notebooks/full_runs/`](notebooks/full_runs/) holds copies of notebooks 02 to 05 executed on the full archive, outputs kept, so the numbers of the whole archive can be read without rerunning anything (notebook 01 covers the full archive in its own later parts).

Binder offers about 2 GB of RAM and one or two CPUs, so it is meant for the sample only. The full-archive runs (several GB of downloads) are not meant for Binder: run them locally, see [`docs/getting_the_data.md`](docs/getting_the_data.md).

### Running the notebooks

**On Binder** (the sample): click a Run badge above. Nothing to install.

**Locally**: create the environment, either `pip install -e ".[dev]"` (see [Installation](#installation); `dev` adds the app, MovingPandas, pyarrow and PyTorch to the library) or `conda env create -f binder/environment.yml` followed by `pip install -e .`. Then open the notebooks in JupyterLab or VS Code with that environment as the kernel.

Two settings at the top of notebooks 02 to 05 choose what they run on; each reads an environment variable, so a notebook can be switched without editing it:

| setting | environment variable | meaning |
|---|---|---|
| `DATA` | `CATRAJ_DATA` | `"sample"` (default) or `"full"`; the results go to `results/<notebook>/<DATA>/` |
| `N_JOBS` | `CATRAJ_N_JOBS` | worker processes; `-1` (default) uses every core, `1` is serial |

Notebook 05 also has `DEVICE` (`CATRAJ_DEVICE`, `"auto"` uses a GPU when there is one, `"cpu"` forces the CPU).

Run them in order, 01 to 05. Notebooks 02, 03 and 04 are independent of each other; **05 reads what 03 wrote** (`results/03_two_stage_cpls_compression/<DATA>/`), so run 03 with the same `DATA` first. **The full archive needs notebook 01 first**: set `DOWNLOAD = True` and `PREPROCESS_FULL = True` there (or run `scripts/download_data.py --months all` and `scripts/prepare_zenodo.py`, see [`docs/getting_the_data.md`](docs/getting_the_data.md)), then run 02 to 05 with `DATA = "full"`.

From the command line:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_two_stage_cpls_compression.ipynb
CATRAJ_DATA=full python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_two_stage_cpls_compression.ipynb
```

(In PowerShell set the variable first: `$env:CATRAJ_DATA = "full"`.) `--inplace` overwrites the notebook with its executed copy, so use `--output` for a copy that leaves it as committed.

**Windows with conda.** The kernel must find the environment's DLLs (matplotlib, MKL, GDAL): either start Jupyter from an activated environment, or put `<env>\Libraryin` on `PATH` for the kernel, for example in the kernel's `kernel.json` (see [Installation](#installation) for registering a kernel). Import `catraj` before numpy: `catraj/__init__.py` sets `MKL_THREADING_LAYER=SEQUENTIAL` (some Intel MKL builds abort the process on the first LAPACK call otherwise), and it only helps if it runs before numpy loads MKL. Notebooks 03 to 05 set the variable themselves in their first cell, before anything else is imported.

---

## Results on the full archive

What the full-archive runs of the notebooks produced (five months, October to
December 2015 and February to March 2016, 14.9 million AIS positions; January
2016 is excluded). The numbers are those of the `summary.csv` and timing files
in `results/<notebook>/full/` (not committed) and of the executed copies in
[`notebooks/full_runs/`](notebooks/full_runs/); every notebook gives the same
table for the sample.

| notebook | result |
|---|---|
| `02_trajectory_annotation` | all 14,906,535 positions of 4,720 vessels annotated with all eight variables (wind speed, air temperature, air pressure, wind direction, sea level, wave height, wave length, wave direction), none missing; 197 s in total (`annotation_timing.csv`) |
| `03_two_stage_cpls_compression` | 2stage-CPLS, ε 1 km, 1.54 m/s, stays of 100 m and 600 s, coastline as barrier: 14,777,418 points of 19,407 trips (2,076 vessels) become 1,636,505, a compression of 88.9 % (21,470 of the kept points are stay points); 270 s |
| `04_cadp_compression` | CADP, one context at a time, on the same trips (points kept of 14,777,418): Douglas–Peucker alone 71,165 (99.5 %); depth 114,940 (99.2 %); wave height 4,836,798 (67.3 %); wave direction 8,306,633 (43.8 %); all three together 9,301,789 (37.1 %). `context_fidelity.csv` gives how far the dropped points' context lies from the compressed track |
| `05_clstm_prediction` | **notebook 05 — full-archive run in progress** |

---

## Installation

Requires Python ≥ 3.10. The geospatial stack installs far more reliably from
conda-forge, especially on Windows:

```bash
conda create -n catraj -c conda-forge python=3.11 geopandas scipy scikit-learn pandas panel hvplot geoviews-core movingpandas stonesoup ipykernel pytest
conda activate catraj

git clone https://github.com/smehri/catraj
cd catraj
pip install -e ".[dev]"
```

`[dev]` installs everything the notebooks and tests use (the app, MovingPandas, pyarrow for the Parquet files, PyTorch for CLSTM, pytest, black, ruff). Use `".[app]"`, `".[predict]"` and so on for a part, or omit the extras for the library alone. `stonesoup` is only an optional extra of
MovingPandas, which warns on import without it.

To run the notebooks from an editor (VS Code, JupyterLab) without activating
the environment first, register it as a kernel and pick **Python (catraj)**.
The kernel carries the variables `conda activate` would set, so GDAL finds its
data files:

```bash
python -m ipykernel install --user --name catraj --display-name "Python (catraj)" ^
    --env GDAL_DATA "%CONDA_PREFIX%\Library\share\gdal" ^
    --env PROJ_DATA "%CONDA_PREFIX%\Library\share\proj"
```

(On Linux or macOS: `$CONDA_PREFIX/share/gdal` and `$CONDA_PREFIX/share/proj`,
with `\` for line continuation.)

---

## Using it

```python
import geopandas as gpd, pandas as pd
from catraj.annotation import IDW, Spline, annotate_points

# Source files often use coded names; give columns names that say what they measure.
obs = gpd.read_file("weather.shp").rename(columns={"Ff": "wind_speed", "local_time": "unix_time"})
stations = pd.DataFrame({
    "x": obs.geometry.x, "y": obs.geometry.y,
    "t": obs.unix_time.astype(float),
    "value": obs.wind_speed.astype(float),
})

traj = gpd.read_file("trajectories.shp").rename(
    columns={"sourcemmsi": "mmsi", "t": "unix_time", "speedoverg": "speed_over_ground"})
traj["wind_speed"] = annotate_points(
    stations,
    traj.geometry.x, traj.geometry.y, traj.unix_time.astype(float),
    spatial_method=IDW(power=2), temporal_method=Spline(),
)
```

Millions of positions are fine: long inputs are processed in time blocks
(`chunk_epochs`), DLT is evaluated for all positions at once, and `n_jobs=-1`
spreads the blocks over every core (as it spreads epochs in
`evaluate_spatial`). In the full-archive run of notebook 02 (14.9 million
positions, `n_jobs=-1`, `annotation_timing.csv`) a linear variable takes about
18 to 19 s for the whole archive, a bearing 30 to 35 s, and all eight variables
197 s in total.

Context attaches as ordinary columns, so it survives the `.iloc` slicing that
trajectory compression performs — a compressed trajectory keeps the context of
the points it retains.

**Every processing step median-filters its inputs by default, over a 30 second
window; pass `filter_window=None` to turn it off.** Raw AIS positions are
noisy, so `annotate_points` filters each station's observation series,
`compress`, `cpls_compress` and `two_stage_cpls_compress` filter the positions
(CADP also its threshold columns, 2stage-CPLS also its speed series), and the
compressed rows carry the filtered values. Time comes from `time_column` or a
DatetimeIndex (MovingPandas); a plain numeric index without `time_column` is
refused rather than filtered by row number. The array-level functions
(`douglas_peucker`, `cadp_select`, `cpls_select`) filter nothing. The notebooks read the preprocessed data, which
is already median-filtered (30 s) by `preprocess_archive`, so they pass
`filter_window=None` and do not filter a second time.

`denoise` is the same step on its own, for another window, the mean (as in the
paper) instead of the median, or one pass shared by several steps. It replaces
each point by the median of the points within a centred time window, for the
positions and any columns you name (bearings in `circular`, filtered through
their unit vector); rows, index and columns are unchanged:

```python
from catraj.preprocessing import denoise

clean = denoise(traj, window=60, columns=["speed_over_ground"],
                time_column="unix_time", by="mmsi")   # seconds; 30 by default
```

Measured on the committed sample's raw AIS (the 237,318 positions of 565
vessels after the range, duplicate and jump checks, before the filter; one
core): a 30 s median of positions, speed, course and heading takes 0.4 s, moves
the median point by 0.0 m (95th percentile 25 m, maximum 305 m), and turns 2
segments into land crossings that the raw segment between the same two rows did
not have (22,666 segments of the raw tracks already intersect the sample's
coastline; 22,662 after filtering). It leaves 22,625
raw points on land and 22,621 after. Hourly station series are unchanged by a
30 s window: none of the 2,302 wind speed, air temperature and air pressure
observations of the sample's 16 stations changes.
`help(catraj.preprocessing.filtering)` has the decisions.

Compression then uses those columns. CADP runs Douglas–Peucker, then adds back
every point whose removal would misrepresent a context variable by more than
its threshold:

```python
from catraj.compression import compress, compression_ratio, dtw

small = compress(
    traj, epsilon=1_000,                          # metres, for Douglas–Peucker
    thresholds={"wind_speed": 0.5, "speed_over_ground": 1.0},
    time_column="unix_time", by="mmsi",           # or pass a movingpandas Trajectory
)                                                 # filtered first: filter_window=30
compression_ratio(len(small), len(traj)), dtw(traj, small)
```

`mode="post"` (the default) is the method as published: one pass against the
Douglas–Peucker result. `mode="iterative"` re-measures after each addition, so
every dropped point is guaranteed within its thresholds and far fewer points
are kept; `help(catraj.compression.cadp)` explains the difference.

CPLS also starts from Douglas–Peucker, but it protects topology instead of
context: it keeps any point whose removal would change the trajectory's
topological signature, as Tienaah et al. define it, through five relations,
all on by default. Two are about a barrier such as a shoreline: the DE-9IM
relation with each neighbouring barrier must stay the same (no crossing the
original did not have), and so must the nine-cell direction model (an island
stays on its side of the track). Three are about the track itself: it may not
gain a self-intersection (`avoid_self_intersection`), lose one
(`keep_self_intersection`), or fold back on itself (`inversion`). Barriers
are shapely geometries (lines or polygons) or a `GeoDataFrame`, in the same
projected CRS as the positions:

```python
from catraj.compression import cpls_compress

land = load_coastline()   # land polygons, from catraj.datasets
safe = cpls_compress(traj, epsilon=1_000, barriers=land, by="mmsi",
                     time_column="unix_time")
```

The result is always a superset of the Douglas–Peucker result, introduces no
crossing that the original trajectory did not already have, and leaves every
nearby barrier on the side it was;
`help(catraj.compression.cpls)` lists the decisions behind it.

The self-intersection and inversion relations act on the track alone, and on
raw AIS they are expensive: the GPS jitter of a moored vessel crosses itself
thousands of times, and keeping those crossings keeps most of its points. On
the sample's raw positions at 1 km, one trajectory per vessel, one core: the
barrier relations alone keep 2,457 of 237,318 points in 163 s (2,453 in 117 s
with the default 30 s filter, which is included in the time). All five
relations on the whole sample did not finish in 10 minutes, so compare them on
the first 60 vessels by MMSI (33,943 positions): barrier relations alone keep
280 points (278 filtered) in 1.1 s (1.2 s), all five 19,099 (16,804 filtered) in
13.3 s (12.9 s). The paper's two-stage CPLS collapses stay points first and
removes that jitter (below). Otherwise, switch relations off by keyword:

```python
safe = cpls_compress(traj, epsilon=1_000, barriers=land, by="mmsi",
                     time_column="unix_time", keep_self_intersection=False, avoid_self_intersection=False,
                     inversion=False)          # barrier relations only
```

2stage-CPLS is that two-stage variant. A stay (a vessel at anchor, found by
its AIS status, by Li et al.'s stay-point detection, or both) becomes two
vertices at its centroid, keeping the duration of the stop; Douglas–Peucker on
the speed series keeps the points where the speed changes; and CPLS then runs
on the collapsed track, so its topological signature is that of the track
with each stay replaced by its centroid. Time is read like CADP does, from the
index or `time_column`; the paper gives no stay thresholds, so none is assumed:

```python
from catraj.compression import two_stage_cpls_compress

small = two_stage_cpls_compress(
    traj, epsilon=1_000, speed_epsilon=1.54,      # metres; metres per second (3 knots)
    barriers=land, time_column="unix_time", by="mmsi",
    status_column="navigation",                   # status 1 and 5: at anchor, moored
    stay_radius=100, stay_duration=600,           # and 10 minutes within 100 m
)
```

The result is a row subset of the input, except that the geometry of each
stay's two kept rows is the centroid; a `stay` column marks them. On the
sample's raw positions (one trajectory per vessel, epsilon 1 km, 1.54 m/s, all
five relations, one core), counting a stay as a run of consecutive stay points:
status alone finds 56 stays (12,436 points) and movement detection added
(100 m, 10 minutes) 242 (122,746 points). Status alone is no help to the
compression, because most moored vessels do not report "at anchor": on the first
60 vessels by MMSI (33,943 positions) it keeps 19,282 points in 12.0 s, as many
as plain CPLS (19,099); with movement detection added, 3,120 in 2.8 s. On the
whole sample, with movement detection, 2stage-CPLS keeps 16,889 of 237,318
points in 35 s (`filter_window=None`) and 15,674 in 37 s with the default
filter.
`help(catraj.compression.two_stage)` lists the decisions behind it.

There is also an interactive app, which compares interpolation methods by
cross-validation before annotating:

```bash
python app/run_app.py       # or double-click app/"Run Annotation App.bat"
```

| | |
|---|---|
| [`docs/getting_the_data.md`](docs/getting_the_data.md) | how to download the public archive the notebooks use: one command, by hand, checksums, troubleshooting |
| [Notebooks](#notebooks) | the notebooks, in reading order, with Binder and nbviewer links |
| [`app/README.md`](app/README.md) | the app |
| [`scripts/README.md`](scripts/README.md) | scripts that process the whole archive end to end |

The modules carry the method in their docstrings: `help(catraj.annotation)`
for the overview, and `help` on any interpolator for what it does and when
it applies.

---

## Data

### Sample data, included

`data/sample/` holds a two-day extract of the archive, 2015-11-09 00:00 to
2015-11-11 00:00 UTC, in two forms: `raw/` (the archive's own CSV and shapefile
layout, 60 MB) and `preprocessed/` (Parquet, 9 MB). The notebooks, the app and
the tests read the preprocessed form with `catraj.datasets`:

| | | |
|---|---|---|
| AIS positions | 237,318 across 565 vessels, median-filtered (30 s) | Naval Academy, France — CC-BY-NC-SA-4.0 |
| Weather observations | 16 coastal stations, hourly (768) | rp5.ua / Met Office / NOAA |
| Sea state | 677,736 grid-cell rows, 3-hourly | SHOM – IFREMER, WAVEWATCH III — cite Boudière et al., 2013 |
| Coastline | 729 land polygons, clipped to the AIS area | European Environment Agency — free to re-use, acknowledge the source |

```python
from catraj.datasets import load_ais, load_weather, load_ocean, load_coastline

ais = load_ais()    # GeoDataFrame in EPSG:32630; unix_time, mmsi, ...
```

`scripts/make_sample.py` cut `raw/` from the full archive and
`scripts/prepare_zenodo.py` (that is, `catraj.preprocessing.preprocess_archive`)
made `preprocessed/`; the same call on the whole archive writes
`data/zenodo_1167595/preprocessed/` (not committed), which the loaders also read
(`load_ais("data/zenodo_1167595/preprocessed")`).

**These carry their own licences**, stricter than the code's — the AIS
extract is CC BY-NC-SA 4.0, so no commercial use. The coastline (EEA) may be
re-used freely, provided the source is acknowledged. See
[`data/LICENSE.md`](data/LICENSE.md). Nothing in `catraj` depends on
them; point it at your own data and the question does not arise.

### The full archive, downloaded

The sample is cut from a public six-month archive (October 2015 – March 2016)
on Zenodo, [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595)
(Ray et al., 2019). The full-archive runs of notebooks 02 to 05 use it, after
preprocessing. They cover October to December 2015 and February to March 2016:
**January 2016 is excluded everywhere**, because the archive's `oc_january.csv`
is a copy of `oc_november.csv`, so January's positions cannot be given a sea
state. The archive is not in the repository. Fetch and preprocess it with
notebook `01_preprocessing` (`DOWNLOAD = True`, `PREPROCESS_FULL = True`) or:

```bash
python scripts/download_data.py --months all  # ~825 MB download, ~5.3 GB extracted; or: annotation (~203 MB) | cadp (~771 MB) | cpls (~53 MB)
python scripts/download_data.py --check       # what is present
python scripts/prepare_zenodo.py --n-jobs -1  # writes data/zenodo_1167595/preprocessed/ (about 2 minutes)
```

The script uses only the standard library. It verifies each archive against
the checksum Zenodo publishes, resumes interrupted downloads, and extracts into
`data/zenodo_1167595/`, which git ignores.
[`docs/getting_the_data.md`](docs/getting_the_data.md)
covers everything else: which notebook needs which files, downloading by hand,
disk space, the order to run things in, troubleshooting, and the licences.
The AIS data is CC BY-NC-SA 4.0 (no commercial use), the sea state must be
cited, and the EEA coastline needs its source acknowledged.

---

## Design notes

**Context lives on the DataFrame**, as ordinary columns, so it survives the
slicing that compression performs.

**The heavy dependencies are optional.** The library needs only numpy, scipy,
pandas, geopandas and scikit-learn; the app and MovingPandas integration are
extras. A hard dependency on either would make the algorithms
un-upstreamable into MovingPandas.

**Where the papers are ambiguous**, the choice taken is stated in the
docstring of the code that makes it, rather than left implicit.

---

## Citation

Please cite the paper for the algorithm you used (see above), and
[MovingPandas](https://doi.org/10.1553/giscience2019_01_s54), which `catraj`
extends. `CITATION.cff` has both in machine-readable form.

## License

**Code:** BSD 3-Clause — the same licence as MovingPandas, so code may be
contributed upstream.

**Data:** third-party, with its own terms. See
[`data/LICENSE.md`](data/LICENSE.md) for the sample, and section 7
of [`getting_the_data.md`](docs/getting_the_data.md) for the
downloaded archive.
