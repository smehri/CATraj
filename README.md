# catraj — context-aware trajectory processing

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
compression and 2stage-CPLS compression.** The prediction algorithm below is
being added in a later release.

---

## Algorithms

| Algorithm | What it does | Paper | Status |
|---|---|---|---|
| **Annotation** | Attaches context observed at coarse resolution to high-frequency trajectory points. Spatial interpolation across observation stations, then temporal interpolation through time. | 2023 | **implemented** |
| **CADP** | Context-Aware Douglas–Peucker. Extends DP with three context domains — environmental, internal, semantic — retaining points that carry contextual information a purely spatial criterion would discard. | 2025 | **implemented** |
| **CPLS** | Context-aware Piecewise Linear Segmentation. Guarantees logical consistency, e.g., a compressed (simplified) vessel trajectory may not cross the shoreline, and keeps its topological signature (barriers, direction, self-intersections). | 2021 | **implemented** |
| **2stage-CPLS** | CPLS plus stay-point preservation. | 2021 | **implemented** |
| **CLSTM** | Context-aware LSTM. Routes each trajectory to a per-type network, avoiding Simpson's paradox in a single global model. | 2021, 2023 | planned |

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
| [`src/catraj/`](src/catraj/) | the library: `annotation`, `preprocessing`, `compression` |
| [`tests/`](tests/) | the test suite (`pytest`) |
| [`notebooks/`](notebooks/) | the tutorials, in reading order |
| [`scripts/`](scripts/) | command-line scripts: the data downloader, whole-archive workflows, and `benchmarks/` (temporary research scripts, to be removed) |
| [`app/`](app/) | the interactive annotation app |
| [`data/`](data/) | all data: the 24-hour sample in `sample/`, committed; the downloaded archive in `zenodo_1167595/` and built files in `cache/`, not committed |
| `results/` | everything the notebooks, scripts and app write, not committed ([`README`](results/README.md)) |
| [`docs/`](docs/) | the source papers (open-access ones) and [`getting_the_data.md`](docs/getting_the_data.md) |

---

## Installation

Requires Python ≥ 3.10. The geospatial stack installs far more reliably from
conda-forge, especially on Windows:

```bash
conda create -n catraj -c conda-forge python=3.11 geopandas scipy scikit-learn pandas panel hvplot geoviews-core movingpandas stonesoup ipykernel pytest
conda activate catraj

git clone https://github.com/smehri/catraj
cd catraj
pip install -e ".[app]"
```

Omit `[app]` for the library alone. `stonesoup` is only an optional extra of
MovingPandas, which warns on import without it.

To run the tutorials from an editor (VS Code, JupyterLab) without activating
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
`evaluate_spatial`). The whole six-month sample archive, 18 million positions,
annotates in under a minute per variable on 32 cores (about 26 s for a linear
variable, 49 s for a bearing).

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
(`douglas_peucker`, `cadp_select`, `cpls_select`) filter nothing. The tutorials run
the algorithms unfiltered, with `filter_window=None`, and show the default result next
to it.

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

On the 24-hour sample it takes about 0.25 s, moves the median point by 0.0 m
(95th percentile 22 m) at 30 s, and creates no land crossing that the raw track
did not have there. Filtering the positions alone raises the number of points
2stage-CPLS keeps (5,228 raw, 12,006 after a 30 s median) because filtered
positions repeat; filtering its speed series too, as it does by default, brings
it to 4,759. `help(catraj.preprocessing.filtering)` has the table and the
decisions. Hourly station series are unchanged by a 30 s window.

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

land = gpd.read_file("data/sample/Coastline/Coastline_UTM.shp")   # land polygons
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
the 24-hour sample at 1 km, the barrier relations alone keep 607 of 106,894
points, all five 43,074 (about 40 s instead of 3 s), without filtering; with
the default 30 s filter, 604 and 35,020. The paper's two-stage
CPLS collapses stay points first and removes that jitter (below). Otherwise,
switch relations off by keyword:

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
24-hour sample (epsilon 1 km, 1.54 m/s, all five relations) status alone finds
19 stays and keeps 44,369 of 106,894 points, hardly fewer than CPLS, because
most moored vessels there do not report "at anchor"; adding movement detection
(100 m, 10 minutes: 98 stays) keeps 5,228 unfiltered (`filter_window=None`) and
4,759 with the default filter, in about 10 s.
`help(catraj.compression.two_stage)` lists the decisions behind it.

There is also an interactive app, which compares interpolation methods by
cross-validation before annotating:

```bash
python app/run_app.py       # or double-click app/"Run Annotation App.bat"
```

| | |
|---|---|
| [`docs/getting_the_data.md`](docs/getting_the_data.md) | how to download the public archive the tutorials use: one command, by hand, checksums, troubleshooting |
| [`notebooks/01_trajectory_annotation.ipynb`](notebooks/01_trajectory_annotation.ipynb) | annotation on real data, from the 24-hour sample up to the whole six-month archive, with interactive plots |
| [`notebooks/02_cadp_compression.ipynb`](notebooks/02_cadp_compression.ipynb) | CADP on the paper's data (AIS + WAVEWATCH III sea state, Feb 2016): download links and licences, both modes, the paper's experiments |
| [`notebooks/03_cpls_compression.ipynb`](notebooks/03_cpls_compression.ipynb) | CPLS on the 24-hour sample and a coastline: where Douglas–Peucker cuts across land, what CPLS adds to prevent it, the choice of ε. No download needed |
| [`app/README.md`](app/README.md) | the app |
| [`scripts/README.md`](scripts/README.md) | scripts that process the whole archive end to end |

The modules carry the method in their docstrings: `help(catraj.annotation)`
for the overview, and `help` on any interpolator for what it does and when
it applies.

---

## Data

### Sample data, included

`data/sample/` holds a matched 24-hour extract that the tutorial and app both use
out of the box:

| | | |
|---|---|---|
| AIS positions | 106,894 across 136 vessels | Naval Academy, France — CC-BY-NC-SA-4.0 |
| Weather observations | 16 coastal stations, hourly | rp5.ua / Met Office / NOAA |
| Coastline | 175 land polygons, clipped to the AIS area | European Environment Agency — free to re-use, acknowledge the source |

AIS and weather cover 2016-02-05 01:00 → 2016-02-06 01:00 UTC; all three layers
are projected to UTM zone 30N.

**These carry their own licences**, stricter than the code's — the AIS
extract is CC BY-NC-SA 4.0, so no commercial use. The coastline (EEA) may be
re-used freely, provided the source is acknowledged. See
[`data/LICENSE.md`](data/LICENSE.md). Nothing in `catraj` depends on
them; point it at your own data and the question does not arise.

### The full archive, downloaded

The sample is cut from a public six-month archive (October 2015 – March 2016)
on Zenodo, [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595)
(Ray et al., 2019). The full annotation run, the CADP tutorial, the workflows
and the benchmarks use it, and so does the optional full-coastline section of
the CPLS tutorial. It is not in the repository. Fetch it with:

```bash
python scripts/download_data.py              # ~825 MB; or: annotation (~203 MB) | cadp (~771 MB) | cpls (~53 MB)
python scripts/download_data.py --check      # what is present
```

The script uses only the standard library. It verifies each archive against
the checksum Zenodo publishes, resumes interrupted downloads, and extracts into
`data/zenodo_1167595/`, which git ignores.
[`docs/getting_the_data.md`](docs/getting_the_data.md)
covers everything else: which tutorial needs which files, downloading by hand,
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
