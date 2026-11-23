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

**This release implements trajectory annotation and CADP compression.** The
other compression and prediction algorithms below are being added in later
releases.

---

## Algorithms

| Algorithm | What it does | Paper | Status |
|---|---|---|---|
| **Annotation** | Attaches context observed at coarse resolution to high-frequency trajectory points. Spatial interpolation across observation stations, then temporal interpolation through time. | 2023 | **implemented** |
| **CADP** | Context-Aware Douglas–Peucker. Extends DP with three context domains — environmental, internal, semantic — retaining points that carry contextual information a purely spatial criterion would discard. | 2025 | **implemented** |
| **CPLS** | Context-aware Piecewise Linear Segmentation. Guarantees logical consistency, e.g., a compressed (simplified) vessel trajectory may not cross the shoreline. | 2021 | planned |
| **2stage-CPLS** | CPLS plus stay-point preservation. | 2021 | planned |
| **CLSTM** | Context-aware LSTM. Routes each trajectory to a per-type network, avoiding Simpson's paradox in a single global model. | 2021, 2023 | planned |

---

## Papers

The two open-access papers are in [`01_docs/papers/`](01_docs/papers/), so each module can be
checked against the method it implements; the 2023 paper is available from its
publisher via the DOI. The implementation does not depend on having them.

| Year | Paper | | |
|---|---|---|---|
| 2025 | *A Novel Context-Aware Douglas–Peucker (CADP) Trajectory Compression Method.* ISPRS Int. J. Geo-Inf., 14(2), 58. | [pdf](01_docs/papers/2025_Mehri_IJGI_CADP.pdf) | [doi](https://doi.org/10.3390/ijgi14020058) |
| 2023 | *A context-aware approach for vessels' trajectory prediction.* Ocean Engineering, 282, 114916. | — | [doi](https://doi.org/10.1016/j.oceaneng.2023.114916) |
| 2021 | *A Contextual Hybrid Model for Vessel Movement Prediction.* IEEE Access, 9, 45600–45613. | [pdf](01_docs/papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf) | [doi](https://doi.org/10.1109/ACCESS.2021.3066463) |

---

## Repository layout

| folder | what it holds |
|---|---|
| [`01_docs/`](01_docs/) | the source papers (open-access ones) |
| [`02_src/catraj/`](02_src/catraj/) | the library: `annotation`, `compression` |
| [`03_tests/`](03_tests/) | the test suite (`pytest`) |
| [`04_app/`](04_app/) | the interactive annotation app |
| [`05_tutorials/`](05_tutorials/) | notebooks, in order, and the data guide and downloader (`00_getting_the_data.md`, `download_data.py`) |
| [`06_workflows/`](06_workflows/) | scripts that run a method over the whole archive |
| [`07_benchmarks/`](07_benchmarks/) | temporary research scripts, to be removed |
| [`08_data/`](08_data/) | all data ([`README`](08_data/README.md)): the 24-hour sample in `sample/`, committed; the downloaded archive in `zenodo_1167595/` and built files in `cache/`, not committed (how to get them: [`00_getting_the_data.md`](05_tutorials/00_getting_the_data.md)) |
| `10_results/` | everything the notebooks and workflows write, not committed ([`README`](10_results/README.md)) |

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
annotates in under half a minute per variable on 32 cores.

Context attaches as ordinary columns, so it survives the `.iloc` slicing that
trajectory compression performs — a compressed trajectory keeps the context of
the points it retains.

Compression then uses those columns. CADP runs Douglas–Peucker, then adds back
every point whose removal would misrepresent a context variable by more than
its threshold:

```python
from catraj.compression import compress, compression_ratio, dtw

small = compress(
    traj, epsilon=1_000,                          # metres, for Douglas–Peucker
    thresholds={"wind_speed": 0.5, "speed_over_ground": 1.0},
    time_column="unix_time", by="mmsi",           # or pass a movingpandas Trajectory
)
compression_ratio(len(small), len(traj)), dtw(traj, small)
```

`mode="post"` (the default) is the method as published: one pass against the
Douglas–Peucker result. `mode="iterative"` re-measures after each addition, so
every dropped point is guaranteed within its thresholds and far fewer points
are kept; `help(catraj.compression.cadp)` explains the difference.

There is also an interactive app, which compares interpolation methods by
cross-validation before annotating:

```bash
python 04_app/run_app.py       # or double-click 04_app/"Run Annotation App.bat"
```

| | |
|---|---|
| [`05_tutorials/00_getting_the_data.md`](05_tutorials/00_getting_the_data.md) | how to download the public archive the tutorials use: one command, by hand, checksums, troubleshooting |
| [`05_tutorials/01_trajectory_annotation.ipynb`](05_tutorials/01_trajectory_annotation.ipynb) | annotation on real data, from the 24-hour sample up to the whole six-month archive, with interactive plots |
| [`05_tutorials/02_cadp_compression.ipynb`](05_tutorials/02_cadp_compression.ipynb) | CADP on the paper's data (AIS + WAVEWATCH III sea state, Feb 2016): download links and licences, both modes, the paper's experiments |
| [`04_app/README.md`](04_app/README.md) | the app |
| [`06_workflows/README.md`](06_workflows/README.md) | scripts that process the whole archive end to end |

The modules carry the method in their docstrings: `help(catraj.annotation)`
for the overview, and `help` on any interpolator for what it does and when
it applies.

---

## Data

### Sample data, included

`08_data/sample/` holds a matched 24-hour extract that the tutorial and app both use
out of the box:

| | | |
|---|---|---|
| AIS positions | 106,894 across 136 vessels | Naval Academy, France — CC-BY-NC-SA-4.0 |
| Weather observations | 16 coastal stations, hourly | rp5.ua / Met Office / NOAA |

Both cover 2016-02-05 01:00 → 2016-02-06 01:00 UTC, projected to UTM zone 30N.

**These carry their own licences**, stricter than the code's — the AIS
extract is CC BY-NC-SA 4.0, so no commercial use. See
[`08_data/LICENSE.md`](08_data/LICENSE.md). Nothing in `catraj` depends on
them; point it at your own data and the question does not arise.

### The full archive, downloaded

The sample is cut from a public six-month archive (October 2015 – March 2016)
on Zenodo, [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595)
(Ray et al., 2019). The full annotation run, the CADP tutorial, the workflows
and the benchmarks use it. It is not in the repository. Fetch it with:

```bash
python 05_tutorials/download_data.py              # ~773 MB; or: annotation (~203 MB) | cadp (~771 MB)
python 05_tutorials/download_data.py --check      # what is present
```

The script uses only the standard library. It verifies each archive against
the checksum Zenodo publishes, resumes interrupted downloads, and extracts into
`08_data/zenodo_1167595/`, which git ignores.
[`05_tutorials/00_getting_the_data.md`](05_tutorials/00_getting_the_data.md)
covers everything else: which tutorial needs which files, downloading by hand,
disk space, the order to run things in, troubleshooting, and the licences.
The AIS data is CC BY-NC-SA 4.0 (no commercial use), and the sea state must be
cited.

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
[`08_data/LICENSE.md`](08_data/LICENSE.md) for the sample, and section 7
of [`00_getting_the_data.md`](05_tutorials/00_getting_the_data.md) for the
downloaded archive.
