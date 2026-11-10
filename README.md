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

**This release implements trajectory annotation.** The compression and
prediction algorithms below are being added in later releases, each with its own
app and tutorial.

---

## Algorithms

| Algorithm | What it does | Paper | Status |
|---|---|---|---|
| **Annotation** | Attaches context observed at coarse resolution to high-frequency trajectory points. Spatial interpolation across observation stations, then temporal interpolation through time. | 2023 | **implemented** |
| **CADP** | Context-Aware Douglas–Peucker. Extends DP with three context domains — environmental, internal, semantic — retaining points that carry contextual information a purely spatial criterion would discard. | 2025 | planned |
| **CPLS** | Context-aware Piecewise Linear Segmentation. Guarantees logical consistency, e.g., a compressed (simplified) vessel trajectory may not cross the shoreline. | 2021 | planned |
| **2stage-CPLS** | CPLS plus stay-point preservation. | 2021 | planned |
| **CLSTM** | Context-aware LSTM. Routes each trajectory to a per-type network, avoiding Simpson's paradox in a single global model. | 2021, 2023 | planned |

---

## Papers

The two open-access papers are in [`papers/`](papers/), so each module can be
checked against the method it implements; the 2023 paper is available from its
publisher via the DOI. The implementation does not depend on having them.

| Year | Paper | | |
|---|---|---|---|
| 2025 | *A Novel Context-Aware Douglas–Peucker (CADP) Trajectory Compression Method.* ISPRS Int. J. Geo-Inf., 14(2), 58. | [pdf](papers/2025_Mehri_IJGI_CADP.pdf) | [doi](https://doi.org/10.3390/ijgi14020058) |
| 2023 | *A context-aware approach for vessels' trajectory prediction.* Ocean Engineering, 282, 114916. | — | [doi](https://doi.org/10.1016/j.oceaneng.2023.114916) |
| 2021 | *A Contextual Hybrid Model for Vessel Movement Prediction.* IEEE Access, 9, 45600–45613. | [pdf](papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf) | [doi](https://doi.org/10.1109/ACCESS.2021.3066463) |

---

## Installation

Requires Python ≥ 3.10. The geospatial stack installs far more reliably from
conda-forge, especially on Windows:

```bash
conda create -n catraj -c conda-forge python=3.11 geopandas scipy scikit-learn pandas panel hvplot geoviews-core movingpandas pytest
conda activate catraj

git clone https://github.com/smehri/catraj
cd catraj
pip install -e ".[app]"
```

Omit `[app]` for the library alone.

---

## Using it

```python
import geopandas as gpd, pandas as pd
from catraj.annotation import IDW, Spline, annotate_points

obs = gpd.read_file("weather.shp")
stations = pd.DataFrame({
    "x": obs.geometry.x, "y": obs.geometry.y,
    "t": obs.local_time.astype(float),
    "value": obs.Ff.astype(float),
})

traj = gpd.read_file("trajectories.shp")
traj["wind_speed"] = annotate_points(
    stations,
    traj.geometry.x, traj.geometry.y, traj.t.astype(float),
    spatial_method=IDW(power=2), temporal_method=Spline(),
)
```

Context attaches as ordinary columns, so it survives the `.iloc` slicing that
trajectory compression performs — a compressed trajectory keeps the context of
the points it retains.

There is also an interactive app, which compares interpolation methods by
cross-validation before annotating:

```bash
python app/run_app.py          # or double-click "Run Annotation App.bat"
```

| | |
|---|---|
| [`tutorials/1-trajectory-annotation.ipynb`](tutorials/1-trajectory-annotation.ipynb) | a worked example on real data, with interactive plots |
| [`app/README.md`](app/README.md) | the app |

The modules carry the method in their docstrings: `help(catraj.annotation)`
for the overview, and `help` on any interpolator for what it does and when
it applies.

---

## Sample data

`datasets/` holds a matched 24-hour extract that the tutorial and app both use
out of the box:

| | | |
|---|---|---|
| AIS positions | 106,894 across 136 vessels | Naval Academy, France — CC-BY-NC-SA-4.0 |
| Weather observations | 16 coastal stations, hourly | rp5.ua / Met Office / NOAA |

Both cover 2016-02-05 01:00 → 2016-02-06 01:00 UTC, projected to UTM zone 30N.

**These carry their own licences**, stricter than the code's — the AIS
extract is CC BY-NC-SA 4.0, so no commercial use. See
[`datasets/LICENSE.md`](datasets/LICENSE.md). Nothing in `catraj` depends on
them; point it at your own data and the question does not arise.

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

**Sample data:** third-party, with its own terms. See
[`datasets/LICENSE.md`](datasets/LICENSE.md).
