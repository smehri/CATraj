# scripts — command-line scripts

Everything here runs from the repository root, in the `catraj` environment
with the package installed (`pip install -e ".[dev]"`, see the main README).

| script | what it does |
|---|---|
| `download_data.py` | downloads the public Zenodo archive (AIS, weather, sea state, coastline) into `data/zenodo_1167595/`, verifying checksums; `--months all` for every sea-state month ([`docs/getting_the_data.md`](../docs/getting_the_data.md)) |
| `prepare_zenodo.py` | run after the download: `catraj.preprocessing.preprocess_archive` on the whole archive. Renames the coded columns, sets "not available" codes to NaN, checks ranges, projects to EPSG:32630, median-filters the columns the notebooks use, writes `data/zenodo_1167595/preprocessed/` (Parquet; needs `pyarrow`; `--n-jobs -1` uses every core; `--data` and `--out` choose other folders, as for the sample) |
| `make_sample.py` | cuts the committed two-day sample `data/sample/raw/` from the downloaded archive (the window, area and sea-state month are options) |
| `cadp_on_annotated_fleet.py` | an earlier, standalone run of CADP over an annotated fleet (below) |

The notebooks do the same work with explanations: `01_preprocessing` for the first two, and `04_cadp_compression` for the last.

## `cadp_on_annotated_fleet.py`

An earlier, standalone run of CADP over a whole annotated fleet, kept as a
script. **Notebook `04_cadp_compression` replaces it** for the current
workflow: it annotates the preprocessed AIS with the sea state itself,
compresses every trip, and works on the sample and on the full archive. The
script reads `results/01_trajectory_annotation/06_fleet_annotated.parquet`,
which only an earlier version of the annotation notebook wrote (the current
`notebooks/02_trajectory_annotation.ipynb` writes
`results/02_trajectory_annotation/<DATA>/annotated_ais.parquet`, with other
columns), so it does not run on a fresh checkout. It writes
`results/02_cadp_compression/`.

Each vessel is one trajectory. Settings are constants at the top of the
script, and every run records them in `summary.json`:

| setting | value |
|---|---|
| spatial threshold ε | 1 km |
| wind speed | 1.0 m/s |
| air temperature | 0.5 °C |
| air pressure | 1.0 mm of mercury |
| wind direction (a bearing) | 10° |
| speed over ground (internal context) | 1.0 kn |

Both modes are run. `post` is the method as published; `iterative`
re-measures after each addition, so no dropped point exceeds its thresholds
against its final neighbours. Each retained point carries `kept_by`: `dp` if
Douglas–Peucker keeps it, otherwise the variable(s) that required it.
`summary.json` also reports, for every variable, how far dropped points lie
from the compressed trajectory (RMSE and maximum), which is the number to
check against the threshold.
