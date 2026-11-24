# scripts — command-line scripts

Everything here runs from the repository root, in the `catraj` environment
with the package installed (`pip install -e .`, see the main README).

| script | what it does |
|---|---|
| `download_data.py` | downloads the public Zenodo archive (AIS, weather, sea state, coastline) into `data/zenodo_1167595/`, verifying checksums ([`docs/getting_the_data.md`](../docs/getting_the_data.md)) |
| `prepare_zenodo.py` | run after the download: renames the archive's coded columns, sets "not available" codes to NaN, median-filters the columns the notebooks use, writes `data/zenodo_1167595/clean/*.parquet` (needs `pyarrow`) |
| `cadp_on_annotated_fleet.py` | runs CADP over the whole annotated fleet (below) |
| [`benchmarks/`](benchmarks/) | temporary research scripts behind the CADP paper's experiments, to be removed |

## Whole-archive processing

`cadp_on_annotated_fleet.py` runs a `catraj` method over a whole dataset end
to end, reading from and writing to `results/`. The notebooks explain the
methods; this applies them at scale.

| script | reads | writes |
|---|---|---|
| `cadp_on_annotated_fleet.py` | `results/01_trajectory_annotation/06_fleet_annotated.parquet`, every AIS position of the archive with the weather it met (from `notebooks/01_trajectory_annotation.ipynb`, `EXPORT_FLEET = True`) | `results/02_cadp_compression/`: the compressed fleet in both CADP modes, a per-vessel table and `summary.json` |

```bash
python scripts/download_data.py annotation     # once: the Zenodo archive (~203 MB)
python scripts/prepare_zenodo.py               # once: cleaned Parquet files for the notebooks
# run notebooks/01_trajectory_annotation.ipynb with the full archive and EXPORT_FLEET = True
python scripts/cadp_on_annotated_fleet.py
```

The data, and the order to run things in, are explained in
[`docs/getting_the_data.md`](../docs/getting_the_data.md).

### CADP on the annotated fleet

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
