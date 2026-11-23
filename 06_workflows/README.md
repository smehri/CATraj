# 06_workflows — whole-archive processing

Scripts that run a `catraj` method over a whole dataset end to end, reading
from and writing to `10_results/`. The tutorials explain the methods; these
apply them at scale. Run them from the repository root, in the `catraj`
environment.

| script | reads | writes |
|---|---|---|
| `01_cadp_on_annotated_fleet.py` | `10_results/01_trajectory_annotation/06_fleet_annotated.parquet`, every AIS position of the archive with the weather it met (from `05_tutorials/01_trajectory_annotation.ipynb`, `EXPORT_FLEET = True`) | `10_results/02_cadp_compression/`: the compressed fleet in both CADP modes, a per-vessel table and `summary.json` |

```bash
python 06_workflows/01_cadp_on_annotated_fleet.py
```

## 01 · CADP on the annotated fleet

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
