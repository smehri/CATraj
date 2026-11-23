# scripts/benchmarks/ — temporary research scripts

> **Not for regular use, and not part of `catraj`.**
> These scripts were written while developing CADP, to evaluate the two
> compression modes against each other. They are kept only until that
> evaluation is finished, and **this folder will be removed** in a later
> release. Nothing in the library, the app or the tutorials depends on it.
>
> To *use* CADP, see [`notebooks/02_cadp_compression.ipynb`](../../notebooks/02_cadp_compression.ipynb)
> and `help(catraj.compression)`.

## What is here

| script | what it does | data it needs |
|---|---|---|
| `prepare_feb2016.py` | Builds `data/cache/cadp_feb2016.pkl`: February 2016 AIS, de-duplicated and projected to UTM 30N, annotated with bottom depth, wave height, wave direction and sea level (IDW over the 4 nearest grid cells, DLT in time, vectorised; checked against `catraj.annotation.annotate_points`). | Zenodo archive, below |
| `paper_experiments.py` | Runs CADP on that file in both modes: threshold sweeps for wave direction, wave height and bottom depth (ε = 1 km); both modes and plain Douglas–Peucker tuned to keep the same number of points; runtime at ε = 10 m. | `data/cache/cadp_feb2016.pkl` |
| `equal_size_sample_data.py` | The same equal-size comparison on the 24-hour sample in `data/sample/`, over every usable context variable (weather annotated onto AIS, plus speed, course, heading and rate of turn), with a sensitivity check on ε. | `data/sample/` only |

"Equal size" means each method's parameter is tuned until all of them keep
the same number of points, so they are compared on *which* points they keep
rather than on *how many*. This is an evaluation device, not something a user
of CADP needs.

## Running

From the repository root, in the `catraj` environment:

```bash
python scripts/benchmarks/equal_size_sample_data.py      # ~2-3 min, no download needed

python scripts/benchmarks/prepare_feb2016.py             # ~30 s, needs the Zenodo files
python scripts/benchmarks/paper_experiments.py           # ~3 min
```

The Zenodo files are the ones the CADP tutorial uses. Get them with
`python scripts/download_data.py cadp` (or see
[`docs/getting_the_data.md`](../../docs/getting_the_data.md)),
so that these exist:

```
data/zenodo_1167595/P1_AIS_Data/nari_dynamic.csv
data/zenodo_1167595/E1_ocean/oc_february.csv
```

## Data and licences

* Ray, C., Dréo, R., Camossi, E., Jousselme, A.-L., Iphar, C. (2019).
  *Heterogeneous integrated dataset for Maritime Intelligence, Surveillance,
  and Reconnaissance.* Data in Brief 25, 104141.
  [doi:10.1016/j.dib.2019.104141](https://doi.org/10.1016/j.dib.2019.104141) ·
  Zenodo [doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595)
* AIS: **CC BY-NC-SA 4.0**. Attribute the Naval Academy Research Institute
  (NARI), no commercial use, and derived data under the same licence.
* Sea state (SHOM – IFREMER, WAVEWATCH III): public, citation mandatory:
  Boudière, E. et al. (2013). International Journal of Marine Energy 3–4,
  e40–e52. [doi:10.1016/j.ijome.2013.11.010](https://doi.org/10.1016/j.ijome.2013.11.010)

Data stays in `data/zenodo_1167595/` and `data/cache/`, which git ignores.
