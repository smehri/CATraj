# Trajectory annotation — interactive app

A point-and-click front end to `catraj.annotation`: load trajectories and
context observations, compare interpolation methods by cross-validation,
annotate, and export.

Any moving object will do — vessels, aircraft, vehicles, wildlife, pedestrians
— and any context measured at fixed stations. The sample data is maritime; the
method is not.

## Running it

```bash
conda activate catraj
python app/run_app.py
```

That opens <http://localhost:5006/annotation_app> in your browser.

Options:

```bash
python app/run_app.py --port 5010      # a different port
python app/run_app.py --no-show        # do not open a browser
python app/run_app.py --address 0.0.0.0  # reachable from other machines
```

Or serve it directly, which is the same thing:

```bash
panel serve app/annotation_app.py --show
```

> On Windows, set `MKL_THREADING_LAYER=SEQUENTIAL` first if you serve it
> directly. Some Intel MKL builds abort the process the first time a LAPACK
> routine runs otherwise. `run_app.py` does this for you.

## The tabs

| | |
|---|---|
| **Overview** | The method as a diagram, with buttons into each stage |
| **1 · Data** | Load the two files and map the columns |
| **2 · Spatial** | Compare methods by leave-one-station-out; inspect surfaces |
| **3 · Temporal** | Compare methods by k-fold |
| **4 · Annotate** | Apply the chosen methods and export |

## A first run

1. **Data** — the sample paths are pre-filled. Click **Load**.
2. **Spatial** — pick a variable, choose methods, **Run cross-validation**.
3. **Temporal** — same, for the temporal stage.
4. **Annotate** — the recommendation table shows each variable's winner. Run,
   then export.

Steps 2 and 3 are optional. Annotating without them warns you which variables
were never compared and offers to continue anyway — if you already know which
methods suit your data, there is no reason to spend the time.

## Field descriptions are optional

Column names like `Ff` and `id_windDir` say nothing on their own. Point the two
optional inputs on the Data tab at a description file and they become
`Ff — Mean wind speed (meters per seconds)` throughout the app.

The format is the one used by the sample archives:

```
Attribute       Data type           Description
------------------------------------------------------------------
*  Ff           integer             Mean wind speed ... (meters per seconds)
*  Td           double precision    Dewpoint temperature ... (°C)
```

Leave the fields blank when the names already mean something.

## What the app will not decide for you

Two choices belong to whoever knows the data:

- **Which variables are bearings.** Nothing is inferred from a column name.
  Wind and wave *direction* are angular: interpolated as ordinary numbers,
  350° and 10° average to 180° — the opposite heading.
- **Which method to use when RMSE and MAE disagree.** That is a judgement about
  whether rare large errors matter more than typical ones. The app reports both
  winners and leaves the choice to you.

## Export

Writes to whatever folder you name — a GeoPackage with geometry, a CSV without,
and the cross-validation score tables behind the choice. The default is
`results/03_app_exports/`; a relative path is taken from the repository root,
and the folder is created if it does not exist.

Annotated columns are renamed if they would collide with an existing column,
ignoring case — GeoPackage column names are case-insensitive. The AIS file has
a timestamp column `t`, so annotating the weather variable `T` (air
temperature) writes `T_ctx` rather than clashing with it.
