# Phase 1: implement CPLS in `catraj` — implementation brief

You are implementing one algorithm, **CPLS (Context-aware Piecewise Linear
Segmentation)**, in an existing, tested, public Python library. This brief is
complete: the method, every design decision, the API, the tests, the data, the
documentation changes, the order of work and how to check each step.

Read the whole brief before you change anything. Where it says **binding**, do
not deviate; if you think a binding item is wrong, stop and ask (section 3).

Everything stated as a fact here was verified on this repository on
2026-11-23, at commit `46c024a` on `main`.

Contents: 1 Mission · 2 The repository · 3 Rules · 4 The method · 5 API ·
6 Implementation · 7 Data · 8 Tests · 9 Notebook · 10 Documentation ·
11 Work plan with gates · 12 Acceptance checklist · 13 Final report ·
Appendix (verified numbers)

---

## 1. Mission and scope

`catraj` implements context-aware trajectory algorithms from three papers.
Annotation and CADP compression are done. CPLS is next. It comes from:

> Mehri, Alesheikh & Basiri (2021). *A Contextual Hybrid Model for Vessel
> Movement Prediction.* IEEE Access 9, 45600–45613.
> `docs/papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf`,
> section IV-A, Algorithm 1 (page 45605).

**CPLS in one sentence:** Douglas–Peucker, plus a second stage that refuses to
drop a point when dropping it would make the simplified trajectory cross a
barrier (a shoreline) that the original did not cross.

**In scope**

1. `src/catraj/compression/cpls.py` and its exports.
2. `tests/test_cpls.py`, plus one test in `tests/test_sample_data.py`.
3. A committed sample coastline in `data/sample/Coastline/`, and the coastline
   archive added to `scripts/download_data.py`.
4. `notebooks/03_cpls_compression.ipynb`, executed, with outputs.
5. Documentation updates listed in section 10.

**Out of scope — do not do these**

- 2stage-CPLS (Algorithm 2), CLSTM, or anything in `catraj.annotation` or `app/`.
- Any change to `cadp.py`, `douglas_peucker.py` or `evaluation.py`.
- Version bump, git tag, release, PyPI, `git push`.
- Removing `scripts/benchmarks/`, reformatting existing files, renaming anything.
- New dependencies.

---

## 2. The repository

### Layout

| path | what it holds |
|---|---|
| `src/catraj/compression/` | `douglas_peucker.py`, `cadp.py`, `evaluation.py`, `__init__.py` — **read all four first**; CPLS must look like it belongs next to them |
| `tests/` | pytest suite, 180 tests, all passing. `helpers.py` has `DATASETS = <root>/data/sample`. Tests import it with `from helpers import DATASETS` |
| `notebooks/` | `01_trajectory_annotation.ipynb`, `02_cadp_compression.ipynb`, committed with outputs |
| `scripts/download_data.py` | standard-library-only downloader for Zenodo record 1167595 |
| `data/sample/` | committed 24-hour sample: `AIS/`, `Weather/` (shapefiles, EPSG:32630) |
| `data/zenodo_1167595/`, `data/cache/`, `results/` | not committed (git-ignored) |
| `docs/` | `getting_the_data.md`, `papers/` |
| `.github/workflows/tests.yml` | CI: `pip install -e ".[test,movingpandas]"` then `pytest`, on Python 3.10, 3.11, 3.12 (Linux) and 3.12 (Windows) |

### Environment

- Windows 11. A conda environment named `catraj` has everything installed:
  Python 3.11, shapely 2.2 (GEOS 3.14), geopandas 1.2, numpy 2.4, pandas 3.1,
  movingpandas, hvplot, geoviews, nbformat, nbconvert.
- `ruff` and `black` are **not** installed. Do not run a formatter. Match the
  surrounding style by hand; keep lines in `src/` to 88 characters.
- Run everything from the repository root through the environment:

  ```bash
  conda run -n catraj python -m pytest -q          # baseline: 180 passed
  conda run -n catraj python some_script.py
  ```

- `conda run` rejects a multi-line `python -c "..."`. For anything longer than
  one line, write a script file in a temporary folder outside the repository
  and run that.
- `pdftotext` is available in Git Bash if you want the paper as text:
  `pdftotext -layout docs/papers/2021_*.pdf <temp>/paper.txt`.

### Compatibility you must keep

CI installs the newest releases, but `pyproject.toml` declares minimums, and
the code must respect them:

- Python **3.10** syntax and standard library only.
- shapely **≥ 2.0**: use the module-level vectorised functions
  (`shapely.linestrings`, `shapely.intersects`, `shapely.prepare`,
  `shapely.get_parts`, `shapely.box`, `shapely.STRtree`). Avoid geometry
  methods added in 2.1+ (for example `geom.segmentize(...)`; the function
  `shapely.segmentize(geom, ...)` is fine).
- geopandas **≥ 0.13**: avoid 1.0-only API in `src/` and `tests/` (for example
  `GeoSeries.union_all()`; use `shapely.union_all(series.values)`).
- numpy ≥ 1.24, pandas ≥ 2.0.

### House style (taken from the existing code)

- Every module opens with a long docstring that states the method, cites the
  paper section, and lists **every decision taken where the paper leaves
  room**. `cadp.py` and `douglas_peucker.py` are the models. This is a stated
  design principle of the project (README, "Design notes").
- NumPy-style docstrings (`Parameters`, `Returns`). `from __future__ import
  annotations`. An explicit `__all__`.
- Array-level function first (`cadp_select`), dataframe-level wrapper second
  (`compress`). Results are boolean masks; dataframes are subset with `.iloc`
  so every column survives.
- Tests are short, named as sentences
  (`test_cadp_adds_the_point_where_context_jumps`), use synthetic data with a
  known answer, and compare optimised code against a slow obvious reference
  written inside the test file (`_dtw_brute`, `_context_dp`).
- Comments explain why, not what. No commented-out code, no TODOs.

---

## 3. Rules

### Git

- Work on `main`, in the existing working tree. Commit locally, once per
  finished step of section 11. **Never push, tag, amend, rebase or force.**
- Use plain `git commit -m ...`. The repository's git config supplies the
  author. Never pass `-c user.name`, `-c user.email` or `--author`.
- **No AI attribution anywhere**: no `Co-Authored-By` trailer, no "generated
  with" line, in commits, code, docs or the notebook. This is the owner's rule.
- Commit subjects are imperative sentences, like the existing ones:
  `Add CADP trajectory compression (catraj.compression)`.
- Stage files by name. **Never `git add -A` or `git add .`**
- Do not touch the branch `backup/before-identity-rewrite` or the tag
  `backup/github-main-before-force-push`.
- Do not commit, edit, move or delete this brief (`CPLS_PHASE1_GUIDE.md`). It
  is excluded locally through `.git/info/exclude`, so `git status` does not
  list it.

### Files

- Do not delete or re-download anything in `data/zenodo_1167595/`,
  `data/cache/` or `results/`. They hold over 1 GB of downloads and hours of
  computed output.
- Do not modify an existing test. If an existing test fails after your change,
  your change is wrong.
- Temporary scripts and outputs go in a temporary folder outside the
  repository, never in the working tree.

### Stop and ask the owner when

- you want to deviate from a **binding** item in sections 4 or 5;
- an existing test fails and the cause is not your new code;
- your numbers on the sample data fall outside the ranges in section 12;
- a licence question comes up that section 7 does not answer;
- you would need a new dependency, or to change `cadp.py`,
  `douglas_peucker.py` or `evaluation.py`.

When you stop, say what you found, what you tried, and what you need decided.
Do not work around the problem.

### Honesty

- Report results as they are. If a test fails, show the output. If you skipped
  a step, say so.
- In docstrings, write "decided here" for the decisions in section 4. Do **not**
  write "confirmed with the author": that phrase in `cadp.py` records a real
  confirmation, and these have not had one yet.

---

## 4. The method (binding)

### What the paper says

Algorithm 1, transcribed (Θd is the offset-distance threshold, `epsilon` here):

```
Input: trajectory T, offset distance Θd
 1  Build topological relations for trajectory T
 2  dmax = 0
 3  imax = 0
 4  for i = 1 to (n−1)
 5      d = E⊥(Ti, T1, Tn)            # distance from Ti to the segment T1–Tn (Eq. 4)
 6      if d > dmax
 7          imax = 1
 8          dmax = d
 9  if dmax > Θd then
10      A = CPLS(T(1, imax), Θd)
11      B = CPLS(T(imax, n), Θd)
12      Tc = A, B(2, n)
13  else
14      for j = 2 to (n−1)
15          Tc = [T1, …, Tj−1, Tj+1, …, Tn]      # Tj is removed
16          if topological signature is consistent then
17              j++
18              goto line 14
19          else
20              add vertex Tj to Tc
21  return Tc
```

And the text around it:

> "To maintain logical consistency, the topological signature of the original
> and simplified trajectory should be the same. […] to compare the topological
> signature of the original and simplified trajectories, the proposed method
> checks the intersection of the trajectories segments and the coastlines."
>
> "When there is no error greater than the given threshold, next for all other
> trajectory vertices p2, …, p(n−1), it eliminates the first point p2. After
> that, it checks the consistency of the topological signature. If removing p2
> destroys the topological signature, it adds it to the trajectory, again.
> Otherwise, it eliminates p2 and continues the algorithm with the next point."

Read section IV-A of the PDF yourself to confirm this transcription before
you write the module docstring.

### The method as implemented

Lines 2–12 are exactly Douglas–Peucker. Lines 13–20 run once inside each
segment that Douglas–Peucker would have collapsed. So:

> **CPLS = Douglas–Peucker, then a topological pass inside each DP segment.**
> The result is always a superset of the DP result. This mirrors CADP
> ("DP first, then add points back"), which is why it fits this package.

Notation: points `P[0] … P[n-1]`; `B` is the set of barrier geometries.

- `hit(i, j)` — the straight segment `P[i]–P[j]` intersects `B`.
- `orig(i, j)` — the original sub-path from `i` to `j` intersects `B`:
  `hit(k, k+1)` is true for some `i ≤ k < j`.
- `consistent(i, j)` — `not hit(i, j) or orig(i, j)`.

```
dp = douglas_peucker(x, y, epsilon)            # stage 1, reuse the existing function
keep = dp.copy()
for each pair (s, e) of consecutive DP-kept indices with e - s >= 2:
    a = s                                      # last retained point
    for j = s+1, …, e-1:                       # stage 2, front to back
        if consistent(a, j+1):
            drop j                             # P[a]–P[j+1] replaces the path through j
        else:
            keep[j] = True
            a = j
return keep
```

### Decisions (binding; each one goes in the module docstring)

| # | Decision | Reason |
|---|---|---|
| D1 | Stage 1 **calls** `douglas_peucker`; do not reimplement it. Line 7 `imax = 1` is read as `imax = i`, and line 4's range as the interior points. | Same typo and same reading as the CADP paper, already documented in `douglas_peucker.py`. |
| D2 | Stage 2 is the literal greedy of lines 14–20: front to back, one point at a time, testing the segment from the last retained point to the point **after** the candidate. | "Tc = [T1 … Tj−1, Tj+1 … Tn]" still contains Tj+1 when the test runs. |
| D3 | "Topological signature is consistent" is an **intersection test** between the new segment and the barriers, with GEOS `intersects`. A segment that only touches a barrier counts as intersecting. | The paper: "checks the intersection of the trajectories segments and the coastlines". Touching is the conservative reading. |
| D4 | **Existing crossings are tolerated.** A new segment may intersect a barrier only if the original sub-path it replaces already does (`consistent` above). | The paper requires the *same* signature for original and simplified, not a crossing-free result. Real data has positions on the mapped land (section 7: 10 original segments in the sample). A strict rule would freeze every such stretch. |
| D5 | Sidedness is **not** checked. If the original passes north of a small island and the new segment passes south without touching it, CPLS accepts it. | The paper tests intersection only. State this as a limitation. |
| D6 | The `epsilon` bound holds against the **DP segment**, as in plain DP. Once stage 2 retains extra points, a dropped point is not guaranteed to lie within `epsilon` of its final neighbours' segment. | A consequence of the published greedy. State it; do not "fix" it. |
| D7 | The greedy is **not minimal**: it can retain a point even when the DP segment itself is consistent, because an intermediate chord was not. | Literal reading of lines 14–20. State it. |
| D8 | Barriers and positions must be in the **same projected CRS**. A mismatch raises; nothing is reprojected silently. | Matches `compress`, which refuses geographic coordinates. |
| D9 | Repeated positions give zero-length segments; they are tested as they are. GEOS treats one as a point (verified: inside a polygon → hit, on its boundary → hit, outside → no hit). | No special case needed. |

### What the result guarantees (each one is a test in section 8)

- G1. `keep ⊇ dp`. With no barriers, or none near the trajectory, `keep == dp`.
- G2. For every pair of consecutive retained points `(a, b)`: `consistent(a, b)`.
  In words: **compression never introduces a crossing.**
- G3. If the original never intersects a barrier, neither does the result.
- G4. The first and last points are retained.
- G5. The result equals the literal reference implementation of section 8.

---

## 5. Public API (binding)

New module `src/catraj/compression/cpls.py`, with
`__all__ = ["Barriers", "CplsSelection", "segment_crossings", "cpls_select", "cpls_compress"]`.
All five are re-exported from `catraj.compression` (add them to its imports,
its `__all__`, and one paragraph about `.cpls` in its module docstring).

Do **not** name any function `cpls`: `from .cpls import cpls` would replace
the module attribute `catraj.compression.cpls` with a function, and
`help(catraj.compression.cpls)` must show the module, as it does for `cadp`.

### `Barriers(geometries, crs=None)`

Barrier geometries, indexed once so that many trajectories can be tested
against them.

- `geometries`: one shapely geometry of any type, an iterable or array of
  shapely geometries, a `GeoSeries`, or a `GeoDataFrame`. Lines (a shoreline)
  and polygons (land) both work. `None` entries and empty geometries are
  dropped. An empty input is valid and means "no barriers".
- `crs`: taken from the `GeoSeries`/`GeoDataFrame` when there is one;
  otherwise the argument, otherwise `None` (unknown).
- `Barriers.hits(x0, y0, x1, y1) -> ndarray of bool`: vectorised; for each
  segment `(x0, y0)–(x1, y1)`, whether it intersects any barrier.
- `len(barriers)`: number of indexed parts.
- Anything else as `geometries` (a string, a number, `None`) raises `TypeError`.

Every public function below accepts either a `Barriers` instance or anything
`Barriers(...)` accepts, and wraps it if needed.

### `segment_crossings(x, y, barriers, keep=None, new_only=False) -> ndarray of bool`

One value per segment of the polyline through the retained points (all points
when `keep is None`), so the length is `keep.sum() - 1` (or `n - 1`); empty
when fewer than two points. `True` where the segment intersects a barrier.
With `new_only=True`, a segment counts only if the original sub-path it
replaces does **not** intersect a barrier: `hit(a, b) and not orig(a, b)`.
This is the evaluation measure: for any CPLS result,
`segment_crossings(x, y, b, keep=mask, new_only=True).any()` is `False`.

### `CplsSelection` (dataclass)

Parallel to `cadp.Selection`:

- `dp`: bool array, the Douglas–Peucker choice.
- `topology`: bool array, points retained by stage 2. Never overlaps `dp`.
- `mask` (property): `dp | topology`.
- `added` (property): same as `topology`; the name `Selection` uses.
- `summary()`: text in the style of `Selection.summary()`:

  ```
  13 points -> 3 kept (76.92% compression)
    Douglas-Peucker      2
    + topology           1
  ```

### `cpls_select(x, y, epsilon, barriers) -> CplsSelection`

Array level. `x`, `y` in trajectory order, in projected units. Input
validation is `douglas_peucker`'s (unequal lengths, non-finite positions,
negative or non-finite `epsilon` → `ValueError`). With `n <= 2` every point is
kept and `topology` is all `False`.

### `cpls_compress(traj, epsilon, barriers, by=None)`

Dataframe level; same contract as `cadp.compress`:

- `traj`: a `movingpandas.TrajectoryCollection`, a `movingpandas.Trajectory`,
  or a `GeoDataFrame` of points in trajectory order. Returns the same kind of
  object holding only the retained rows, all columns intact.
- `by`: for a `GeoDataFrame` holding several trajectories, the column naming
  each one (`groupby(by, sort=False)`, each compressed on its own).
- No time argument: CPLS does not use time. Row order is trajectory order.
- A geographic CRS on the trajectory raises `ValueError` (reuse the wording in
  `cadp._select_frame`).
- If the trajectory and the barriers both have a CRS and they differ, raise
  `ValueError` whose message contains `CRS` and tells the user to reproject
  the barriers with `.to_crs(...)`. If either CRS is unknown, assume they match.
- Build the `Barriers` object **once** per call and reuse it for every group
  and every trajectory in a collection.
- `import movingpandas` only inside the branches that need it, as `compress` does.
- Copy the dispatch pattern of `compress` into this function. About twenty
  duplicated lines are accepted; do not refactor `cadp.py` to share them.

---

## 6. Implementation

The semantics above are binding. The approach below is recommended: it was
prototyped and measured (appendix). You may do it differently if G1–G5 and
the speed targets in section 12 hold.

1. **Barriers.** `parts = shapely.get_parts(array_of_geometries)`, drop empty
   ones, `shapely.prepare(parts)`, `tree = shapely.STRtree(parts)`.
2. **`hits`.** Build all segments at once with `shapely.linestrings` from an
   `(m, 2, 2)` coordinate array. `tree.query(lines)` **without** a predicate
   returns bounding-box candidates as `[line_index, part_index]`. Then
   `shapely.intersects(parts[part_index], lines[line_index])`, with the
   prepared parts as the first argument. Large land polygons are tested fast
   this way: 106,758 segments against the sample coastline took 0.1 s.
3. **Skip what cannot be affected.** Every chord between two points of a DP
   segment lies inside the bounding box of that segment's points. If that box
   intersects no barrier, stage 2 drops everything there, which is exactly what
   the greedy would do. Compute the boxes for all DP segments, query the tree
   once, and run stage 2 only on the flagged segments. At sea, far from land,
   this is nearly all of them.
4. **`orig`.** Compute `hit(k, k+1)` for all original segments in one `hits`
   call, only if some segment was flagged. With `cs = [0, cumsum(hit)]`,
   `orig(i, j)` is `cs[j] - cs[i] > 0`.
5. **Greedy without one Python call per point.** From the last retained point
   `a`, test the chords `a → k` for a block of `k = a+2, a+3, …` in one `hits`
   call. The first `k` that is not consistent means point `k-1` is retained and
   becomes the new `a`. If none fails, continue with the next block. Grow the
   block size geometrically (for example 64, 128, …) instead of testing the
   whole rest of the segment each time, so a segment with many retained points
   does not go quadratic. This gives the same answer as the one-at-a-time
   loop: the first failing chord from `a` is the same either way.

Pitfalls:

- `np.minimum.reduceat` needs care at the last segment (an end index of `n` is
  out of range). Test a trajectory whose last DP segment is flagged.
- `STRtree.query` returns indices into the array the tree was built from. Keep
  that array.
- `shapely.prepare` works in place on the array you pass; prepare the array
  you later index.

---

## 7. Data

### What was found

The public Zenodo record the repository already uses
([doi:10.5281/zenodo.1167595](https://doi.org/10.5281/zenodo.1167595), Ray et
al., 2019) includes a coastline for the AIS area:

| | |
|---|---|
| Zenodo file | `[C2] European Coastline.zip` |
| Size, MD5 | 52,613,917 bytes, `52036af60c8427fcda4b867c706bf649` (both from the record's API) |
| Use this layer | `Europe Coastline (Polygone).shp` (+ `.shx`, `.dbf`, `.prj`): 71,514 polygons, fields `Shape_Leng`, `Shape_Area`, **EPSG:3035** (ETRS89-LAEA) |
| The polygons are | **land** (verified: 99.998 % of sample AIS positions fall outside them) |
| Not used | `Europe Coastline.shp`: the same coast as a single 47 MB LineString |
| Also in the zip | `README_parameters.txt`, `README_EU_Coastline.png` |
| Source | European Environment Agency, "EEA coastline for analysis", version 08/07/2013, derived from EU-Hydro and GSHHG, made for 1:100,000 analysis |
| Licence | EEA standard re-use policy: re-use for commercial or non-commercial purposes is permitted free of charge, provided the source is acknowledged. Copyright holder: Directorate-General for Environment (DG ENV). <http://www.eea.europa.eu/legal/copyright> |

The archive is **already downloaded and extracted** on this machine, in
`data/zenodo_1167595/C2_European_Coastline.zip` and
`data/zenodo_1167595/C2_European_Coastline/` (git-ignored). Use it; do not
download it again.

Because the EEA licence allows redistribution with attribution, a small clip
can be committed with the sample. (The AIS sample next to it stays under its
own stricter licence, CC BY-NC-SA 4.0.)

### 7a. Create the sample coastline: `data/sample/Coastline/`

Clip the land polygons to the bounding box of the sample AIS positions plus a
5,000 m margin, in EPSG:32630, the CRS of the other sample layers. This recipe
was run and gives **175 polygons, 12,049 vertices, about 0.2 MB**:

```python
import geopandas as gpd, shapely

ais = gpd.read_file("data/sample/AIS/AIS_DLT_UTM.shp")              # EPSG:32630
xmin, ymin, xmax, ymax = ais.total_bounds
box = shapely.box(xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)
box_3035 = gpd.GeoSeries([shapely.segmentize(box, 1000)], crs=32630).to_crs(3035).iloc[0]

src = "data/zenodo_1167595/C2_European_Coastline/Europe Coastline (Polygone).shp"
land = gpd.read_file(src, bbox=tuple(box_3035.bounds)).to_crs(32630)
land = gpd.clip(land, box)
land = land[~land.geometry.is_empty].explode(index_parts=False)
land = land[land.geom_type == "Polygon"][["geometry"]].reset_index(drop=True)
land.to_file("data/sample/Coastline/Coastline_UTM.shp")
```

Run it once from a temporary script; do not commit the script. Then:

- Commit every sidecar that was written (`.shp`, `.shx`, `.dbf`, `.prj`,
  `.cpg`). `.gitignore` already re-includes `data/sample/**`; confirm with
  `git status` that the files show as untracked, not ignored.
- Write `data/sample/Coastline/Coastline_parameters.txt` in the format of
  `data/sample/AIS/AIS_parameters.txt` (Object, File name, Source, Dataset
  version, SRID, Coverage, Volume, Licence, Description), then add the recipe
  above so the file can be rebuilt. State that the polygons are land, clipped
  from the EEA layer via the Zenodo record, reprojected from EPSG:3035 to
  EPSG:32630.

### 7b. Extend `scripts/download_data.py`

The script must stay standard-library only.

1. Add to `ARCHIVES`:
   `"[C2] European Coastline.zip"` with `md5="52036af60c8427fcda4b867c706bf649"`,
   `size=52_613_917`, `folder="C2_European_Coastline"`, and
   `members=["Europe Coastline (Polygone).shp", "Europe Coastline (Polygone).shx",
   "Europe Coastline (Polygone).dbf", "Europe Coastline (Polygone).prj",
   "README_parameters.txt"]`.
2. Add `TARGETS["cpls"] = ["[C2] European Coastline.zip"]` and include it in
   `TARGETS["all"]`.
3. Update the module docstring (usage lines, licence paragraph: add the EEA
   coastline), the `target` help text, and the sizes. New totals: `cpls`
   ≈ 53 MB; `all` ≈ 825 MB (was ≈ 773 MB); `annotation` and `cadp` unchanged.

Check: `conda run -n catraj python scripts/download_data.py cpls --check`
prints `[x]` and exits 0 (the files are already there), and
`... download_data.py --check` lists five archives.

To exercise the real download path without touching the existing data, write a
temporary script that imports `scripts/download_data.py` as a module, sets its
`DATA` to a new folder **inside the repository** such as
`data/cache/_download_test` (it must be under `ROOT`, because `fetch` prints a
path relative to `ROOT`; `data/cache/` is git-ignored), calls
`fetch("[C2] European Coastline.zip", ["feb"], False)`, checks that the five
members were extracted, and then deletes that folder.

---

## 8. Tests

New file `tests/test_cpls.py`. Use these names or names as clear. Every case
marked *verified* was run against a prototype of section 4 and the expected
values are exact.

Put this reference in the test file. It is the specification in executable
form, and G5 is "the library equals this":

```python
def _reference_cpls(x, y, epsilon, barrier):
    """Algorithm 1 read literally: one intersection test per candidate point."""
    n = len(x)
    dp = douglas_peucker(x, y, epsilon)
    keep = dp.copy()

    def seg(i, j):
        return shapely.LineString([(x[i], y[i]), (x[j], y[j])])

    orig = [barrier.intersects(seg(i, i + 1)) for i in range(n - 1)]
    kept = np.flatnonzero(dp)
    for s, e in zip(kept[:-1], kept[1:]):
        a = s
        for j in range(s + 1, e):
            if barrier.intersects(seg(a, j + 1)) and not any(orig[a:j + 1]):
                keep[j] = True
                a = j
    return keep
```

(`barrier` is a single shapely geometry; pass `shapely.union_all([...])`.)

### Hand-worked cases (all verified)

| test | input | expected retained indices |
|---|---|---|
| `test_wall_keeps_the_point_that_clears_it` | points `(0,0) (2,0) (4,3) (6,3) (8,0) (10,0)`; barrier `LineString([(5,-1),(5,2)])`; `epsilon=10` | DP `[0, 5]`; CPLS `[0, 3, 5]` |
| `test_detour_around_a_wall_is_kept` | `x = 0,1000,2000,3000,4000,4500,5000,5500,6000,7000,8000,9000,10000`; `y = 0,0,0,0,0,80,100,80,0,0,0,0,0`; barrier `LineString([(5000,-50),(5000,50)])`; `epsilon=500` | DP `[0, 12]`; CPLS `[0, 7, 12]`; also assert `segment_crossings(x, y, wall, keep=dp).any()` and not `.any()` for the CPLS mask |
| `test_touching_a_barrier_counts_as_crossing` | points `(0,0) (5,1) (10,0)`; `epsilon=10`; barrier `Point(5, 0)` | `[0, 1, 2]`; and with `Point(5, 0.5)` instead: `[0, 2]` |
| `test_an_existing_crossing_is_tolerated` | `x = linspace(-10, 20, 31)`, `y = 5`; barrier `box(0, 0, 10, 10)`; `epsilon=1` | DP keeps 2 points and CPLS keeps the same 2 |

### Properties (parametrise over at least seeds 0–4 and several `epsilon`)

Build random walks like `_walk` in `tests/test_compression.py`, and random
barriers scattered over the walk's extent: a mix of buffered points (polygons)
and short `LineString`s, some of them deliberately on the path.

- `test_cpls_matches_the_literal_algorithm` — G5: `cpls_select(...).mask`
  equals `_reference_cpls(...)`. Include a case where the **last** DP segment
  is affected, and one with enough barriers that many points are retained.
- `test_cpls_is_a_superset_of_dp` — G1, and `topology` never overlaps `dp`.
- `test_cpls_never_introduces_a_crossing` — G2:
  `not segment_crossings(x, y, b, keep=mask, new_only=True).any()`, while the
  DP mask does have new crossings for at least one seed (assert that too, so
  the test cannot pass vacuously).
- `test_a_path_clear_of_barriers_stays_clear` — G3: drop every barrier the
  original path touches, then assert the result has no crossings at all.
- `test_no_barriers_is_plain_dp` — with `[]`, with `shapely.GeometryCollection()`,
  and with a barrier far outside the extent: `mask == dp`, `added` empty.

### Interface

- `test_barriers_accept_every_documented_form` — one geometry, a list, a
  `MultiPolygon`, a `GeoSeries`, a `GeoDataFrame`, a `Barriers` instance: all
  give the same mask. A list with `None` and an empty geometry in it works.
- `test_segment_crossings_shapes_and_new_only` — lengths (`n-1`,
  `keep.sum()-1`, empty for fewer than two points) and the `new_only` rule on
  the hand-worked cases.
- `test_cpls_edge_cases` — `n` = 0, 1, 2; negative `epsilon` and a NaN position
  raise `ValueError`; `barriers=None` and `barriers="coast"` raise `TypeError`.
- `test_summary_reports_the_topology_stage` — `"topology"` is in `summary()`.
- `test_cpls_handles_a_long_trajectory` — 100,000 points along a coast of many
  small barriers; assert it completes, G1 and G2 hold. No timing assertion.

### Dataframes (mirror the `compress` tests; reuse the `_frame` idea)

- `test_cpls_compress_keeps_every_column` — columns unchanged, index a subset.
- `test_cpls_compress_by_group` — each group equals compressing it alone.
- `test_cpls_compress_refuses_geographic_coordinates` — `match="geographic"`.
- `test_cpls_compress_refuses_a_crs_mismatch` — frame in EPSG:32630, barriers
  as a `GeoSeries` in EPSG:3035 → `ValueError`, `match="CRS"`. Barriers given
  as plain shapely geometries (no CRS) are accepted.
- `test_cpls_compress_movingpandas_objects` — `@pytest.mark.movingpandas`,
  `pytest.importorskip("movingpandas")`; a `Trajectory` returns a `Trajectory`
  with the same id, a `TrajectoryCollection` returns a collection of equal size.

### Sample data (`@pytest.mark.data`)

- In `tests/test_cpls.py`, `test_cpls_on_the_sample_fleet`: read the AIS sample
  and `data/sample/Coastline/Coastline_UTM.shp`; sort with
  `sort_values(["sourcemmsi", "t"], kind="stable")`; `epsilon = 1_000`,
  `by="sourcemmsi"`. Assert, summed over vessels: plain DP has new crossings
  (`> 0`); CPLS has none; `len(dp) <= len(cpls) < len(ais)`; the CPLS index
  contains the DP index. Use inequalities, not the exact counts of section 12:
  a different GEOS version may move a touching case.
- In `tests/test_sample_data.py`, `test_sample_coastline_fits_the_ais`: the
  layer is EPSG:32630, all polygons, non-empty; its bounds overlap the AIS
  bounds and lie inside the clip box (AIS bounds ± 5,000 m); at least 99.9 %
  of AIS positions do not intersect it. (Its bounds do **not** contain the AIS
  bounds: the traffic reaches west into open sea, where there is no land.)

---

## 9. Notebook: `notebooks/03_cpls_compression.ipynb`

It must run **from a fresh clone with no download**: sample AIS and sample
coastline only. Study `02_cadp_compression.ipynb` first and follow its
conventions (**never open an `.ipynb` as raw text**: the two existing
notebooks are 3 to 6 MB of embedded outputs; print the cell sources with a
few lines of `nbformat` from a temporary script instead): the opening cell (including the
`MKL_THREADING_LAYER` line), how it locates the repository root, column
renaming to meaningful names, colour constants, hvplot figures on a basemap,
section headings `## 1 · …`, and a closing `## Notes` cell. Keep the same
kernelspec metadata (`python3`).

Sections:

1. **The problem.** One or two paragraphs and one map: a vessel whose
   Douglas–Peucker simplification at ε = 1 km cuts across land.
2. **Data.** Load the sample AIS and the sample coastline; say where the
   coastline comes from and its licence; show how well they agree (positions
   on land, original segments touching land).
3. **Compress.** `cpls_compress` against plain DP (`compress` with no
   thresholds) at ε = 1 km over the fleet. A table: points kept, compression
   ratio, DTW, new crossings, vessels affected.
4. **One vessel up close.** Map with the land polygons, the original track, DP
   and CPLS, with the points CPLS retained for topology highlighted.
   `CplsSelection.summary()` printed.
5. **Choosing ε.** Sweep ε = 100, 500, 1,000, 5,000 m: crossings DP
   introduces, points CPLS adds, compression ratio of both.
6. **With MovingPandas.** A `TrajectoryCollection` in, a collection out.
7. **Speed.** Time DP and CPLS on the fleet.
8. **The full coastline (optional).** A cell that runs only if
   `data/zenodo_1167595/C2_European_Coastline/` exists: read the polygons with
   a `bbox`, reproject, and use them as barriers. Say how to get the archive
   (`python scripts/download_data.py cpls`).
9. **Notes.** Decisions D3–D7 in plain language, and that 2stage-CPLS is not
   implemented yet.

Build the notebook with `nbformat` from a temporary script (not committed),
then execute it in place:

```bash
conda run -n catraj jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 notebooks/03_cpls_compression.ipynb
```

Requirements: it executes top to bottom with no error and no warning noise in
the outputs; it writes nothing to disk; every number quoted in a markdown cell
matches the output of the code cell it describes; the file stays under 5 MB
(subsample points for maps, as tutorial 1 does). Confirm it ran in the
`catraj` environment (`sys.executable` in a throwaway run, not in the final
notebook).

---

## 10. Documentation

Update each of these. Change only what CPLS touches; keep each file's voice.

| file | change |
|---|---|
| `README.md` | Algorithm table: CPLS status → **implemented**. The sentence "This release implements trajectory annotation and CADP compression" → add CPLS. "Using it": a short `cpls_compress` example after the CADP one. Tutorial table: add notebook 03. "Data": add the sample coastline row (source EEA, licence) and the `cpls` download target; update `~773 MB` → `~825 MB`. |
| `CITATION.cff` | Abstract: mention CPLS. The 2021 reference's `notes`: CPLS implemented in `catraj.compression`; two-stage CPLS and CLSTM planned. Do not change `version`. |
| `docs/README.md` | Papers table, 2021 row: "CPLS: `catraj.compression`; two-stage CPLS, CLSTM (planned)". |
| `docs/getting_the_data.md` | Section 1 table: add notebook 03 (no download) and the `cpls` target. Section 2: "uses four of them" → five, add the coastline row (link, 52.6 MB, extracted files, MD5); disk-space table. Section 4: folder layout. Section 7: EEA licence. All totals. |
| `data/README.md` | Tree: `sample/Coastline/` and `zenodo_1167595/C2_European_Coastline/`. Both "used by" tables. Licences: add the coastline. |
| `data/LICENSE.md` | New section "Coastline — `sample/Coastline/`" in the format of the AIS and Weather sections: source, licence (EEA re-use policy, attribution required, copyright DG ENV), coverage, full dataset, extract here (175 polygons). |
| `scripts/README.md` | Only if the `download_data.py` row needs it. |
| `src/catraj/compression/__init__.py` | Docstring paragraph for `.cpls`; exports. |

After editing, check every relative link you added resolves to a real file,
and grep the repository for `773` and `four of them` to catch a stale total.

---

## 11. Work plan with gates

Do the steps in order. Do not start a step until the previous gate passes.
One commit per step.

**Step 0 — Baseline (no changes).**
Read this brief; read the four files in `src/catraj/compression/`,
`tests/test_compression.py`, section IV-A of the paper.
*Gate:* `git status --short` shows nothing;
`conda run -n catraj python -m pytest -q` reports `180 passed`. If not, stop.

**Step 1 — Core algorithm.** `cpls.py` with `Barriers`, `segment_crossings`,
`CplsSelection`, `cpls_select`; exports; `tests/test_cpls.py` with the
reference, the hand-worked cases, the properties and the interface tests.
Write the hand-worked tests first and watch them fail before the code exists.
*Gate:* all new tests pass; the four hand-worked cases give exactly the
indices in section 8; full suite passes.

**Step 2 — Dataframes.** `cpls_compress` and its tests, including MovingPandas.
*Gate:* full suite passes, no test skipped that was not skipped before.

**Step 3 — Data.** Sample coastline (7a), downloader (7b), the two sample-data
tests, `data/README.md`, `data/LICENSE.md`.
*Gate:* `git status` shows the new sample files as untracked (not ignored);
both `--check` commands of 7b behave as described; the download test of 7b
passes and its folder is deleted; full suite passes; the numbers of section 12
are within range.

**Step 4 — Notebook.**
*Gate:* executes cleanly from top to bottom; under 5 MB; `git status` shows no
stray output files.

**Step 5 — Documentation.** Section 10.
*Gate:* links resolve; no stale totals; full suite still passes.

**Step 6 — Final check and report.** Section 12, then section 13.

---

## 12. Acceptance checklist

Run each item and record the actual result in your report.

1. `conda run -n catraj python -m pytest -q` → all pass; the count is 180 plus
   your new tests; nothing newly skipped.
2. `conda run -n catraj python -m pytest -q tests/test_cpls.py -v` → every test
   named in section 8 is present.
3. **Portability.** `git grep -n "\.union_all()" -- src tests` returns
   nothing. No Python 3.11+ syntax in `src/` or `tests/`.
4. **Numbers on the sample** (AIS sorted by `["sourcemmsi", "t"]`, stable;
   136 vessels, 106,894 points; the sample coastline as barriers). The
   prototype gave the appendix table. At ε = 1,000 m you must get: plain DP
   keeps **about 547** points and introduces **about 31** new crossings in
   **about 19** vessels; CPLS keeps **about 586** points and introduces **0**.
   Accept DP points within ±5, crossings within 25–40, CPLS extra points within
   30–50. CPLS new crossings must be exactly 0. Outside these ranges: stop and
   find out why (usual causes: sort order, or a different coastline clip).
5. **Speed.** CPLS over the whole sample fleet at ε = 1,000 m in under 5 s
   (prototype: 0.2 s). A 100,000-point synthetic trajectory with barriers
   along it in under 10 s.
6. `help(catraj.compression.cpls)` shows the **module** docstring with the
   method and decisions D1–D9; `catraj.compression.cpls_compress` is callable.
7. The notebook re-executes cleanly with the command in section 9.
8. `git status --short` is clean. `git log --oneline` shows
   your commits on top of `46c024a`, none with an AI attribution line. Nothing
   was pushed: `git status -sb` shows `main` ahead of `github/main`.
9. `git diff 46c024a --stat` touches only files this brief names.

---

## 13. Final report

End with a report in this shape, in plain prose and short tables:

1. **Done** — one line per step of section 11, with its commit hash.
2. **Checklist** — each item of section 12 with the actual output or number.
3. **Deviations** — anything you did differently from this brief, and why.
   "None" if none.
4. **For the author to confirm** — decisions D3–D7 restated in one line each,
   since the docstring says "decided here" until the author confirms them.
5. **Not done / open** — anything skipped, uncertain or worth a second look.

---

## Appendix: verified numbers (2026-11-23)

Sample AIS (`data/sample/AIS/AIS_DLT_UTM.shp`): 106,894 points, 136 vessels,
EPSG:32630; columns `sourcemmsi, navigation, rateofturn, speedoverg,
courseover, trueheadin, lon, lat, t, vesselType`. Bounds in EPSG:32630:
`[278974, 5253849, 451655, 5429918]`; in degrees: longitude −5.978 to −3.643,
latitude 47.420 to 48.996.

Coastline against that sample (clip of section 7a):

| | |
|---|---|
| polygons, vertices | 175, 12,049 |
| AIS positions on land | 2 of 106,894 (in 2 vessels) |
| original consecutive segments touching land | 10 of 106,758 |
| distance from a position to land, 1 % / 10 % / median / 90 % | 28 m / 101 m / 450 m / 6,094 m |

Whole archive extent (longitude −10 to 0, latitude 45 to 51): 875 polygons,
65,335 vertices after clipping; reading them with a `bbox` takes 0.3 s.

Prototype of section 4 on the sample fleet (optimised version equal to the
literal reference for every vessel at every ε):

| ε (m) | DP keeps | CPLS keeps | added | new crossings by DP (vessels) | new crossings by CPLS | DP time | CPLS time | literal reference time |
|---|---|---|---|---|---|---|---|---|
| 100 | 1,339 | 1,341 | 2 | 1 (1) | 0 | 0.03 s | 0.10 s | 1.4 s |
| 500 | 687 | 705 | 18 | 16 (11) | 0 | 0.01 s | 0.15 s | 1.5 s |
| 1,000 | 547 | 586 | 39 | 31 (19) | 0 | 0.01 s | 0.16 s | 1.5 s |
| 5,000 | 323 | 431 | 108 | 57 (31) | 0 | 0.01 s | 0.37 s | 1.5 s |

Segments that intersect land but are *licensed* by D4 (the original sub-path
already touches land) remain in both results: 9 or 10 at every ε.
