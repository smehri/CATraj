# Implement CLSTM in `catraj`: implementation brief

You are implementing the last algorithm of the package, **CLSTM
(Context-aware Long Short-Term Memory)**, the prediction half of the hybrid
model, in an existing, tested, public Python library. This brief contains
everything you need: the method, every design decision, the API, the tests,
the data, the documentation changes, the order of work and how to check each
step.

Read the whole brief before you change anything. Where it says **binding**, do
not deviate. If you think a binding item is wrong, stop and ask (section 3).

Facts marked *verified* were checked on 2026-11-24 at commit `0939661` on
`main`. Those checks ran on the author's first machine. You run on a second
machine, so check the environment yourself first (step 0).

Contents: 1 Mission · 2 The repository · 3 Rules · 4 The method · 5 API ·
6 Implementation · 7 Data · 8 Tests · 9 Notebook · 10 Benchmark on the full
archive · 11 Documentation · 12 Work plan with gates · 13 Acceptance
checklist · 14 Final report

---

## 1. Mission and scope

`catraj` implements context-aware trajectory algorithms from three papers.
Annotation, CADP, CPLS and 2stage-CPLS are done. CLSTM is the only thing left.
It comes from:

> Mehri, Alesheikh & Basiri (2021). *A Contextual Hybrid Model for Vessel
> Movement Prediction.* IEEE Access 9, 45600–45613.
> `docs/papers/2021_Mehri_IEEE-Access_Contextual-Hybrid-Model_CPLS-CLSTM.pdf`,
> sections III-A (problem formulation), IV-B to IV-E, V-B.

**CLSTM in one sentence:** a curvilinear (dead-reckoning) model predicts a
vessel's next three positions from its speed and course, and an LSTM network,
**one per vessel type**, takes those three predicted points and returns the
three corrected positions. The ordinary LSTM it is compared with is the same
network trained once on every vessel type together.

**In scope**

1. A new subpackage `src/catraj/prediction/` (`__init__.py`, `curvilinear.py`,
   `clstm.py`).
2. The optional dependency `predict = ["torch>=2.0"]` in `pyproject.toml`, and
   the CI change in section 8.
3. `tests/test_curvilinear.py`, `tests/test_clstm.py`, and one sample-data test.
4. `notebooks/04_clstm_prediction.ipynb`, executed, with outputs.
5. `scripts/benchmarks/clstm_full_archive.py` (section 10), run on the GPU,
   with its results reported (not committed: `results/` is git-ignored).
6. The documentation updates listed in section 11.

**Out of scope: do not do these**

- Any change to `catraj.annotation`, `catraj.compression`,
  `catraj.preprocessing` or `app/`.
- Version bump, git tag, release, PyPI, `git push`.
- Reformatting existing files, renaming anything, removing anything.
- Any dependency besides `torch` (and only as the optional extra).
- Anomaly detection, fleet prediction, other contexts than vessel type. The
  paper names these as out of scope or future work.

---

## 2. The repository

### Layout

| path | what it holds |
|---|---|
| `src/catraj/compression/` | `douglas_peucker.py`, `cadp.py`, `cpls.py`, `two_stage.py`, `evaluation.py`. **Read `cadp.py` and `two_stage.py` first**: the module docstrings, the dataframe dispatch (`GeoDataFrame` / `Trajectory` / `TrajectoryCollection`, `by=`, `time_column=`), the error messages. CLSTM must look like it belongs next to them |
| `src/catraj/_time.py`, `_crs.py` | shared helpers for reading time and refusing geographic CRSs. Reuse them |
| `tests/` | pytest suite. *Verified:* **447 passed** at `0939661` (about 90 s). `helpers.py` has `DATASETS = <root>/data/sample` |
| `notebooks/` | tutorials 01–03, committed with outputs |
| `scripts/download_data.py` | standard-library downloader for Zenodo record 1167595. Target `annotation` fetches `[P1] AIS Data.zip`, whose members include `nari_dynamic.csv` **and** `nari_static.csv` |
| `scripts/benchmarks/` | experiment scripts on the full archive. Read `paper_experiments.py` for the style |
| `data/sample/` | committed 24 h sample: `AIS/`, `Weather/`, `Coastline/` (EPSG:32630) |
| `data/zenodo_1167595/`, `data/cache/`, `results/` | not committed (git-ignored) |
| `.github/workflows/tests.yml` | CI: `pip install -e ".[test,movingpandas]"`, then `pytest -ra`, on Python 3.10/3.11/3.12 (Linux) and 3.12 (Windows) |

### Environment

- A conda environment named `catraj` with the package installed editable
  (`pip install -e ".[dev]"`). If the machine lacks it, create it and record
  how in your report.
- **PyTorch is new.** Install it into `catraj` with the **CUDA** build that
  matches the machine's driver, from the selector at <https://pytorch.org>
  (on Windows: `pip install torch --index-url https://download.pytorch.org/whl/cuXXX`).
  Plain `pip install torch` on Windows gives a CPU-only build. Check with:

  ```bash
  conda run -n catraj python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

  `cuda.is_available()` must print `True` before step 5 (the benchmark).
  Tests and the notebook do not need the GPU.
- `ruff` and `black` may not be installed. Do not run a formatter on existing
  files. Match the surrounding style by hand; keep lines to 88 characters.
- `conda run` rejects a multi-line `python -c "..."`. For anything longer, write
  a script in a temporary folder outside the repository and run that.
- `pdftotext` (Git Bash) gives the paper as text. Without `-layout`, the two
  columns come out in reading order.

### Compatibility you must keep

- Python **3.10** syntax and standard library only.
- numpy ≥ 1.24, pandas ≥ 2.0, geopandas ≥ 0.13, shapely ≥ 2.0 (see the APIs
  `two_stage.py` already uses; no `GeoSeries.union_all()`).
- torch ≥ 2.0: plain `torch.nn.LSTM`, `torch.nn.Linear`, `torch.optim.Adam`.
  No `torch.compile`, no Lightning, no other wrapper.
- **`import catraj` and `import catraj.prediction` must work without torch.**
  Only `clstm.py` needs torch, and it imports it inside the functions that
  use it (section 6).

### House style (taken from the existing code)

- Every module opens with a long docstring that states the method, cites the
  paper section, and lists **every decision taken where the paper leaves
  room**. `two_stage.py` is the model.
- NumPy-style docstrings, `from __future__ import annotations`, explicit
  `__all__`.
- Array level first, dataframe level second.
- Tests are short, named as sentences, use synthetic data with a known
  answer, and compare against a slow obvious reference inside the test file.
- Comments explain why, not what. No commented-out code, no TODOs.

---

## 3. Rules

### Git

- Work on `main`, in the existing working tree. Commit locally once per
  finished step of section 12. **Never push, tag, amend, rebase or force.**
  The owner pushes after review.
- Plain `git commit -m ...`. The repository's git config supplies the author.
  Never pass `-c user.name`, `-c user.email` or `--author`.
- **No AI attribution anywhere**: no `Co-Authored-By` trailer, no "generated
  with" line, in commits, code, docs or the notebook. This is the owner's rule.
- Commit subjects are imperative sentences, like the existing ones:
  `Add 2stage-CPLS compression (catraj.compression.two_stage)`.
- Stage files by name. **Never `git add -A` or `git add .`**
- Do not edit, move or delete this brief (`CLSTM_GUIDE.md`).

### Files

- Do not delete or re-download anything in `data/zenodo_1167595/`,
  `data/cache/` or `results/` that is already there.
- Do not modify an existing test. If an existing test fails after your change,
  your change is wrong.
- Temporary scripts and outputs go in a temporary folder outside the
  repository. Benchmark outputs go in `results/04_clstm/` (git-ignored).

### GPU

- The machine's GPU may be shared. Before the benchmark, check `nvidia-smi`.
  If another process is using most of the memory, stop and ask.
- Tests force `device="cpu"`. The notebook uses `device="auto"` and must also
  finish on a CPU-only machine in under 10 minutes.

### Stop and ask the owner when

- you want to deviate from a **binding** item in sections 4 or 5;
- an existing test fails and the cause is not your new code;
- the synthetic tests of section 8 do not reach their thresholds after the
  implementation matches section 4 (do **not** tune thresholds to pass);
- on the full archive, CLSTM is **worse** than the curvilinear model alone
  (section 13, item 5);
- you would need any dependency other than torch, or to change existing modules.

When you stop, say what you found, what you tried, and what you need decided.
Do not work around the problem.

### Honesty

- Report results as they are, with their outputs. If you skipped a step, say so.
- In docstrings, write "decided here" for decisions C5–C14 and H1–H6. The
  author decided C1–C4 on 2026-11-24 (see section 4); write "decided with the
  author (2026-11-24)" for those, as `two_stage.py` does for its own.
- Do not claim the paper's 15.68 % improvement is reproduced. Report your
  numbers. Different data (Brest, not the US east coast), different period,
  different types.

---

## 4. The method (binding)

### What the paper says

Read sections III-A, IV-B to IV-E and V-B yourself before writing the module
docstrings. In short:

- **Hybrid framework (Fig. 5).** AIS → preprocessing → 2stage-CPLS
  compression → curvilinear prediction T′ → CLSTM correction → T.
- **Curvilinear model (Eqs. 7–11).** With high-frequency AIS, motion between
  two messages is taken as linear, "both normal and tangential accelerations
  can be set to zero":
  `L_i = v_i · dt_i` (8), `dt_i = t_{i+1} − t_i` (9),
  `Δx_i = L_i · sin Φ_i`, `Δy_i = L_i · cos Φ_i` (10),
  `(x′_{i+1}, y′_{i+1}) = (x_i, y_i) + (Δx_i, Δy_i)` (11).
  From the current position, speed, course (and those at t−1) it predicts
  **three** future positions `T′ = [p′_{t+1}, p′_{t+2}, p′_{t+3}]`.
- **Time step (Eq. 12).** `Δt = (Σ dt_i) / n`, the mean interval between
  consecutive vertices of the trajectory.
- **Deviation (Eq. 2).** `δ_k = (x_k − x′_k) i + (y_k − y′_k) j`. The network
  learns to turn T′ into T.
- **Input (Eq. 21).** `{x′_{t+1}, y′_{t+1}, t+1; x′_{t+2}, y′_{t+2}, t+2;
  x′_{t+3}, y′_{t+3}, t+3}`.
- **Output (Eq. 22).** `{x_{t+1}, y_{t+1}; x_{t+2}, y_{t+2}; x_{t+3}, y_{t+3}}`,
  the predicted actual positions.
- **Training error (Eq. 20)** is printed as `e_k = √(y_k² − o_k²)`; the text
  says "the square root of the difference of output value and expected
  output", i.e. a root-squared error.
- **Evaluation (Eqs. 23–25).** RMSE of `Δx = x_pred − x_actual`,
  `Δy = y_pred − y_actual`, and the point-wise horizontal error (the distance
  between each predicted vertex and the actual one).
- **Context (IV-E, V-B).** "a unique LSTM deep learning neural network for each
  vessel type group"; seven groups in the paper (tanker, tug, cargo, sailing,
  passenger, other, fishing). "Other" holds the vessels whose type is null. A
  group with too little data (military, 20 trajectories) was eliminated.
- **Comparison.** The ordinary LSTM, trained on all types together, same data.
- **Split.** 80 % of the trajectories for training, 20 % for evaluation.

### Decisions taken with the author (2026-11-24)

| # | Decision |
|---|---|
| C1 | **Input and output exactly as Eqs. 21 and 22.** Each sample is a sequence of three steps; step j is `(x′_{t+j}, y′_{t+j}, τ_j)`. The output is the three **absolute** positions `(x_{t+j}, y_{t+j})`. No observed history, no relative coordinates. |
| C2 | **One network per vessel type** (the context), plus the ordinary LSTM as the same class with no context column (section 5). |
| C3 | **PyTorch**, as the optional extra `catraj[predict]`. It runs on an NVIDIA GPU on Windows and Linux. |
| C4 | **Data.** Tests and the tutorial run on the committed sample with no download. Accuracy results come from the full Brest archive, through the benchmark script (section 10). |

### Decisions taken here (each goes in the module docstring)

| # | Decision | Reason |
|---|---|---|
| C5 | **Constant speed and course.** T′ holds `v_t` and `Φ_t` for all three steps: `p′_{t+j} = p_t + j·Δt·v_t·(sin Φ_t, cos Φ_t)`. The values at t−1 are not used. | The paper sets both accelerations to zero, and those accelerations are the only way t−1 enters the model. State it. |
| C6 | **Speed and course come from the AIS fields**, SOG and COG, of the row at t. SOG is in knots unless `speed_unit="native"` (position units per second); `1 knot = 1852/3600 m/s`. COG is in degrees clockwise from north. | "velocity v, and course Φ of the vessel" are AIS state fields in section III-B. |
| C7 | **Unavailable values drop the sample.** SOG `>= 102.3` (AIS "not available"), COG `>= 360`, or NaN in either. *Verified* on the sample: 8,602 rows have SOG 102.3 and 9,045 have COG 360. | They are codes, not measurements. |
| C8 | **Grid north is taken as north.** The positions are projected (UTM); COG is true north. The convergence is under 2° over the Brest area. | State it as a limitation. No correction. |
| C9 | **Δt per trajectory, by Eq. 12**, over the trajectory as given (in the framework, the 2stage-CPLS output): `Δt = (t_last − t_first)/(n − 1)`. | Eq. 12 is the mean of the intervals. |
| C10 | **Every vertex is a prediction origin**, and the actual positions at `t + jΔt` are **linearly interpolated in time** along the same trajectory. A sample whose `t + 3Δt` lies past the last vertex is dropped. | The paper trains and validates on the compressed dataset. 2stage-CPLS keeps the vertices where the speed changes (by `speed_epsilon`), so time interpolation between kept vertices is faithful within that tolerance. |
| C11 | **τ_j is the time offset `j·Δt` in seconds**, not an absolute timestamp. | Eq. 21's "t+1, t+2, t+3" are the times of the predicted epochs. An absolute UNIX time would only tell the network the date. |
| C12 | **Vessel type groups.** Null, empty or missing type → `"Other"` (the paper's definition). A group with fewer than `min_samples` training samples (default 200) gets no network and is listed in `dropped_groups`. Predictions for a group with no network are NaN. | The paper eliminated a group with too little data. Silently borrowing another group's network would defeat C2. |
| C13 | **Split by trajectory**, never by sample: all samples of one trajectory go to the same side. Seeded. | Samples of one trajectory overlap in time. Splitting them would leak. |
| C14 | **A trajectory is what the caller passes**: a `Trajectory`, each member of a `TrajectoryCollection`, or each `by` group of a `GeoDataFrame`. No gap splitting inside `catraj.prediction`. | The caller splits at gaps before compression (movingpandas `ObservationGapSplitter`). The tutorial and the benchmark show how. |

### Network and training (decided here; all are arguments with these defaults)

| # | Item | Default |
|---|---|---|
| H1 | Architecture | `nn.LSTM(input_size=3, hidden_size=64, num_layers=1, batch_first=True)` over the 3-step sequence, then `nn.Linear(64, 2)` applied to each step's hidden state → `(m, 3, 2)`. Initial hidden and cell states zero (the paper: "the initial state of the cell … set to be zero"). Weights use PyTorch's default initialisation: zero weights, read literally, would never train. |
| H2 | Scaling | Per network, standardise each input feature (x′, y′, τ) and each output coordinate (x, y) with the mean and standard deviation of **that network's training samples**. Store them; invert on prediction. Absolute UTM values (≈ 3·10⁵, 5·10⁶) cannot be fed raw. |
| H3 | Loss | Mean squared error on the standardised outputs. Eq. 20 is the root of a squared error; minimising the MSE minimises it. |
| H4 | Optimiser | Adam, learning rate `1e-3`, batch size 256, at most 200 epochs. |
| H5 | Early stopping | Hold out 10 % of the **training trajectories** (by C13, seeded) as validation; stop when validation loss has not improved for 10 epochs; restore the best weights. If a group has fewer than 2 training trajectories, train for `max_epochs` with no validation. |
| H6 | Determinism | `seed` (default 0) seeds numpy and torch for splitting, initialisation and batch order. On CPU two fits with the same seed give identical predictions. On CUDA they may differ slightly. |

### What the result guarantees (each one is a test in section 8)

- R1. `curvilinear_predict` is exact on a straight constant-velocity track.
- R2. `make_samples` reproduces a hand-worked trajectory exactly (Δt, origins,
  inputs, targets, dropped samples).
- R3. No trajectory is on both sides of a split.
- R4. On synthetic data where each type has its own systematic deviation from
  dead reckoning, CLSTM beats both the curvilinear model and the ordinary LSTM
  by the margins in section 8. This is the Simpson-paradox argument of
  section IV-E, in executable form.
- R5. Same seed, same data, CPU → identical predictions. Save and load →
  identical predictions.

---

## 5. Public API (binding)

### `catraj.prediction.curvilinear` (numpy only)

`__all__ = ["KNOT", "mean_interval", "curvilinear_predict"]`

- `KNOT = 1852 / 3600`. One knot in metres per second.
- `mean_interval(t) -> float`. Eq. 12. `t` in seconds, increasing. Fewer than
  two values or non-increasing times → `ValueError`.
- `curvilinear_predict(x, y, speed, course, dt, steps=3) -> (px, py)`. All
  arrays have the same length `m`. `speed` is in position units per second,
  `course` in degrees clockwise from north, `dt` a scalar or a length-`m`
  array. Returns two `(m, steps)` arrays (C5). Unequal lengths or a negative
  `dt` → `ValueError`.

### `catraj.prediction.clstm` (needs torch at fit/predict time only)

`__all__ = ["PredictionSamples", "make_samples", "split_by_trajectory",
"ClstmPredictor", "prediction_errors"]`

Do **not** name a function or class `clstm`: `help(catraj.prediction.clstm)`
must show the module.

#### `PredictionSamples` (dataclass)

| field | shape | content |
|---|---|---|
| `inputs` | `(m, steps, 3)` float64 | `(x′, y′, τ)` per step, by Eq. 21 |
| `targets` | `(m, steps, 2)` float64 | actual `(x, y)` per step, by C10 |
| `origin` | `(m, 2)` | position at t |
| `time` | `(m,)` | t of the origin, in seconds |
| `dt` | `(m,)` | the trajectory's Δt |
| `group` | `(m,)` object | vessel type after C12's null→"Other" (or `"all"` with no context) |
| `trajectory` | `(m,)` object | trajectory id |

- `len(samples)` → `m`. `samples.curvilinear` (property) → `inputs[..., :2]`.
- `samples.subset(mask_or_indices)` → a `PredictionSamples`.
- `PredictionSamples.concat(list)` (classmethod).

#### `make_samples(traj, *, speed_column, course_column, context_column=None, time_column=None, by=None, steps=3, speed_unit="knots") -> PredictionSamples`

- `traj`: a `movingpandas.TrajectoryCollection`, a `movingpandas.Trajectory`,
  or a `GeoDataFrame` of points in trajectory order. Time is read as
  `two_stage_cpls_compress` reads it (`time_column`, or a `DatetimeIndex`).
  Reuse that logic through `catraj._time`; if it is private to `two_stage.py`,
  copy it (no refactor of existing modules).
- A geographic CRS raises `ValueError` (reuse `catraj._crs`).
- `by`: the trajectory-id column for a `GeoDataFrame` holding several
  trajectories; trajectory id = the `by` value, or the movingpandas id.
- `context_column`: the vessel-type column. A type must be constant within a
  trajectory; if it is not, take the most frequent non-null value and say so
  in the docstring.
- `speed_unit`: `"knots"` or `"native"`; anything else → `ValueError`.
- Missing columns → `KeyError` naming the column.

#### `split_by_trajectory(samples, test_size=0.2, seed=0) -> (train, test)`

C13. Shuffle the unique trajectory ids with `numpy.random.default_rng(seed)`
and take the first `round(test_size · n_trajectories)` (at least 1, and at
most `n − 1`) as test. Fewer than 2 trajectories → `ValueError`.

#### `ClstmPredictor`

```python
ClstmPredictor(*, context=True, hidden_size=64, num_layers=1, lr=1e-3,
               batch_size=256, max_epochs=200, patience=10,
               validation_size=0.1, min_samples=200, seed=0, device="auto")
```

- `context=True`: one network per `samples.group` value (CLSTM).
  `context=False`: one network for every sample, ignoring the group (the
  paper's ordinary LSTM).
- `device`: `"auto"` → `"cuda"` if `torch.cuda.is_available()` else `"cpu"`;
  or any string `torch.device` accepts.
- Constructing it without torch raises `ImportError` with the message
  `"CLSTM needs PyTorch: pip install catraj[predict]"` (plus a pointer to the
  CUDA build).
- `fit(samples) -> self`. Trains each network (H1–H6). Sets `groups_` (trained
  group names, sorted), `dropped_groups_`, and `history_`: a dict, group →
  list of `(epoch, train_loss, validation_loss)` tuples.
- `predict(samples) -> ndarray (m, steps, 2)`. Absolute positions; NaN rows
  for groups with no network (C12). Before `fit` → `RuntimeError`.
- `save(path)` / `ClstmPredictor.load(path, device="auto")`: one file via
  `torch.save` holding the configuration, each network's `state_dict` and its
  scalers. Load on CPU works for a model trained on GPU (`map_location`).
- `summary() -> str`: per group, number of training samples, epochs run,
  best validation loss; then the dropped groups.

#### `prediction_errors(predicted, samples) -> pandas.DataFrame`

Eqs. 23–25 and the point-wise error. One row per group plus a final row
`"all"`. Columns: `samples`, `rmse` (Eq. 25 over all steps), `rmse_t+1`,
`rmse_t+2`, `rmse_t+3`, `mean_error`, `median_error` (point-wise, in position
units). NaN predictions are excluded and counted in a column `missing`.
`predicted` may also be `samples.curvilinear`, to score the physical model alone.

### `catraj.prediction` (`__init__.py`)

Re-export everything above. The module docstring describes the hybrid
framework in one paragraph and points to `.curvilinear` and `.clstm`. Import
`clstm` normally: it must not import torch at module level.

---

## 6. Implementation

The semantics above are binding. The approach below is recommended.

1. **Lazy torch.** In `clstm.py`, a private `_torch()` does
   `try: import torch except ImportError: raise ImportError(...) from None` and
   returns the module. The `nn.Module` subclass is built inside a function that
   takes `torch`, so the class definition does not run at import:

   ```python
   def _network(torch, hidden_size, num_layers):
       class _Net(torch.nn.Module):
           def __init__(self):
               super().__init__()
               self.lstm = torch.nn.LSTM(3, hidden_size, num_layers, batch_first=True)
               self.head = torch.nn.Linear(hidden_size, 2)
           def forward(self, x):
               out, _ = self.lstm(x)          # zero initial states (H1)
               return self.head(out)
       return _Net()
   ```

2. **Samples, vectorised per trajectory.** With `t` (seconds), `x`, `y`:
   `dt = (t[-1] - t[0]) / (n - 1)`; origins `k` with `t[k] + steps*dt <= t[-1]`;
   targets `np.interp(t[k] + j*dt, t, x)` (and `y`). Then drop the C7 rows.
   Duplicate timestamps break `np.interp` monotonicity: keep the first row of
   each duplicate time (state it in the docstring).
3. **Training loop.** Tensors in `float32` on the device; a manual loop with
   `torch.randperm` (seeded generator) for batches. No `DataLoader` workers.
   On Windows they would also need `if __name__ == "__main__"` guards in the
   notebook.
4. **Scalers** as numpy arrays `mean`, `std` (std 0 → 1), applied on the CPU
   before moving to the device.
5. **Predict** in batches of 65,536 under `torch.no_grad()` and `model.eval()`.

Pitfalls:

- A trajectory shorter than 2 vertices, or one with `Δt = 0`, yields no samples.
  It is not an error.
- `context_column` with a categorical dtype: convert to `str` after filling nulls.
- Keep everything float64 in `PredictionSamples`. Cast to float32 only at the
  network's boundary: UTM northings lose metres in float32 before scaling.

---

## 7. Data

### The sample (committed, used by tests and the tutorial)

*Verified:* `data/sample/AIS/AIS_DLT_UTM.shp`, 106,894 points, 136 vessels,
24 h (UNIX 1454634002 to 1454720400), EPSG:32630. Columns `sourcemmsi,
navigation, rateofturn, speedoverg (knots), courseover (degrees), trueheadin,
lon, lat, t (UNIX s), vesselType`. One type per vessel. Reporting interval:
median 10 s, 90 % 13 s, 99 % 200 s; 265 gaps over 30 minutes.

| vesselType | vessels | points |
|---|---|---|
| Fishing | 47 | 16,513 |
| Other | 22 | 4,063 |
| Cargo | 16 | 8,201 |
| Passenger | 10 | 25,906 |
| Tanker | 10 | 7,378 |
| Sailing Craft | 10 | 1,260 |
| Reserved | 9 | 25,447 |
| Dredger, Pleasure Craft, SAR | 3 each | 174, 414, 5,741 |
| Plilot Vessel (sic), Tug, Military | 1 each | 3,248, 8,548, 1 |

Keep the labels as they are (including the misspelling in the data).

The sample is far too small for meaningful accuracy: after 2stage-CPLS only
a few thousand vertices remain. In the tutorial, say so plainly. It shows the
API, not the paper's result.

### The full archive (benchmark only)

`[P1] AIS Data.zip` holds `nari_dynamic.csv` (1.12 GB; columns `sourcemmsi,
navigationalstatus, rateofturn, speedoverground, courseoverground,
trueheading, lon, lat, t`, WGS 84, October 2015 to March 2016) and
`nari_static.csv` (86 MB; `shiptype` code per message). The codes map to names
through `P1_AIS_Codes/Ship Types List.csv` (`shiptype_min` … `shiptype_max`).
Read `P1_AIS_Codes/README_parameters.txt` and check the columns yourself. Also
check how the sample's `vesselType` was derived, so that the benchmark labels
match the sample's (look in `scripts/benchmarks/` and the git log). If they
cannot be matched, stop and ask.

Run `python scripts/download_data.py annotation --check` first. If
`nari_static.csv` is missing from `data/zenodo_1167595/P1_AIS_Data/`, extract
it from the zip that is already there. Do not download again if the zip exists.

---

## 8. Tests

### CI

Add a `torch` marker to `[tool.pytest.ini_options].markers`
(`"torch: needs the optional PyTorch dependency"`). Every test that trains or
loads a network starts with `torch = pytest.importorskip("torch")` and carries
`@pytest.mark.torch`.

In `.github/workflows/tests.yml`, add one step to the **Linux, Python 3.12**
job only, before the tests:

```yaml
      # The CPU build of PyTorch, so the CLSTM tests run instead of being
      # skipped; the CUDA build would add gigabytes for nothing.
      - name: Install PyTorch (CPU)
        if: matrix.os == 'ubuntu-latest' && matrix.python == '3.12'
        run: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `tests/test_curvilinear.py` (no torch)

- `test_straight_track_is_predicted_exactly`: R1. Speed 5, course 90°, from
  (100, 200), `dt=10`, `steps=3` → `px = [150, 200, 250]`, `py = [200]*3`.
  Course 0° moves +y; course 180° moves −y (use `np.testing.assert_allclose`,
  `atol=1e-9`).
- `test_dt_can_differ_per_point`.
- `test_mean_interval_is_eq_12`: `[0, 10, 30, 60]` → 20.
- `test_curvilinear_rejects_bad_input`: unequal lengths, negative `dt`,
  `mean_interval` of one value or of a non-increasing series.

### `tests/test_clstm.py`

Samples (no torch):

- `test_make_samples_hand_worked`: R2. One trajectory: `t = [0, 10, 20, 30,
  40, 50, 60]`, `x = 0, 10, 20, 30, 40, 50, 60`, `y = 0`, SOG `1/KNOT` knots
  (1 m/s), COG 90 → Δt 10; origins 0–3 (4 samples; origin 4 needs t=70);
  targets equal the curvilinear points exactly; `τ = [10, 20, 30]`.
  Then bend the track (`y` jumps by 5 m at t ≥ 30) and assert the
  interpolated targets by hand.
- `test_unavailable_speed_or_course_drops_the_sample`: SOG 102.3, COG 360, NaN.
- `test_null_type_becomes_other` and `test_types_follow_the_trajectory`.
- `test_make_samples_accepts_every_form`: a `GeoDataFrame` with `by`, a
  `Trajectory`, a `TrajectoryCollection` (`@pytest.mark.movingpandas`): same
  samples. Time from `time_column` and from a `DatetimeIndex`.
- `test_make_samples_refuses_geographic_coordinates`, missing column → `KeyError`.
- `test_split_keeps_trajectories_whole`: R3, plus test share within one
  trajectory of 20 %, and determinism by seed.

Network (`@pytest.mark.torch`, `device="cpu"`, small: under 30 s together):

Build the synthetic fleet once (a module-level fixture): about 60 trajectories
on straight-ish random headings, 3 types, 20 each. The *actual* motion of
type A drifts +100 m east per step relative to dead reckoning, type B +100 m
north, type C none. (Generate actual positions; give the rows the SOG and COG
of the dead-reckoning motion, so T′ misses the drift.) Keep coordinates
around UTM magnitudes (x ≈ 4·10⁵, y ≈ 5.3·10⁶) to exercise H2.

- `test_clstm_learns_each_types_deviation`: R4. On the test split, mean
  point-wise error: CLSTM < 0.3 × curvilinear's, and CLSTM < 0.5 × the
  ordinary LSTM's (`context=False`), which cannot tell A from B. Use
  `max_epochs` large enough, `patience` default. If these margins fail with a
  correct implementation, **stop and report**. Do not loosen them.
- `test_fit_is_deterministic_on_cpu`: R5.
- `test_save_and_load_give_the_same_predictions`: R5 (`tmp_path`).
- `test_small_groups_get_no_network`: `min_samples` above one group's count →
  in `dropped_groups_`, its predictions NaN, `prediction_errors` counts them in
  `missing`.
- `test_predict_before_fit_raises`.
- `test_prediction_errors_matches_eq_25`: against a hand computation.
- `test_missing_torch_gives_install_hint` (no torch needed): monkeypatch
  `sys.modules["torch"] = None`, construct `ClstmPredictor()`, assert
  `ImportError` matching `"catraj\\[predict\\]"`.
- `test_import_does_not_need_torch`: in a subprocess with `torch` blocked
  (`sys.modules["torch"] = None` before `import catraj.prediction`), the
  import succeeds.

### Sample data (`@pytest.mark.data`, in `tests/test_clstm.py`)

`test_clstm_runs_on_the_sample_fleet` (`@pytest.mark.torch` too): sort the
sample by `["sourcemmsi", "t"]` (stable); compress with
`two_stage_cpls_compress` using the README's parameters (ε 1,000 m, 1.54 m/s,
the sample coastline, `status_column="navigation"`, `stay_radius=100`,
`stay_duration=600`, `time_column="t"`, `by="sourcemmsi"`); `make_samples`
with `context_column="vesselType"`; split; fit with `max_epochs=3`,
`min_samples=50`, CPU. Assert shapes, finite predictions for trained groups,
and at least one dropped group. No accuracy assertion.

---

## 9. Notebook: `notebooks/04_clstm_prediction.ipynb`

Runs **from a fresh clone with no download**, CPU or GPU, in under 10 minutes
on CPU. Study `03_cpls_compression.ipynb` first and follow its conventions:
opening cell (with the `MKL_THREADING_LAYER` line), repository-root lookup,
column renaming, colour constants, hvplot maps on a basemap, `## 1 · …`
headings, a closing `## Notes`. **Never open an `.ipynb` as raw text**: print
cell sources with `nbformat` from a temporary script.

Sections:

1. **The idea.** The hybrid framework (Fig. 5) in a paragraph: dead
   reckoning, then a correction learned per vessel type; why per type
   (Simpson's paradox).
2. **Data.** The sample; vessel types table; the 2stage-CPLS compression with
   the README parameters, and how many vertices remain per type.
3. **The curvilinear model.** One vessel: T′ against the actual track for a
   few origins, on a map. Δt by Eq. 12.
4. **Samples.** `make_samples`, the split by trajectory, counts per type.
5. **Train.** CLSTM (`min_samples` small enough that 3–5 groups train) and the
   ordinary LSTM. `summary()`. Loss curves from `history_`.
6. **Evaluate.** `prediction_errors` for curvilinear, ordinary LSTM and CLSTM,
   side by side. One vessel's predictions on a map. **Say plainly that a
   24-hour sample cannot support conclusions**, and point to section 10's
   benchmark for the real comparison.
7. **Save and load.**
8. **Notes.** C5–C14 and H1–H6 in plain language; the full archive benchmark
   (`python scripts/benchmarks/clstm_full_archive.py`).

Build it with `nbformat` from a temporary script, then execute it in place:

```bash
conda run -n catraj jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 notebooks/04_clstm_prediction.ipynb
```

Requirements: executes top to bottom with no error and no warning noise; writes
nothing to disk except into a temporary directory for section 7; every number
quoted in markdown matches its code cell's output; under 5 MB; uses
`seed=0` everywhere so a re-run gives the same numbers on CPU.

---

## 10. Benchmark on the full archive: `scripts/benchmarks/clstm_full_archive.py`

This is where the GPU is used. A command-line script (argparse), in the style
of `paper_experiments.py`:

```bash
conda run -n catraj python scripts/benchmarks/clstm_full_archive.py \
    --start 2015-10-01 --end 2015-12-01 --device auto
```

Steps:

1. Read `nari_dynamic.csv` with `usecols` and the period filter (in chunks if
   memory needs it); convert to EPSG:32630; sort by `["sourcemmsi", "t"]`,
   stable.
2. Vessel type per MMSI from `nari_static.csv` (most frequent `shiptype` code),
   mapped to names as the sample's `vesselType` is (section 7). Null → "Other".
3. Split each vessel's track into trajectories at gaps over 30 minutes
   (`--max-gap 1800`); drop trajectories with fewer than 10 points.
4. Compress with `two_stage_cpls_compress` (README parameters), barriers from
   the full EEA coastline if `data/zenodo_1167595/C2_European_Coastline/`
   exists (as tutorial 3 section 8), otherwise the sample coastline, and say
   which in the output.
5. `make_samples` (`context_column` = the type); `split_by_trajectory`
   (`test_size=0.2`, `seed=0`).
6. Fit `ClstmPredictor()` and `ClstmPredictor(context=False)` on the device.
   Print the device and the GPU name.
7. Write to `results/04_clstm/`: `errors.csv` (curvilinear, ordinary LSTM,
   CLSTM: `prediction_errors` rows, with a `model` column), `summary.txt` (both
   `summary()` outputs, counts at every stage, timings), and the two saved
   models.

Report in your final report: the counts at every stage, the error table, the
relative improvement of CLSTM over the ordinary LSTM in mean point-wise error
(the paper's measure, 15.68 % on its data), and training times.

---

## 11. Documentation

Change only what CLSTM touches; keep each file's voice.

| file | change |
|---|---|
| `README.md` | Algorithm table: CLSTM status → **implemented**. The release sentence: add CLSTM prediction, and remove "The prediction algorithm below is being added in a later release." "Using it": a short `make_samples` → `ClstmPredictor` → `prediction_errors` example, with the `pip install catraj[predict]` line and the note on the CUDA build. Tutorial table: notebook 04. The benchmark's headline numbers from section 10, labelled with the period and data. |
| `CITATION.cff` | Abstract: one sentence on CLSTM. The 2021 reference's `notes`: CPLS, two-stage CPLS and CLSTM all implemented (in `catraj.compression` and `catraj.prediction`). Also add the 2023 reference's `notes` if it has none, saying CLSTM draws on it. Do not change `version`. |
| `docs/README.md` | Papers table, 2021 row: `CPLS, two-stage CPLS: catraj.compression; CLSTM: catraj.prediction`. |
| `docs/getting_the_data.md` | Section 1 table: notebook 04 (no download) and the benchmark (needs `annotation` target; `cpls` optional). |
| `scripts/README.md` | The benchmark row. |
| `results/README.md` | `04_clstm/`. |
| `pyproject.toml` | `predict = ["torch>=2.0"]` under optional dependencies, with a comment like the `app` one; add `predict` to `dev`; the `torch` marker; keywords `"trajectory prediction"`, `"LSTM"`. |

After editing, check every relative link resolves, and grep for `planned` and
`later release` to catch stale wording.

---

## 12. Work plan with gates

Do the steps in order. Do not start a step until the previous gate passes. One
commit per step.

**Step 0: baseline (no changes).** Read this brief, `two_stage.py`, `cadp.py`,
`_time.py`, `_crs.py`, `tests/test_two_stage.py`, and sections III-A, IV-B to
IV-E and V-B of the paper. Set up the environment (section 2).
*Gate:* `git status --short` is clean; `pytest -q` reports `447 passed`; torch
imports and reports CUDA available. If not, stop.

**Step 1: curvilinear model.** `curvilinear.py`, `prediction/__init__.py`,
`tests/test_curvilinear.py`. Tests first, watch them fail.
*Gate:* new tests pass; full suite passes.

**Step 2: samples and split.** `PredictionSamples`, `make_samples`,
`split_by_trajectory`, `prediction_errors`; their tests.
*Gate:* the hand-worked case gives exactly section 8's values; full suite passes.

**Step 3: networks.** `ClstmPredictor`, the extra, the marker, CI; the network
tests and the sample-data test.
*Gate:* every test of section 8 passes, R4 at the stated margins; with torch
blocked, the suite still passes (the torch tests skip).

**Step 4: notebook.**
*Gate:* executes cleanly, CPU, under 10 minutes; under 5 MB; `git status`
shows no stray files.

**Step 5: benchmark.** The script; run it on the GPU (check `nvidia-smi` first).
*Gate:* finishes; `results/04_clstm/` written; numbers recorded for the report.
Commit the script only.

**Step 6: documentation.** Section 11.
*Gate:* links resolve; no stale wording; full suite passes.

**Step 7: final check and report.** Section 13, then section 14.

---

## 13. Acceptance checklist

Run each item and record the actual result.

1. `pytest -q` → all pass; the count is 447 plus your new tests; nothing newly
   skipped (except torch tests in an environment without torch).
2. `pytest -q tests/test_curvilinear.py tests/test_clstm.py -v` → every test
   named in section 8 is present.
3. **Without torch:** in a fresh environment, or with `torch` blocked,
   `import catraj, catraj.prediction` works and `curvilinear_predict` runs.
4. **Portability.** No Python 3.11+ syntax; `git grep -n "\.union_all()" -- src tests`
   returns nothing.
5. **Full archive** (section 10): CLSTM's mean point-wise error is below the
   curvilinear model's. If it is not, stop and report before documenting. Its
   comparison with the ordinary LSTM is reported as it comes out, better or
   worse.
6. `help(catraj.prediction.clstm)` shows the module docstring with C1–C14 and
   H1–H6; `help(catraj.prediction.curvilinear)` shows Eqs. 8–12 and C5.
7. The notebook re-executes cleanly with the command in section 9.
8. `git status --short` is clean. Your commits sit on top of the commit that
   added this brief, none with an AI attribution line. Nothing was pushed.
9. `git diff <that commit> --stat` touches only files this brief names.

---

## 14. Final report

End with a report in this shape, in plain prose and short tables:

1. **Done**: one line per step of section 12, with its commit hash.
2. **Checklist**: each item of section 13 with the actual output or number.
3. **Benchmark**: section 10's table, counts, timings, GPU used.
4. **Deviations**: anything done differently from this brief, and why.
   "None" if none.
5. **For the author to confirm**: C5–C14 and H1–H6, one line each, since the
   docstrings say "decided here" until the author confirms them.
6. **Not done / open**: anything skipped, uncertain or worth a second look.
