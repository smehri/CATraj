"""TEMPORARY research script -- see scripts/benchmarks/README.md; not part of catraj.

Build the CADP paper's dataset: February 2016 AIS annotated with sea state.

AIS:   [P1] nari_dynamic.csv, Feb 2016 UTC, duplicates and "not available"
       codes removed, projected to UTM 30N.
Ocean: [E1] oc_february.csv (WAVEWATCH III, SHOM-IFREMER), 3-hourly, 43,206
       sea points.  Annotated as catraj.annotation does with IDW(k=4, power=2)
       across the grid and DLT through time, bearings on the unit circle --
       vectorised, because the grid is fixed: neighbours and weights are found
       once per AIS position, not refitted per epoch.

usage: python scripts/benchmarks/prepare_feb2016.py   (from the repository root)
"""
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2])
SRC = f"{ROOT}/data/zenodo_1167595"
OUT = f"{ROOT}/data/cache/cadp_feb2016.pkl"
LO, HI = pd.Timestamp("2016-02-01").value // 10**9, pd.Timestamp("2016-03-01").value // 10**9
to_utm = Transformer.from_crs(4326, 32630, always_xy=True)
t0 = time.time()

# ---------------------------------------------------------------- AIS
a = pd.read_csv(f"{SRC}/P1_AIS_Data/nari_dynamic.csv")
a = a[(a.t >= LO) & (a.t < HI)]
n_raw = len(a)
a = a.drop_duplicates(["sourcemmsi", "t"]).sort_values(["sourcemmsi", "t"]).reset_index(drop=True)
a["speed"] = a.speedoverground.where(a.speedoverground < 102.3)
a["cog"] = a.courseoverground.where(a.courseoverground < 360)
a["heading"] = a.trueheading.where(a.trueheading < 360).astype(float)
a["x"], a["y"] = to_utm.transform(a.lon.to_numpy(), a.lat.to_numpy())
print(f"AIS Feb 2016: {n_raw:,} rows -> {len(a):,} after de-duplication, "
      f"{a.sourcemmsi.nunique():,} vessels  ({time.time() - t0:.0f}s)")

# -------------------------------------------------------------- ocean
oc = pd.read_csv(f"{SRC}/E1_ocean/oc_february.csv",
                 dtype={"lon": "float64", "lat": "float64", "dpt": "float32", "wlv": "float32",
                        "hs": "float32", "lm": "float32", "dir": "float32", "ts": "int64"})
sea = oc.dpt > -16000
oc = oc[sea]
oc.loc[oc.hs < 0, "hs"] = np.nan
oc.loc[oc["dir"] < 0, "dir"] = np.nan
oc.loc[oc.wlv < -300, "wlv"] = np.nan
grid = oc[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
grid["gid"] = np.arange(len(grid))
oc = oc.merge(grid, on=["lon", "lat"])
epochs = np.sort(oc.ts.unique())
print(f"ocean: {len(grid):,} sea points x {len(epochs)} epochs "
      f"({pd.to_datetime(epochs[0], unit='s')} .. {pd.to_datetime(epochs[-1], unit='s')})")

gx, gy = to_utm.transform(grid.lon.to_numpy(), grid.lat.to_numpy())
V = {}
for var in ("hs", "dir", "wlv", "dpt"):
    arr = np.full((len(epochs), len(grid)), np.nan, dtype="float64")
    arr[np.searchsorted(epochs, oc.ts.to_numpy()), oc.gid.to_numpy()] = oc[var].to_numpy()
    V[var] = arr

# ---------------------------------------------------- spatial weights
K, POWER = 4, 2.0
dist, idx = cKDTree(np.column_stack([gx, gy])).query(np.column_stack([a.x, a.y]), k=K)
exact = dist[:, 0] == 0
w = np.where(exact[:, None], 0.0, 1.0 / np.maximum(dist, 1e-12) ** POWER)
w[exact, 0] = 1.0
a["dist_to_grid_m"] = dist[:, 0]

# --------------------------------------------- temporal bracketing (DLT)
t = a.t.to_numpy(float)
k = np.clip(np.searchsorted(epochs, t, side="right") - 1, 0, len(epochs) - 2)
frac = np.clip((t - epochs[k]) / (epochs[k + 1] - epochs[k]), 0.0, 1.0)


def idw_at(values_at_epoch_rows):
    """IDW over the K neighbours, ignoring neighbours with no value."""
    vals = values_at_epoch_rows
    ok = np.isfinite(vals)
    ww = np.where(ok, w, 0.0)
    s = ww.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, np.nansum(np.where(ok, vals, 0.0) * ww, axis=1) / s, np.nan)


def annotate(var, circular=False):
    arr = V[var]
    if circular:
        # As catraj does: spatial stage on components -> a bearing per epoch;
        # temporal stage on the unit vectors of those bearings.
        rad = np.radians(arr)
        cos, sin = np.cos(rad), np.sin(rad)
        ang = [np.arctan2(idw_at(sin[e, idx]), idw_at(cos[e, idx]))
               for e in (k[:, None], k[:, None] + 1)]
        u = np.cos(ang[0]) + (np.cos(ang[1]) - np.cos(ang[0])) * frac
        v = np.sin(ang[0]) + (np.sin(ang[1]) - np.sin(ang[0])) * frac
        return np.mod(np.degrees(np.arctan2(v, u)), 360.0)
    lo, hi = idw_at(arr[k[:, None], idx]), idw_at(arr[k[:, None] + 1, idx])
    return lo + (hi - lo) * frac


for var, circ in (("dpt", False), ("hs", False), ("dir", True), ("wlv", False)):
    a[var] = annotate(var, circ)
    print(f"annotated {var:<4} non-null {a[var].notna().mean():6.1%}  "
          f"range {np.nanmin(a[var]):.2f} .. {np.nanmax(a[var]):.2f}  ({time.time() - t0:.0f}s)")

# ------------------------------ cross-check against catraj.annotation itself
from catraj.annotation import DLT, IDW, angular_residual, annotate_points

rng = np.random.default_rng(0)
sample = rng.choice(len(a), 300, replace=False)
for var, circ in (("hs", False), ("dir", True)):
    st = pd.DataFrame({"x": np.tile(gx, len(epochs)), "y": np.tile(gy, len(epochs)),
                       "t": np.repeat(epochs.astype(float), len(grid)),
                       "value": V[var].ravel()})
    st = st[st.t.between(epochs[k[sample]].min(), epochs[k[sample] + 1].max())]
    ref = annotate_points(st, a.x.to_numpy()[sample], a.y.to_numpy()[sample], t[sample],
                          spatial_method=IDW(k=K, power=POWER, circular=circ),
                          temporal_method=DLT(circular=circ), circular=circ,
                          filter_window=None)  # compared with an unfiltered reference
    mine = a[var].to_numpy()[sample]
    diff = np.abs(angular_residual(mine, ref)) if circ else np.abs(mine - ref)
    print(f"check {var}: max |vectorised - annotate_points| over 300 points = {np.nanmax(diff):.2e}")

cols = ["sourcemmsi", "t", "x", "y", "lon", "lat", "navigationalstatus", "speed", "cog",
        "heading", "rateofturn", "dpt", "hs", "dir", "wlv", "dist_to_grid_m"]
NAMES = {  # the archive's coded names -> what they measure
    "sourcemmsi": "mmsi", "t": "unix_time", "navigationalstatus": "navigational_status",
    "speed": "speed_over_ground", "cog": "course_over_ground", "heading": "true_heading",
    "rateofturn": "rate_of_turn", "dpt": "bottom_depth", "hs": "wave_height",
    "dir": "wave_direction", "wlv": "sea_level", "dist_to_grid_m": "distance_to_grid_m"}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
a[cols].rename(columns=NAMES).to_pickle(OUT)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
