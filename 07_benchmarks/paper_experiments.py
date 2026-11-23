"""TEMPORARY research script -- see 07_benchmarks/README.md; not part of catraj.

Re-run the CADP paper's experiments (Mehri et al. 2025, sections 3.2 and 3.4)
on the February 2016 AIS + WAVEWATCH III data, with both CADP modes.

Settings as written in the paper:
  3.2  epsilon = 1 km; wave direction thresholds 1, 1.5, 2, 2.5 deg (Table 1);
       wave height and bottom depth thresholds swept (Figure 2); DP baseline.
  3.4  epsilon = 10 m; wave height 0.15 m, wave direction 1.5 deg, bottom depth
       80 m; trajectories of 1000..100000 points; 10 repeats (Table 3).
Trajectories: one per vessel (MovingPandas TrajectoryCollection by MMSI).
DTW: on a fixed sample of 100 vessels (100-5000 points), same for every method.

usage: python 07_benchmarks/paper_experiments.py   (from the repository root)
"""
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{ROOT}/src")
from catraj.compression import Context, cadp_select, context_errors, douglas_peucker, dtw
from catraj.compression.cadp import _refine

a = pd.read_pickle(f"{ROOT}/08_data/cache/cadp_feb2016.pkl")
N = len(a)
mm = a.mmsi.to_numpy()
B = np.r_[np.flatnonzero(np.r_[True, mm[1:] != mm[:-1]]), N]
SEG = list(zip(B[:-1], B[1:]))
X, Y, T = a.x.to_numpy(), a.y.to_numpy(), a.unix_time.to_numpy(float)
VARS = {"wave_height": ("wave height", "m", False), "wave_direction": ("wave direction", "deg", True),
        "bottom_depth": ("bottom depth", "m", False)}
C = {v: a[v].to_numpy(float) for v in VARS}

rng = np.random.default_rng(0)
sizes = np.diff(B)
eligible = [i for i, n in enumerate(sizes) if 100 <= n <= 5000]
DTW_SAMPLE = sorted(rng.choice(eligible, 100, replace=False).tolist())


def dp_masks(eps):
    return [douglas_peucker(X[s:e], Y[s:e], eps) for s, e in SEG]


def total(ms):
    return int(sum(m.sum() for m in ms))


def R(ms):
    return 100.0 * (1 - total(ms) / N)


def dtw_sample(ms):
    out = 0.0
    for i in DTW_SAMPLE:
        s, e = SEG[i]
        xy = np.column_stack([X[s:e], Y[s:e]])
        out += dtw(xy, xy[ms[i]])
    return out / 1000.0  # km


def fidelity(ms, v):
    circ = VARS[v][2]
    errs = np.concatenate([context_errors(T[s:e], C[v][s:e], m, circular=circ)[~m]
                           for (s, e), m in zip(SEG, ms)])
    errs = errs[np.isfinite(errs)]
    return (np.sqrt(np.mean(errs**2)), errs.max()) if errs.size else (0.0, 0.0)


t0 = time.time()
EPS = 1000.0
DP = dp_masks(EPS)
print(f"{N:,} points, {len(SEG):,} trajectories (one per vessel); "
      f"DTW sample: {len(DTW_SAMPLE)} vessels, {sum(sizes[i] for i in DTW_SAMPLE):,} points\n")
print("=== 3.2  epsilon = 1 km ===")
print(f"DP alone: kept {total(DP):,}  R = {R(DP):.2f}%  (paper: 76.78%)  "
      f"DTW(sample) = {dtw_sample(DP):,.0f} km")

# Post mode: errors against DP once per variable, then any threshold is a comparison.
ERR = {v: [context_errors(T[s:e], C[v][s:e], m, circular=VARS[v][2])
           for (s, e), m in zip(SEG, DP)] for v in VARS}


def post_masks(v, thr):
    return [m | (np.nan_to_num(err, nan=0.0) > thr) for m, err in zip(DP, ERR[v])]


def iter_masks(v, thr, base=None):
    base = base or DP
    out = []
    for (s, e), m in zip(SEG, base):
        if e - s <= 2:
            out.append(m)
            continue
        keep, _, _ = _refine(T[s:e], [Context(C[v][s:e], thr, VARS[v][2])], ["c"], m)
        out.append(keep)
    return out


SWEEPS = {"wave_direction": [1.0, 1.5, 2.0, 2.5], "wave_height": [0.05, 0.1, 0.15, 0.25, 0.5],
          "bottom_depth": [10.0, 20.0, 40.0, 80.0, 160.0]}
for v, thrs in SWEEPS.items():
    name, unit, _ = VARS[v]
    print(f"\n-- {name} ({unit}) --  "
          + ("[Table 1: R 70.47 / 72.15 / 72.84 / 73.25 %]" if v == "wave_direction" else "[Figure 2]"))
    print(f"{'threshold':>10} | {'post: R%':>9} {'DTW km':>9} {'ctx RMSE':>9} {'ctx max':>8} | "
          f"{'iterative: R%':>13} {'DTW km':>9} {'ctx RMSE':>9} {'ctx max':>8}")
    for thr in thrs:
        po, it = post_masks(v, thr), iter_masks(v, thr)
        fp, fi = fidelity(po, v), fidelity(it, v)
        print(f"{thr:>10g} | {R(po):>9.2f} {dtw_sample(po):>9,.0f} {fp[0]:>9.3f} {fp[1]:>8.2f} | "
              f"{R(it):>13.2f} {dtw_sample(it):>9,.0f} {fi[0]:>9.3f} {fi[1]:>8.2f}")
    sys.stdout.flush()

print(f"\n({time.time() - t0:.0f}s)\n\n=== equal budget, epsilon = 1 km for CADP ===")


def calibrate(fn, target, lo, hi):
    best = None
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        ms = fn(mid)
        k = total(ms)
        if best is None or abs(k - target) < abs(best[1] - target):
            best = (mid, k, ms)
        if abs(k - target) <= 0.02 * target:
            break
        lo, hi = (mid, hi) if k > target else (lo, mid)
    return best


print(f"{'variable':<16}{'budget':>8} | {'ctx RMSE DP / post / iter':>28} | "
      f"{'ctx max DP / post / iter':>27} | {'DTW km DP / post / iter':>27}")
for target in (30_000, 100_000):
    dpb = calibrate(dp_masks, target, 0.01, 1000.0)
    dtw_dp = dtw_sample(dpb[2])
    for v in VARS:
        po = calibrate(lambda p: post_masks(v, p), target, 1e-4, 1e4)
        it = calibrate(lambda p: iter_masks(v, p), target, 1e-4, 1e4)
        f = [fidelity(dpb[2], v), fidelity(po[2], v), fidelity(it[2], v)]
        d = [dtw_dp, dtw_sample(po[2]), dtw_sample(it[2])]
        note = "" if all(abs(k - target) <= 0.05 * target for k in (po[1], it[1])) else \
            f"  (kept post {po[1]:,}, iter {it[1]:,})"
        print(f"{VARS[v][0]:<16}{target:>8,} | {f[0][0]:>8.3g} {f[1][0]:>9.3g} {f[2][0]:>9.3g} | "
              f"{f[0][1]:>8.3g} {f[1][1]:>8.3g} {f[2][1]:>8.3g} | "
              f"{d[0]:>8,.0f} {d[1]:>8,.0f} {d[2]:>8,.0f}{note}")
        sys.stdout.flush()

print(f"\n({time.time() - t0:.0f}s)\n\n=== 3.4  runtime: epsilon = 10 m; hs 0.15 m, dir 1.5 deg, "
      f"dpt 80 m; 10 repeats ===  [Table 3, ms: DP 14 / 66 / 150 / 778 / 3190]")
longest = int(np.argmax(sizes))
s0, _ = SEG[longest]
print(f"source: longest trajectory, {sizes[longest]:,} points; first n points used")
SET = {"wave_height": 0.15, "wave_direction": 1.5, "bottom_depth": 80.0}
cols = ["DP"] + [f"post {v}" for v in SET] + [f"iter {v}" for v in SET]
print(f"{'n':>7} " + " ".join(f"{c:>15}" for c in cols) + "   (mean +- std ms)")
for n in (1000, 5000, 10000, 50000, 100000):
    sl = slice(s0, s0 + n)
    x, y, t = X[sl], Y[sl], T[sl]
    row = []
    for c in cols:
        times = []
        for _ in range(10):
            t1 = time.perf_counter()
            if c == "DP":
                douglas_peucker(x, y, 10.0)
            else:
                mode, v = c.split()
                cadp_select(x, y, t, 10.0, contexts=[Context(C[v][sl], SET[v], VARS[v][2])],
                            mode="post" if mode == "post" else "iterative")
            times.append((time.perf_counter() - t1) * 1000)
        row.append(f"{np.mean(times):7.1f} +-{np.std(times):5.1f}")
    print(f"{n:>7} " + " ".join(f"{r:>15}" for r in row))
    sys.stdout.flush()
print(f"\n(total {time.time() - t0:.0f}s)")
