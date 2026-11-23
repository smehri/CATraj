"""TEMPORARY research script -- see 07_benchmarks/README.md; not part of catraj.

Equal-budget comparison over every usable context variable.

post vs iterative CADP vs plain DP, each tuned to keep the same number of
points fleet-wide; then DTW and context fidelity (error of dropped points
against the final trajectory) are compared.  Also: nominal navigation status,
and sensitivity to the spatial epsilon.
"""
import os
import sys
from pathlib import Path
import warnings

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1])
from catraj.annotation import IDW, Spline, annotate_points, decode_compass
from catraj.compression import Context, cadp_select, context_errors, douglas_peucker, dtw

# The sample's coded column names -> what they measure
AIS_NAMES = {"sourcemmsi": "mmsi", "t": "unix_time", "navigation": "navigational_status",
             "speedoverg": "speed_over_ground", "courseover": "course_over_ground",
             "trueheadin": "true_heading", "rateofturn": "rate_of_turn", "vesselType": "vessel_type"}
WEATHER_NAMES = {"id_station": "station_id", "local_time": "unix_time", "T": "air_temperature",
                 "P": "air_pressure", "U": "relative_humidity", "Td": "dew_point",
                 "VV": "visibility", "Ff": "wind_speed", "id_windDir": "wind_direction_code"}
ais = (gpd.read_file(f"{ROOT}/08_datasets/AIS/AIS_DLT_UTM.shp").rename(columns=AIS_NAMES)
          .sort_values(["mmsi", "unix_time"]))
w = gpd.read_file(f"{ROOT}/08_datasets/context/Weather/Weather_Observation_UTM.shp").rename(
    columns=WEATHER_NAMES)

# AIS "not available" codes -> NaN
ais["speed_over_ground"] = ais.speed_over_ground.where(ais.speed_over_ground < 102.3)
ais["course_over_ground"] = ais.course_over_ground.where(ais.course_over_ground < 360)
ais["true_heading"] = ais.true_heading.where(ais.true_heading < 360)
ais["rate_of_turn"] = ais.rate_of_turn.where(ais.rate_of_turn.abs() < 127)

# Weather variables annotated onto every AIS point (IDW + spline)
WEATHER = {"air_temperature": False, "air_pressure": False, "relative_humidity": False,
           "dew_point": False, "visibility": False, "wind_speed": False, "wind_direction": True}
w["wind_direction"] = decode_compass(w.wind_direction_code.astype(float))
for col, circ in WEATHER.items():
    st = pd.DataFrame({"x": w.geometry.x, "y": w.geometry.y,
                       "t": w.unix_time.astype(float), "value": w[col].astype(float)})
    ais[col] = annotate_points(st, ais.geometry.x, ais.geometry.y, ais.unix_time.astype(float),
                               spatial_method=IDW(circular=circ),
                               temporal_method=Spline(circular=circ), circular=circ)

VARS = [  # name, unit, circular, domain
    ("air_temperature", "degC", False, "env"), ("air_pressure", "mmHg", False, "env"),
    ("relative_humidity", "%", False, "env"), ("dew_point", "degC", False, "env"),
    ("visibility", "km", False, "env"), ("wind_speed", "m/s", False, "env"),
    ("wind_direction", "deg", True, "env"),
    ("speed_over_ground", "kn", False, "int"), ("course_over_ground", "deg", True, "int"),
    ("true_heading", "deg", True, "int"), ("rate_of_turn", "AIS", False, "int"),
]

G = []
for _, g in ais.groupby("mmsi"):
    if len(g) > 2:
        d = dict(x=g.geometry.x.to_numpy(), y=g.geometry.y.to_numpy(),
                 t=g.unix_time.to_numpy(float), nav=g.navigational_status.to_numpy())
        for v, *_ in VARS:
            d[v] = g[v].to_numpy(float)
        G.append(d)
M = sum(len(g["x"]) for g in G)


def masks_for(method, var=None, p=None, circ=False, eps=1000.0):
    if method == "DP":
        return [douglas_peucker(g["x"], g["y"], p) for g in G]
    return [cadp_select(g["x"], g["y"], g["t"], eps, contexts=[Context(g[var], p, circ)],
                        mode=method).mask for g in G]


kept = lambda ms: int(sum(m.sum() for m in ms))


def calibrate(fn, target, lo, hi):
    best = None
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        ms = fn(mid)
        k = kept(ms)
        if best is None or abs(k - target) < abs(best[1] - target):
            best = (mid, k, ms)
        if abs(k - target) <= 0.01 * target:
            break
        lo, hi = (mid, hi) if k > target else (lo, mid)
    return best


def fidelity(ms, var, circ):
    e = np.concatenate([context_errors(g["t"], g[var], m, circular=circ)[~m]
                        for g, m in zip(G, ms)])
    e = e[np.isfinite(e)]
    return np.sqrt(np.mean(e**2)), e.max()


def total_dtw(ms):
    return sum(dtw(np.column_stack([g["x"], g["y"]]),
                   np.column_stack([g["x"][m], g["y"][m]])) for g, m in zip(G, ms)) / 1000


BUDGETS = [1500, 6000]
dp_at = {b: calibrate(lambda p: masks_for("DP", p=p), b, 1e-2, 5e4) for b in BUDGETS}
dtw_dp = {b: total_dtw(dp_at[b][2]) for b in BUDGETS}

print(f"{M} points, {len(G)} vessels.  CADP spatial eps = 1000 m (DP alone keeps "
      f"{kept(masks_for('DP', p=1000.0))}).  Budgets: {BUDGETS}.")
for v, *_ in VARS:
    pass
print("missing values after cleaning: " + ", ".join(
    f"{v} {100 * ais[v].isna().mean():.0f}%" for v, *_ in VARS))
print()
hdr = (f"{'variable':<19}{'dom':<4}{'budget':>7} | {'ctx RMSE  DP / post / iter':>30} | "
       f"{'ctx max  DP / post / iter':>28} | {'DTW km  DP / post / iter':>29}")
print(hdr)
print("-" * len(hdr))
summary = []
for v, unit, circ, dom in VARS:
    for b in BUDGETS:
        r = {}
        for method in ("post", "iterative"):
            p, k, ms = calibrate(lambda p: masks_for(method, v, p, circ), b, 1e-5, 1e4)
            r[method] = (*fidelity(ms, v, circ), total_dtw(ms), k)
        dpr = (*fidelity(dp_at[b][2], v, circ), dtw_dp[b])
        po, it = r["post"], r["iterative"]
        print(f"{v:<19}{dom:<4}{b:>7} | {dpr[0]:>8.3g} {po[0]:>9.3g} {it[0]:>9.3g}   | "
              f"{dpr[1]:>8.3g} {po[1]:>8.3g} {it[1]:>8.3g}   | "
              f"{dpr[2]:>8,.0f} {po[2]:>9,.0f} {it[2]:>9,.0f}"
              + ("" if abs(po[3] - b) < 0.05 * b and abs(it[3] - b) < 0.05 * b
                 else f"   (kept post {po[3]}, iter {it[3]})"))
        summary.append((v, b, dpr, po, it))

print("\nWins for iterative over post (lower is better):")
for name, idx in [("context RMSE", 0), ("context max", 1), ("DTW", 2)]:
    wins = sum(it[idx] < po[idx] for *_, po, it in summary)
    print(f"  {name:<13} {wins} / {len(summary)}")
print("Wins for iterative over plain DP on context RMSE: "
      f"{sum(it[0] < dpr[0] for _, _, dpr, _, it in summary)} / {len(summary)}")

# Nominal: navigation status boundaries on top of DP at 1 km
dp1 = masks_for("DP", p=1000.0)
nom = [cadp_select(g["x"], g["y"], g["t"], 1000.0, nominal={"nav": g["nav"]}).mask for g in G]
changes = sum(int((np.diff(pd.factorize(g["nav"])[0]) != 0).sum()) for g in G)
print(f"\nNominal navigation status: {changes} status changes in the fleet; boundaries add "
      f"{kept(nom) - kept(dp1)} points to DP's {kept(dp1)}.")

# Sensitivity to the spatial epsilon (speed, budget 6000)
print("\nSpatial-epsilon sensitivity -- speed, budget 6000:")
print(f"{'eps':>7} {'mode':<10} {'DP part':>8} {'thr':>8} {'kept':>6} {'ctx RMSE':>9} "
      f"{'ctx max':>8} {'DTW km':>9}")
for eps in (250.0, 1000.0, 4000.0):
    base = kept(masks_for("DP", p=eps))
    for method in ("post", "iterative"):
        p, k, ms = calibrate(lambda p: masks_for(method, "speed_over_ground", p, False, eps), 6000, 1e-5, 1e4)
        rmse, mx = fidelity(ms, "speed_over_ground", False)
        print(f"{eps:>7.0f} {method:<10} {base:>8} {p:>8.3g} {k:>6} {rmse:>9.3f} {mx:>8.2f} "
              f"{total_dtw(ms):>9,.0f}")
