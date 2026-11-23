"""CADP over the whole annotated fleet.

Reads the annotated archive written by ``05_tutorials/01_trajectory_annotation.ipynb``
(every AIS position with the weather it met) and compresses each vessel's
trajectory with Context-Aware Douglas--Peucker, in both modes, writing the
result to ``10_results/02_cadp_compression/``.

usage (from the repository root, in the catraj environment)::

    python 06_workflows/01_cadp_on_annotated_fleet.py

Settings are the constants below; every run records them in ``summary.json``.

Outputs
-------
cadp_post.parquet, cadp_iterative.parquet
    The retained positions, every input column kept, plus ``kept_by``: ``dp``
    for points Douglas--Peucker keeps, otherwise the context variable(s) that
    required the point.
per_vessel.csv
    Positions in and out per vessel, for Douglas--Peucker alone and both modes.
summary.json
    Settings, totals, compression ratios, run time, and the context fidelity
    of each result: for every dropped point, how far its context lies from
    the interpolation between its final neighbours (RMSE and maximum).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))
from catraj.compression import Context, cadp_select, compression_ratio, context_errors  # noqa: E402

# ---- settings -------------------------------------------------------------------------
SOURCE = ROOT / "10_results" / "01_trajectory_annotation" / "06_fleet_annotated.parquet"
OUT = ROOT / "10_results" / "02_cadp_compression"
EPSILON = 1_000.0                     # metres, the spatial threshold (as in the CADP paper)
THRESHOLDS = {                        # each in its variable's own units
    "wind_speed": 1.0,                # m/s
    "air_temperature": 0.5,           # degrees C
    "air_pressure": 1.0,              # mm of mercury
    "wind_direction": 10.0,           # degrees, on the circle
    "speed_over_ground": 1.0,         # knots -- internal context
}
CIRCULAR = {"wind_direction"}
MODES = ("post", "iterative")


def main():
    started = time.perf_counter()
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE.relative_to(ROOT)} -- run "
                         "05_tutorials/01_trajectory_annotation.ipynb with EXPORT_FLEET = True")
    fleet = pd.read_parquet(SOURCE).sort_values(["mmsi", "time"], kind="stable")
    fleet = fleet.reset_index(drop=True)
    n = len(fleet)
    t_all = ((fleet.time - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
    x_all, y_all = fleet.x.to_numpy(), fleet.y.to_numpy()
    ctx_all = {v: fleet[v].to_numpy(dtype=float) for v in THRESHOLDS}
    mmsi = fleet.mmsi.to_numpy()
    bounds = np.r_[0, np.flatnonzero(mmsi[1:] != mmsi[:-1]) + 1, n]
    print(f"{n:,} positions, {len(bounds) - 1:,} vessels  ({time.perf_counter() - started:.0f} s)")

    keep = {m: np.zeros(n, dtype=bool) for m in MODES}
    dp_keep = np.zeros(n, dtype=bool)
    reason = {m: np.full(n, "", dtype=object) for m in MODES}
    errors = {m: {v: [] for v in THRESHOLDS} for m in MODES}
    timing = {m: 0.0 for m in MODES}

    for s, e in zip(bounds[:-1], bounds[1:]):
        sl = slice(s, e)
        contexts = [Context(ctx_all[v][sl], thr, v in CIRCULAR, v) for v, thr in THRESHOLDS.items()]
        for mode in MODES:
            t0 = time.perf_counter()
            sel = cadp_select(x_all[sl], y_all[sl], t_all[sl], EPSILON, contexts=contexts, mode=mode)
            timing[mode] += time.perf_counter() - t0
            mask = sel.mask
            keep[mode][sl] = mask
            dp_keep[sl] = sel.dp
            why = np.where(sel.dp, "dp", "").astype(object)
            for name, flagged in sel.context.items():
                extra = flagged & ~sel.dp
                why[extra] = np.where(why[extra] == "", name, why[extra] + "," + name)
            reason[mode][sl] = why
            for v in THRESHOLDS:
                err = context_errors(t_all[sl], ctx_all[v][sl], mask, circular=v in CIRCULAR)[~mask]
                errors[mode][v].append(err[np.isfinite(err)])

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "epsilon_m": EPSILON, "thresholds": THRESHOLDS, "circular": sorted(CIRCULAR),
        "positions": int(n), "vessels": int(len(bounds) - 1),
        "douglas_peucker": {"kept": int(dp_keep.sum()),
                            "compression_ratio_pct": round(compression_ratio(dp_keep.sum(), n), 3)},
    }
    per_vessel = pd.DataFrame({"mmsi": mmsi[bounds[:-1]],
                               "vessel_type": fleet.vessel_type.to_numpy()[bounds[:-1]],
                               "positions": np.diff(bounds),
                               "dp_kept": np.add.reduceat(dp_keep, bounds[:-1])})
    for mode in MODES:
        out = fleet[keep[mode]].assign(kept_by=reason[mode][keep[mode]])
        path = OUT / f"cadp_{mode}.parquet"
        out.to_parquet(path, index=False)
        per_vessel[f"{mode}_kept"] = np.add.reduceat(keep[mode], bounds[:-1])
        fidelity = {}
        for v in THRESHOLDS:
            err = np.concatenate(errors[mode][v])
            fidelity[v] = {"threshold": THRESHOLDS[v],
                           "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4) if err.size else 0.0,
                           "max": round(float(err.max()), 4) if err.size else 0.0}
        reasons = pd.Series(reason[mode][keep[mode]]).value_counts().head(12)
        summary[mode] = {
            "kept": int(keep[mode].sum()),
            "compression_ratio_pct": round(compression_ratio(keep[mode].sum(), n), 3),
            "added_for_context": int((keep[mode] & ~dp_keep).sum()),
            "seconds": round(timing[mode], 1),
            "file": path.relative_to(ROOT).as_posix(),
            "file_mb": round(path.stat().st_size / 2 ** 20, 1),
            "context_fidelity_of_dropped_points": fidelity,
            "kept_by_top": {k: int(c) for k, c in reasons.items()},
        }
        print(f"{mode:<10} kept {keep[mode].sum():>10,}  R = {summary[mode]['compression_ratio_pct']:6.2f}%  "
              f"({timing[mode]:.0f} s)  -> {path.relative_to(ROOT).as_posix()}")
    per_vessel["post_ratio_pct"] = (100 * (1 - per_vessel.post_kept / per_vessel.positions)).round(2)
    per_vessel["iterative_ratio_pct"] = (100 * (1 - per_vessel.iterative_kept / per_vessel.positions)).round(2)
    per_vessel.sort_values("positions", ascending=False).to_csv(OUT / "per_vessel.csv", index=False)
    summary["total_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Douglas-Peucker alone keeps {dp_keep.sum():,} "
          f"(R = {summary['douglas_peucker']['compression_ratio_pct']:.2f}%)")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}/  ({summary['total_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
