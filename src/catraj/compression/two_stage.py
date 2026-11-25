"""Two-stage CPLS (2stage-CPLS) -- [Mehri2021] section IV-A, Algorithm 2.

A vessel at anchor is a cloud of GPS jitter.  Compressed as it is, the cloud
keeps its points -- CPLS keeps the crossings of the jitter (see
:mod:`.cpls`, "What it costs") -- or, if it is simplified away, "it appears as
if the vessel moves slowly, whereas in the original uncompressed trajectory
the vessel stops moving for a period of time" [58].  2stage-CPLS replaces each
stay by its centroid first, finds where the speed changes, and then runs CPLS
on what is left.

The method, as Algorithm 2 states it (cleaned of its typesetting)::

    walk T; consecutive stay points are collected, replaced by their centroid,
        and T is split into pieces Ti at the stays                (lines 1-10)
    Tv = CPLS_Ev(V, Theta_v)       # the speed series, error Ev     (line 11)
    for consecutive vertices i, j of Tv:
        Tc += CPLS(T(i..j), Theta_d)                             (lines 12-15)
    return Tc

where ``Ev = |v_i - v'_i|`` (Eq. 6) and ``v'_i`` is the speed interpolated
between the ends of the segment.  The text: stay points are those whose AIS
status says the vessel is at anchor; "the vessels' speed that is estimated at
the trajectory simplification stage ... the ratio between distance and time
interval".  The experiments use Theta_v = 1.54 m/s (3 knots).

Decisions taken with the author (2026-11-24)
--------------------------------------------

* **A1. Stay points: status or movement.**  "the method starts to check the
  status field of each point ... the status value indicates that the vessel is
  at anchor" (Fig. 4, after [68]).  A point is a stay point if its status code
  is in ``stay_status`` (default ``(1, 5)``: at anchor, moored) **or** it
  belongs to a movement stay: Li et al.'s stay-point detection, as surveyed in
  Yu (2015), the paper's reference [68].  From an anchor ``i`` take the
  longest run ``i..j-1`` of points all within ``stay_radius`` of ``p_i``; if
  ``t[j-1] - t[i] >= stay_duration`` they are a stay and the search resumes at
  ``j``, otherwise at ``i + 1``.  (Li's loop forgets the run that reaches the
  end of the trajectory; here it is tested like any other -- decided here.)
  Either source can be off: no status given, no status; ``stay_radius`` and
  ``stay_duration`` both ``None``, no movement detection; one without the
  other raises.  The paper gives no thresholds, so there are no defaults.  A
  ready boolean mask, ``stays=``, overrides both.  A *stay* is a maximal run
  of consecutive stay points.
* **A2. A stay becomes two vertices at its centroid: arrival and departure.**
  "it computes the centroid of p_i, ..., p_i+k.  Those points are eliminated
  and substituted by their centroid."  The centroid is the mean of the run's
  projected x and y.  The run's first and last points are kept, moved to the
  centroid, with their own times, so that the duration of the stop survives --
  the reason de Vries [58] introduced the two-stage method.  The points
  between are dropped.  A one-point stay is one vertex, unmoved.  In the
  dataframe result these are the original rows (every column, index label and
  time kept) with only the geometry replaced, and a boolean column marks them.
* **A3. Speed comes from positions and time.**  On the collapsed track, ``v_i``
  is the central difference ``|p[i+1] - p[i-1]| / (t[i+1] - t[i-1])``, one-sided
  at the two ends, in position units per time unit (metres per second for UTM
  and UNIX seconds; ``speed_epsilon`` is in the same units).  Line 11 is
  Douglas--Peucker on that series with the vertical error ``|v_i - v'_i|``,
  ``v'_i`` interpolated **in time** between the segment's ends (de Vries' PLS
  on the speed series), run once over the whole collapsed track.  Its
  breakpoints are the speed vertices.
* **A4. The signature is that of the collapsed trajectory; nothing but the
  centroid remains of a stay.**  Line 14 applies CPLS to ``T(i..j)``, a piece
  of the collapsed track.  So the forced vertices -- the first and last
  points, both ends of every stay, every speed breakpoint -- are never
  dropped; Douglas--Peucker with ``epsilon`` runs in each piece between
  consecutive forced vertices; and the CPLS topological pass of
  :mod:`.cpls` (all five relations, the same switches) then runs once over the
  whole collapsed track, sub-paths and chords alike taken from its
  coordinates.  A stay is one zero-length segment there (D11 of :mod:`.cpls`
  makes it a point) and is never a candidate.

* **P1. Filtering is on by default.**  The positions are median-filtered
  over a centred time window of ``filter_window`` seconds, 30 by default, by
  :func:`catraj.preprocessing.denoise`; ``filter_window=None`` switches it
  off and restores the unfiltered results exactly.  Time comes from
  ``time_column`` or a DatetimeIndex; any other index raises a ``ValueError``.
  This is the dataframe-level step, :func:`two_stage_cpls_compress`; the
  array-level :func:`two_stage_cpls_select` filters no positions.
* **P2. Sequential.**  The positions are filtered first.  The speed series
  (A3) is computed from the *filtered* positions, and is then median-filtered
  itself, in time, with the same window (``speed_window`` of
  :func:`two_stage_cpls_select`, the one array-level filter, because the
  speed series exists only inside it) before line 11's Douglas--Peucker.  Stay
  detection (A1) and everything after run on the filtered positions.  The
  result rows carry the filtered geometry (a stay's two ends, the centroid of
  the filtered points).

Decided here
------------

* **B4. Times** must not decrease along a trajectory (else ``ValueError``).
  Repeated times are allowed: where the central difference has a zero time
  span, ``v_i`` is undefined and takes the value of the nearest defined
  neighbour (the earlier one on a tie; all undefined, zero).  If a speed
  segment has zero time span, ``v'`` is interpolated by position in the
  sequence.
* The speed series is read on the collapsed track, so a stay's two ends carry
  the speed of a vessel that arrives, waits and leaves; ``speed_epsilon=inf``
  switches the speed stage off (first and last points aside).
* The first and last points are reported under ``speed`` (the speed series'
  own end points), unless they are stay ends.
* The pieces of the Douglas--Peucker stage are the stretches between forced
  vertices, not the Douglas--Peucker segments of the whole track, so
  ``epsilon`` bounds each piece on its own.

What the result guarantees
--------------------------

Relative to the collapsed trajectory -- which differs from the raw one only
inside stays, where the stay's points are replaced by its centroid -- the
guarantees of :mod:`.cpls` hold: every unit keeps the DE-9IM relation and
nine direction cells of its sub-path for every neighbouring barrier, so
compression introduces no crossing (``segment_crossings(sel.x, sel.y, ...,
keep=sel.mask, new_only=True)`` measures it on the collapsed coordinates);
units meet if and only if their sub-paths do; no two consecutive units invert.
Every forced vertex is kept, and a stay keeps exactly its two centroid
vertices: no other point of a stay run is ever kept.

Against the *raw* track the guarantee is weaker, by design: the collapsed
track moves each stay's points to the centroid, by up to the stay's extent.
A unit's raw sub-path can therefore differ from its chord in its barrier
relations, its self-intersections and its inversions, and only at a stay.
The collapsed track is the reference the algorithm keeps; the raw one is not.
(These differences were counted on an earlier, 24-hour sample; they have not
been re-counted on the two-day sample, so no figures are quoted.)

A unit of one segment of the collapsed track is never a candidate, so it is
never tested against the unit before it for inversion; it is the only
exemption, and the only way a unit pair can invert against the collapsed
track.

What it costs
-------------

Collapsing stays removes the jitter that makes plain CPLS keep so many points,
if the stays are found.  On the two-day sample (2015-11-09 and 10: 237,318
positions of 565 vessels, one trajectory per vessel, the sample coastline,
``epsilon`` 1000 m, ``speed_epsilon`` 1.54 m/s, all five relations on, one
core), counting a stay as a run of consecutive stay points, status alone finds
56 stays (12,436 points) and movement detection added (100 m / 600 s) 242
(122,746 points).  On the first 60 vessels by MMSI (33,943 positions):

=============================================  ========  =======
setting                                        kept      seconds
=============================================  ========  =======
plain CPLS                                       19,099       13
2stage, status only                              19,282       12
2stage, status + 100 m / 600 s                    3,120        3
=============================================  ========  =======

On the whole sample, with movement detection (status + 100 m / 600 s),
2stage-CPLS keeps 16,889 of 237,318 points in 35 s with ``filter_window=None``
and 15,674 in 37 s with the default filter (positions 30 s median, speed
series 30 s median); plain CPLS did not finish on the whole sample in 10
minutes (barrier relations alone keep 2,457 points, 163 s).  The unfiltered
rows above were measured with ``filter_window=None``.  Without the filter on
the speed series (``speed_window=None``) the speed series swings between zero
and double where consecutive points fall in the same window, so the speed
series needs the filter too (see :mod:`catraj.preprocessing.filtering`).

Status alone finds little: most moored vessels in the sample do not report
"at anchor", so the jitter that costs the points is mostly not found.
Movement detection finds it.  Seconds are single-threaded wall time.

References
----------
[Mehri2021] Mehri, S., Alesheikh, A. A., Basiri, A. (2021).  A Contextual
    Hybrid Model for Vessel Movement Prediction.  IEEE Access 9, 45600-45613.
[58] de Vries, G. K. D., van Someren, M. (2012).  Machine learning for vessel
    trajectories using compression, alignments and domain knowledge.  Expert
    Systems with Applications 39(18), 13426-13439.
[68] Yu, Y. (2015).  Trajectory data mining: an overview, which surveys the
    stay-point detection of Li et al.; the paper's source for Fig. 4.
Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., Ma, W.-Y. (2008).  Mining user
    similarity based on location history.  ACM SIGSPATIAL GIS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._crs import _check_crs
from .._time import times_of
from .cpls import (
    RULES,
    _RULE_NAMES,
    _as_barriers,
    _Stage2,
)
from ..preprocessing.filtering import denoise, rolling_filter
from .._parallel import map_groups, parallel_map
from .douglas_peucker import douglas_peucker

__all__ = [
    "TwoStageSelection",
    "stay_points",
    "two_stage_cpls_select",
    "two_stage_cpls_compress",
]


def _runs(mask):
    """First and last index of every maximal run of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2] - 1


def _movement_stays(x, y, t, radius, duration):
    """Li et al.'s stay-point detection (A1); the tail run is tested too."""
    n = x.size
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j, step = i + 1, 64
        while j < n:  # first point beyond the radius, found in growing blocks
            hi = min(j + step, n)
            far = np.flatnonzero(np.hypot(x[j:hi] - x[i], y[j:hi] - y[i]) > radius)
            if far.size:
                j += int(far[0])
                break
            j, step = hi, step * 2
        if t[j - 1] - t[i] >= duration:
            out[i:j] = True
            i = j
        else:
            i += 1
    return out


def stay_points(
    x, y, t, *, stay_radius=None, stay_duration=None, status=None,
    stay_status=(1, 5),
):
    """Which points are stay points (decision A1).

    A point is a stay point if its ``status`` code is in ``stay_status``, or
    if it belongs to a movement stay: from an anchor ``i``, the longest run of
    points all within ``stay_radius`` of it, when it lasts at least
    ``stay_duration`` (Li et al., as surveyed in Yu 2015).

    Parameters
    ----------
    x, y, t : array-like
        Positions in projected units and times in trajectory order.
    stay_radius, stay_duration : float, optional
        Both or neither: movement detection is off without them.
    status : array-like, optional
        The AIS navigational status of each point; unused when omitted.
    stay_status : iterable
        Status codes that mean the vessel is not under way.

    Returns
    -------
    ndarray of bool
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if not (x.size == y.size == t.size):
        raise ValueError("x, y and t must have equal length")
    if (stay_radius is None) != (stay_duration is None):
        raise ValueError("stay_radius and stay_duration must be given together")
    out = np.zeros(x.size, dtype=bool)
    if status is not None:
        status = np.asarray(status, dtype=float).ravel()
        if status.size != x.size:
            raise ValueError("status must have one value per point")
        out |= np.isin(status, np.asarray(list(stay_status), dtype=float))
    if stay_radius is not None:
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("positions must be finite; drop missing points first")
        out |= _movement_stays(x, y, t, float(stay_radius), float(stay_duration))
    return out


@dataclass
class TwoStageSelection:
    """Which points 2stage-CPLS keeps, and why.

    The masks are over the raw points and do not overlap, except that
    ``stays`` marks every stay point, kept or not.  ``stay_ends`` are the
    kept stay vertices (each stay's first and last point, moved to its
    centroid); ``speed`` the speed breakpoints (the first and last points
    included); ``dp`` the points Douglas--Peucker added in the pieces between
    those; ``topology`` the points stage 2 retained on top of them, with
    ``rules`` as in :class:`.CplsSelection`.  ``x, y`` are the vertex
    coordinates: the raw positions, except that a stay point is at its
    stay's centroid.
    """

    stays: np.ndarray
    stay_ends: np.ndarray
    speed: np.ndarray
    dp: np.ndarray
    topology: np.ndarray
    rules: dict = field(default_factory=dict)
    x: np.ndarray = None
    y: np.ndarray = None

    @property
    def mask(self):
        return self.stay_ends | self.speed | self.dp | self.topology

    def summary(self):
        """Point counts, for reading at a glance."""
        n = int(self.mask.size)
        kept = int(self.mask.sum())
        runs = len(_runs(self.stays)[0])
        lines = [
            f"{n} points -> {kept} kept "
            f"({100.0 * (1 - kept / n) if n else 0.0:.2f}% compression)",
            f"  stays                {runs} ({int(self.stays.sum())} points)",
            f"  stay ends            {int(self.stay_ends.sum())}",
            f"  + speed              {int(self.speed.sum())}",
            f"  + Douglas-Peucker    {int(self.dp.sum())}",
            f"  + topology           {int(self.topology.sum())}",
        ]
        for name, flagged in self.rules.items():
            lines.append(f"      {name:<18} {int(flagged.sum())}")
        return "\n".join(lines)


def _speeds(x, y, t):
    """Central-difference speed of every point (A3, B4)."""
    m = x.size
    lo = np.maximum(np.arange(m) - 1, 0)
    hi = np.minimum(np.arange(m) + 1, m - 1)
    span = t[hi] - t[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.hypot(x[hi] - x[lo], y[hi] - y[lo]) / span
    defined = np.flatnonzero(span > 0)
    if defined.size == 0:
        return np.zeros(m)
    v = v[defined][_nearest(defined, m)]
    return v


def _nearest(defined, m):
    """For each position, the slot in ``defined`` of the nearest one (ties: earlier)."""
    pos = np.arange(m)
    right = np.clip(np.searchsorted(defined, pos), 0, defined.size - 1)
    left = np.clip(right - 1, 0, defined.size - 1)
    use_right = (defined[right] - pos) < (pos - defined[left])
    return np.where(use_right, right, left)


def _speed_dp(v, t, epsilon):
    """Douglas--Peucker on a series with the error ``|v_i - v'_i|`` (A3)."""
    m = v.size
    keep = np.zeros(m, dtype=bool)
    keep[[0, m - 1]] = True
    pending = [(0, m - 1)]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        i = np.arange(first + 1, last)
        if t[last] > t[first]:
            frac = (t[i] - t[first]) / (t[last] - t[first])
        else:
            frac = (i - first) / (last - first)
        errors = np.abs(v[i] - (v[first] + frac * (v[last] - v[first])))
        worst = int(np.argmax(errors))
        if errors[worst] > epsilon:
            split = first + 1 + worst
            keep[split] = True
            pending.append((split, last))
            pending.append((first, split))
    return keep


def two_stage_cpls_select(
    x, y, t, epsilon, speed_epsilon, barriers, *,
    stays=None, stay_radius=None, stay_duration=None, status=None,
    stay_status=(1, 5), speed_window=30, **rules,
):
    """Run 2stage-CPLS on arrays and report which points are kept, and why.

    Parameters
    ----------
    x, y : array-like
        Positions in trajectory order, in projected units.
    t : array-like
        Times, non-decreasing, in one unit (seconds for ``speed_epsilon`` in
        metres per second).
    epsilon : float
        The Douglas--Peucker threshold, in the positions' units.
    speed_epsilon : float
        The threshold of the speed stage, in position units per time unit;
        ``inf`` switches the stage off.
    barriers : Barriers or anything :class:`Barriers` accepts
    stays : array-like of bool, optional
        A ready mask of stay points; overrides everything below.
    stay_radius, stay_duration, status, stay_status
        As in :func:`stay_points`.
    speed_window : float or None
        Width, in the unit of ``t``, of the median that filters the speed
        series in time before line 11 (decision P2): ``30`` by default,
        ``None`` for the unfiltered series.  The positions are not filtered
        here -- that is :func:`two_stage_cpls_compress`'s step.
    **rules
        ``topology``, ``direction``, ``avoid_self_intersection``,
        ``keep_self_intersection``, ``inversion``, as in :func:`.cpls_select`.

    Returns
    -------
    TwoStageSelection
        ``selection.mask`` marks the retained raw points.
    """
    unknown = set(rules) - _RULE_NAMES
    if unknown:
        raise TypeError(f"unknown relation(s): {sorted(unknown)}")
    flags = {name: True for name in _RULE_NAMES}
    flags.update(rules)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = x.size
    if not (y.size == t.size == n):
        raise ValueError(
            f"x, y and t must have equal length; got {n}, {y.size}, {t.size}"
        )
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite, non-negative distance")
    if np.isnan(speed_epsilon) or speed_epsilon < 0:
        raise ValueError("speed_epsilon must be non-negative (inf switches it off)")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(t).all()):
        raise ValueError(
            "positions and times must be finite; drop missing points first"
        )
    if (np.diff(t) < 0).any():
        raise ValueError("times must not decrease along the trajectory")
    barriers = _as_barriers(barriers)

    if stays is None:
        stays = stay_points(
            x, y, t, stay_radius=stay_radius, stay_duration=stay_duration,
            status=status, stay_status=stay_status,
        )
    else:
        stays = np.asarray(stays, dtype=bool).ravel()
        if stays.size != n:
            raise ValueError("stays must have one value per point")
    starts, ends = _runs(stays)

    # The collapsed trajectory (A2): a stay is its first and last point, both
    # moved to the centroid.  Everything below happens on it (A4).
    vx, vy = x.copy(), y.copy()
    stay_end = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        vx[s:e + 1], vy[s:e + 1] = x[s:e + 1].mean(), y[s:e + 1].mean()
        stay_end[[s, e]] = True
    interior = stays & ~stay_end
    index = np.flatnonzero(~interior)
    cx, cy, ct = vx[index], vy[index], t[index]
    m = index.size
    c_end = stay_end[index]

    c_speed = np.ones(m, dtype=bool)
    if m > 2:
        speed = _speeds(cx, cy, ct)
        if speed_window is not None:
            speed = rolling_filter(ct, speed, speed_window)
        c_speed = _speed_dp(speed, ct, speed_epsilon)
    forced = c_speed | c_end
    c_dp = np.zeros(m, dtype=bool)
    kept = np.flatnonzero(forced)
    for first, last in zip(kept[:-1], kept[1:]):
        if last - first >= 2:
            piece = douglas_peucker(cx[first:last + 1], cy[first:last + 1], epsilon)
            c_dp[first:last + 1] |= piece
    stage1 = forced | c_dp
    keep = stage1.copy()
    c_rules = {name: np.zeros(m, dtype=bool) for name in RULES}
    if m > 2:
        _Stage2(cx, cy, barriers, flags).run(stage1, keep, c_rules)

    def raw(c_mask):
        out = np.zeros(n, dtype=bool)
        out[index[c_mask]] = True
        return out

    return TwoStageSelection(
        stays=stays,
        stay_ends=raw(c_end),
        speed=raw(c_speed & ~c_end),
        dp=raw(c_dp & ~forced),
        topology=raw(keep & ~stage1),
        rules={name: raw(f) for name, f in c_rules.items()},
        x=vx,
        y=vy,
    )


def _select_frame(df, epsilon, speed_epsilon, barriers, time_column,
                  status_column, options, rules):
    _check_crs(df, barriers)
    t = df[time_column].to_numpy(dtype=float) if time_column else times_of(df)
    status = df[status_column].to_numpy() if status_column else None
    geom = df.geometry
    return two_stage_cpls_select(
        geom.x.to_numpy(), geom.y.to_numpy(), t, epsilon, speed_epsilon,
        barriers, status=status, **options, **rules,
    )


def _result_frame(df, sel, stay_column):
    out = df.iloc[sel.mask].copy()
    moved = sel.stay_ends[sel.mask]
    if moved.any():
        import geopandas as gpd

        geom = out.geometry.copy()
        keep = np.flatnonzero(sel.mask)[moved]
        geom.iloc[np.flatnonzero(moved)] = gpd.points_from_xy(sel.x[keep], sel.y[keep])
        out[out.geometry.name] = geom
    if stay_column is not None:
        out[stay_column] = moved
    return out


def two_stage_cpls_compress(
    traj, epsilon, speed_epsilon, barriers, *, time_column=None, by=None,
    status_column=None, stay_status=(1, 5), stay_radius=None,
    stay_duration=None, stay_column="stay", filter_window=30, n_jobs=1,
    **rules,
):
    """Compress trajectories with 2stage-CPLS.

    The result is a row subset of the input -- every column, index label and
    time of a retained point survives -- except that the geometry of a stay's
    two kept points is replaced by the stay's centroid.

    Parameters
    ----------
    traj : movingpandas.Trajectory, TrajectoryCollection or GeoDataFrame
        Positions in a projected CRS, in trajectory order.
    epsilon : float
        Douglas--Peucker threshold, in the CRS's units (metres).
    speed_epsilon : float
        Speed threshold, in metres per second for UNIX-second times.
    barriers : Barriers or anything :class:`Barriers` accepts
    time_column : str, optional
        Numeric time column (UNIX seconds); the index by default, as in
        :func:`.compress`.
    by : str, optional
        For a GeoDataFrame holding several trajectories: the column naming
        each one.
    status_column : str, optional
        AIS navigational status; its ``stay_status`` codes make stay points.
    stay_status, stay_radius, stay_duration
        See :func:`stay_points`.
    stay_column : str or None
        Name of the boolean column that is True on the two kept points of
        every stay; ``None`` omits it.
    filter_window : float or None
        Width, in seconds, of the time-windowed median applied first:
        ``30`` by default, ``None`` for no filtering.  The positions are
        denoised, the speed series is computed from the filtered positions
        and filtered again with the same window (P2 of the module docstring).
        The result's geometry is the filtered one.
    n_jobs : int or None
        Processes over which the trajectories are shared out: ``1`` (the
        default) for none, ``-1`` for every logical CPU.  Each trajectory is
        still compressed on its own, so the result is the same.  See
        :func:`catraj.parallel_map`.
    **rules
        The five CPLS switches, as in :func:`.cpls_compress`.

    Returns
    -------
    The same kind of object, holding only the retained points.
    """
    unknown = set(rules) - _RULE_NAMES
    if unknown:
        raise TypeError(f"unknown relation(s): {sorted(unknown)}")
    barriers = _as_barriers(barriers)
    options = dict(stay_status=stay_status, stay_radius=stay_radius,
                   stay_duration=stay_duration)

    def one(df):
        if filter_window is not None:
            df = denoise(df, filter_window, time_column=time_column)
        sel = _select_frame(df, epsilon, speed_epsilon, barriers, time_column,
                            status_column,
                            dict(options, speed_window=filter_window), rules)
        return _result_frame(df, sel, stay_column)

    def again(tr):
        return two_stage_cpls_compress(
            tr, epsilon, speed_epsilon, barriers, time_column=time_column,
            status_column=status_column, stay_column=stay_column,
            filter_window=filter_window, **options, **rules,
        )

    if hasattr(traj, "trajectories"):  # TrajectoryCollection
        import movingpandas as mpd

        return mpd.TrajectoryCollection(parallel_map(
            again, traj.trajectories, n_jobs, [len(tr.df) for tr in traj.trajectories]))

    if hasattr(traj, "df") and hasattr(traj, "id"):  # Trajectory
        import movingpandas as mpd

        return mpd.Trajectory(one(traj.df), traj.id)

    if by is not None:
        parts = map_groups(one, traj, by, n_jobs)
        return pd.concat(parts) if parts else traj.iloc[:0]

    return one(traj)
