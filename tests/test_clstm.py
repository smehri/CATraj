import subprocess
import sys
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from catraj.compression import two_stage_cpls_compress
from catraj.prediction.clstm import (
    ClstmPredictor,
    PredictionSamples,
    make_samples,
    prediction_errors,
    split_by_trajectory,
)
from catraj.prediction.curvilinear import KNOT, curvilinear_predict



# ------------------------------------------------------------- fixtures


def _frame(t, x, y, sog, cog, vessel_type=None, ident=None, crs=32630):
    t = np.asarray(t, dtype=float)
    data = {"t": t, "sog": sog, "cog": cog}
    if vessel_type is not None:
        data["type"] = vessel_type
    if ident is not None:
        data["id"] = ident
    return gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(x, y), crs=crs
    )


def _straight(n=7, kind="Cargo", ident="a", t0=0.0, y=0.0):
    t = t0 + 10.0 * np.arange(n)
    return _frame(t, 10.0 * np.arange(n), np.full(n, y), 1 / KNOT, 90.0,
                  vessel_type=kind, ident=ident)


def _make(df, **kw):
    kw.setdefault("speed_column", "sog")
    kw.setdefault("course_column", "cog")
    kw.setdefault("time_column", "t")
    return make_samples(df, **kw)


# --------------------------------------------------------------- samples


def test_make_samples_hand_worked():
    df = _straight()
    s = _make(df, context_column="type", by="id")
    assert isinstance(s, PredictionSamples)
    assert len(s) == 4  # origin 4 would need t = 70
    assert s.inputs.shape == (4, 3, 3) and s.targets.shape == (4, 3, 2)
    assert s.inputs.dtype == s.targets.dtype == np.float64
    np.testing.assert_array_equal(s.dt, 10.0)
    np.testing.assert_array_equal(s.time, [0, 10, 20, 30])
    np.testing.assert_array_equal(s.origin, [[0, 0], [10, 0], [20, 0], [30, 0]])
    for k in range(4):
        np.testing.assert_allclose(s.inputs[k, :, 0], 10.0 * k + np.array([10, 20, 30]),
                                   atol=1e-9)
        np.testing.assert_allclose(s.inputs[k, :, 1], 0, atol=1e-9)
        np.testing.assert_array_equal(s.inputs[k, :, 2], [10, 20, 30])
    np.testing.assert_allclose(s.targets[..., 0], s.inputs[..., 0], atol=1e-9)
    np.testing.assert_allclose(s.targets[..., 1], s.inputs[..., 1], atol=1e-9)
    np.testing.assert_allclose(s.curvilinear, s.inputs[..., :2])
    assert list(s.group) == ["Cargo"] * 4 and list(s.trajectory) == ["a"] * 4

    # Bend the track: y jumps by 5 m at t >= 30.  The inputs still follow SOG
    # and COG from each origin (origin 3 now sits at y = 5); the targets bend.
    bent = df.copy()
    bent["geometry"] = gpd.points_from_xy(
        10.0 * np.arange(7), [0, 0, 0, 5, 5, 5, 5]
    )
    b = _make(bent)
    np.testing.assert_allclose(b.inputs[..., 0], s.inputs[..., 0], atol=1e-9)
    np.testing.assert_allclose(b.inputs[..., 2], s.inputs[..., 2], atol=1e-9)
    np.testing.assert_allclose(
        b.inputs[..., 1], [[0] * 3, [0] * 3, [0] * 3, [5] * 3], atol=1e-9
    )
    np.testing.assert_allclose(b.targets[..., 0], s.targets[..., 0], atol=1e-9)
    np.testing.assert_allclose(
        b.targets[..., 1], [[0, 0, 5], [0, 5, 5], [5, 5, 5], [5, 5, 5]], atol=1e-9
    )


def test_targets_are_interpolated_in_time():
    # Eq. 12: dt = 40 / 3; the epochs 13.33, 26.67 and 40 fall between vertices.
    df = _frame([0, 10, 30, 40], [0, 10, 30, 40], [0, 10, 0, 10], 1.0, 90.0)
    s = _make(df, speed_unit="native")
    assert len(s) == 1
    np.testing.assert_allclose(s.dt, 40 / 3)
    dt = 40 / 3
    np.testing.assert_allclose(s.targets[0, :, 0], [dt, 2 * dt, 40], atol=1e-9)
    np.testing.assert_allclose(
        s.targets[0, :, 1], [10 - 10 * (dt - 10) / 20, 10 * (30 - 2 * dt) / 20, 10],
        atol=1e-9,
    )
    np.testing.assert_allclose(s.inputs[0, :, 2], [dt, 2 * dt, 3 * dt])


def test_speed_unit_and_steps_and_duplicate_times():
    df = _straight()
    native = _make(df, speed_unit="native")
    np.testing.assert_allclose(native.inputs[0, :, 0], np.array([1, 2, 3]) * 10 / KNOT)
    s5 = _make(df, steps=2)
    assert s5.inputs.shape == (5, 2, 3) and s5.targets.shape == (5, 2, 2)
    with pytest.raises(ValueError, match="speed_unit"):
        _make(df, speed_unit="mph")

    # A repeated timestamp keeps its first row.
    dup = pd.concat([df.iloc[:3], df.iloc[2:3].assign(sog=99.0), df.iloc[3:]])
    dup = dup.reset_index(drop=True)
    a, b = _make(dup), _make(df)
    np.testing.assert_allclose(a.inputs, b.inputs)
    np.testing.assert_allclose(a.targets, b.targets)


def test_unavailable_speed_or_course_drops_the_sample():
    df = _straight()
    df["sog"] = [102.3, 1 / KNOT, np.nan, 1 / KNOT, 1 / KNOT, 1 / KNOT, 1 / KNOT]
    s = _make(df)
    np.testing.assert_array_equal(s.time, [10, 30])

    df = _straight()
    df["cog"] = [90.0, 360.0, 90.0, np.nan, 90.0, 90.0, 90.0]
    s = _make(df)
    np.testing.assert_array_equal(s.time, [0, 20])

    # The rows stay usable as targets: dt is the trajectory's.
    np.testing.assert_array_equal(s.dt, 10.0)
    np.testing.assert_allclose(s.targets[1, :, 0], [30, 40, 50], atol=1e-9)


def test_short_or_degenerate_trajectories_give_no_samples():
    one = _straight(n=1)
    flat = _frame([5, 5, 5], [0, 1, 2], [0, 0, 0], 1.0, 90.0)
    short = _straight(n=3)  # origin 0 would need t = 30 > 20
    for df in (one, flat, short):
        s = _make(df)
        assert len(s) == 0
        assert s.inputs.shape == (0, 3, 3) and s.targets.shape == (0, 3, 2)


def test_null_type_becomes_other():
    a = _straight(kind=None, ident="a")
    b = _straight(kind="", ident="b")
    c = _straight(kind=pd.Categorical([None] * 7, categories=["Tug"]), ident="c")
    d = _straight(kind="Tug", ident="d")
    s = _make(pd.concat([a, b, c, d]), context_column="type", by="id")
    assert list(s.group) == ["Other"] * 12 + ["Tug"] * 4

    # Categorical with values becomes str.
    e = _straight(kind=pd.Categorical(["Tug"] * 7), ident="e")
    s = _make(e, context_column="type")
    assert list(s.group) == ["Tug"] * 4 and s.group.dtype == object


def test_types_follow_the_trajectory():
    a = _straight(kind="Cargo", ident="a")
    b = _straight(kind="Tug", ident="b", t0=1000.0, y=50.0)
    s = _make(pd.concat([a, b]), context_column="type", by="id")
    assert list(s.group) == ["Cargo"] * 4 + ["Tug"] * 4
    assert list(s.trajectory) == ["a"] * 4 + ["b"] * 4

    # A type that is not constant: the most frequent non-null value.
    mixed = _straight(ident="m")
    mixed["type"] = ["Tug", None, "Cargo", "Cargo", "Tug", "Cargo", None]
    s = _make(mixed, context_column="type")
    assert set(s.group) == {"Cargo"}

    # No context column: one group.
    assert set(_make(a).group) == {"all"}


def _two_frames():
    a = _straight(kind="Cargo", ident="a")
    b = _straight(n=9, kind="Tug", ident="b", t0=1000.0, y=50.0)
    b["cog"] = 45.0
    return a, b


def _assert_same(p, q):
    for name in ("inputs", "targets", "origin", "time", "dt"):
        np.testing.assert_allclose(getattr(p, name), getattr(q, name), atol=1e-9)
    assert list(p.group) == list(q.group)
    assert list(p.trajectory) == list(q.trajectory)


def test_make_samples_accepts_every_form():
    a, b = _two_frames()
    both = pd.concat([a, b])
    ref = _make(both, context_column="type", by="id")
    assert len(ref) == 4 + 6 and list(ref.trajectory) == ["a"] * 4 + ["b"] * 6

    # Time from a DatetimeIndex instead of a column.
    dated = both.set_index(pd.to_datetime(both["t"], unit="s")).drop(columns="t")
    _assert_same(
        make_samples(dated, speed_column="sog", course_column="cog",
                     context_column="type", by="id"),
        ref,
    )
    # The same, with the index and no ``by``: one trajectory per call.
    only_a = make_samples(
        dated[dated["id"] == "a"], speed_column="sog", course_column="cog",
        context_column="type",
    )
    assert len(only_a) == 4 and list(only_a.group) == ["Cargo"] * 4


@pytest.mark.movingpandas
def test_make_samples_accepts_movingpandas_objects():
    mpd = pytest.importorskip("movingpandas")
    a, b = _two_frames()
    both = pd.concat([a, b])
    ref = _make(both, context_column="type", by="id")

    def traj(df, ident):
        df = df.set_index(pd.to_datetime(df["t"], unit="s")).drop(columns="t")
        return mpd.Trajectory(df, ident)

    ta, tb = traj(a, "a"), traj(b, "b")
    kw = dict(speed_column="sog", course_column="cog", context_column="type")
    _assert_same(make_samples(mpd.TrajectoryCollection([ta, tb]), **kw), ref)

    single = make_samples(ta, **kw)
    assert list(single.trajectory) == ["a"] * 4
    _assert_same(single, ref.subset(np.arange(4)))
    # time_column works on a Trajectory too.
    tc = mpd.Trajectory(
        a.set_index(pd.to_datetime(a["t"], unit="s")), "a"
    )
    _assert_same(make_samples(tc, time_column="t", **kw), single)


def test_make_samples_refuses_geographic_coordinates():
    df = _straight().set_crs(32630, allow_override=True).to_crs(4326)
    with pytest.raises(ValueError, match="geographic"):
        _make(df)


def test_make_samples_names_a_missing_column():
    df = _straight()
    with pytest.raises(KeyError, match="sog"):
        _make(df.drop(columns="sog"))
    with pytest.raises(KeyError, match="cog"):
        _make(df.drop(columns="cog"))
    with pytest.raises(KeyError, match="kind"):
        _make(df, context_column="kind")
    with pytest.raises(KeyError, match="trip"):
        _make(df, by="trip")
    with pytest.raises(KeyError, match="when"):
        _make(df, time_column="when")


def test_decreasing_times_are_refused():
    df = _frame([0, 10, 5, 20], [0, 1, 2, 3], [0, 0, 0, 0], 1.0, 90.0)
    with pytest.raises(ValueError, match="decrease"):
        _make(df)


# ---------------------------------------------------------------- split


def _fleet(n=10, per=7):
    return PredictionSamples.concat([
        _make(_straight(n=per + 3, ident=f"v{i}", t0=1000.0 * i,
                        kind="Cargo" if i % 2 else "Tug"),
              context_column="type", by="id")
        for i in range(n)
    ])


def test_concat_and_subset():
    a, b = _two_frames()
    sa = _make(a, context_column="type")
    sb = _make(b, context_column="type")
    both = PredictionSamples.concat([sa, sb])
    assert len(both) == len(sa) + len(sb)
    np.testing.assert_array_equal(both.subset(np.arange(len(sa))).inputs, sa.inputs)
    mask = np.zeros(len(both), dtype=bool)
    mask[-2:] = True
    np.testing.assert_array_equal(both.subset(mask).targets, sb.targets[-2:])
    assert len(both.subset([])) == 0
    with pytest.raises(ValueError):
        PredictionSamples.concat([])


def test_split_keeps_trajectories_whole():
    samples = _fleet(n=10)
    train, test = split_by_trajectory(samples, test_size=0.2, seed=3)
    assert set(train.trajectory).isdisjoint(set(test.trajectory))
    assert len(train) + len(test) == len(samples)
    assert len(set(test.trajectory)) == 2 and len(set(train.trajectory)) == 8
    for ident in set(samples.trajectory):
        side = train if ident in set(train.trajectory) else test
        assert (side.trajectory == ident).sum() == (samples.trajectory == ident).sum()

    again = split_by_trajectory(samples, test_size=0.2, seed=3)
    assert list(again[1].trajectory) == list(test.trajectory)
    other = {s for s in range(20)
             if set(split_by_trajectory(samples, 0.2, s)[1].trajectory)
             != set(test.trajectory)}
    assert other  # the seed matters


def test_split_matches_the_rule_in_the_brief():
    samples = _fleet(n=10)
    ids = list(dict.fromkeys(samples.trajectory))
    rng = np.random.default_rng(7)
    order = list(ids)
    rng.shuffle(order)
    _, test = split_by_trajectory(samples, test_size=0.3, seed=7)
    assert set(test.trajectory) == set(order[:3])


def test_split_clamps_and_refuses():
    samples = _fleet(n=3)
    _, test = split_by_trajectory(samples, test_size=0.0)
    assert len(set(test.trajectory)) == 1
    train, _ = split_by_trajectory(samples, test_size=0.99)
    assert len(set(train.trajectory)) == 1
    with pytest.raises(ValueError):
        split_by_trajectory(_fleet(n=1))


# --------------------------------------------------------------- errors


def _scoring_samples():
    targets = np.zeros((4, 3, 2))
    return PredictionSamples(
        inputs=np.zeros((4, 3, 3)), targets=targets, origin=np.zeros((4, 2)),
        time=np.arange(4.0), dt=np.ones(4),
        group=np.array(["A", "A", "B", "B"], dtype=object),
        trajectory=np.array(["a", "a", "b", "b"], dtype=object),
    )


def test_prediction_errors_matches_eq_25():
    s = _scoring_samples()
    pred = np.zeros((4, 3, 2))
    pred[0, :, 0] = [3, 0, 0]          # distance 3 at t+1
    pred[0, :, 1] = [4, 0, 0]          # ... now 5
    pred[1, 1] = [6, 8]                # distance 10 at t+2
    pred[2, 2] = [0, 2]                # distance 2 at t+3
    pred[3] = np.nan                   # missing
    table = prediction_errors(pred, s)
    assert list(table.index) == ["A", "B", "all"]
    assert list(table.columns) == [
        "samples", "missing", "rmse", "rmse_t+1", "rmse_t+2", "rmse_t+3",
        "mean_error", "median_error",
    ]
    a = table.loc["A"]
    assert a["samples"] == 2 and a["missing"] == 0
    d = np.array([5, 0, 0, 0, 10, 0.0])
    assert a["rmse"] == pytest.approx(np.sqrt((d ** 2).sum() / 6))
    assert a["rmse_t+1"] == pytest.approx(np.sqrt(25 / 2))
    assert a["rmse_t+2"] == pytest.approx(np.sqrt(100 / 2))
    assert a["rmse_t+3"] == 0
    assert a["mean_error"] == pytest.approx(15 / 6)
    assert a["median_error"] == 0
    b = table.loc["B"]
    assert b["samples"] == 1 and b["missing"] == 1
    assert b["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert b["rmse_t+3"] == pytest.approx(2.0)
    allrow = table.loc["all"]
    assert allrow["samples"] == 3 and allrow["missing"] == 1
    d_all = np.concatenate([d, [0, 0, 2.0]])
    assert allrow["rmse"] == pytest.approx(np.sqrt((d_all ** 2).sum() / 9))
    assert allrow["mean_error"] == pytest.approx(d_all.mean())
    assert allrow["median_error"] == pytest.approx(np.median(d_all))


def test_prediction_errors_scores_the_physical_model_alone():
    s = _make(_straight(), context_column="type")
    table = prediction_errors(s.curvilinear, s)
    assert table.loc["all", "samples"] == 4
    assert table.loc["all", "rmse"] == pytest.approx(0, abs=1e-9)

    # With a drift of 5 m east at every step the error is exactly 5 m.
    px, py = curvilinear_predict(s.origin[:, 0], s.origin[:, 1],
                                 np.ones(4), np.full(4, 90.0), s.dt)
    off = np.stack([px + 5, py], axis=-1)
    t = prediction_errors(off, s)
    assert t.loc["all", "mean_error"] == pytest.approx(5)
    assert t.loc["all", "rmse"] == pytest.approx(5)


def test_prediction_errors_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        prediction_errors(np.zeros((3, 3, 2)), _scoring_samples())


# ------------------------------------------------------------- the networks
#
# The synthetic fleet of section IV-E's argument: three types whose actual
# motion departs from dead reckoning in their own way.

_DRIFT = {"A": (100.0, 0.0), "B": (0.0, 100.0), "C": (0.0, 0.0)}


@lru_cache(maxsize=1)
def _synthetic_fleet():
    """60 trajectories, 20 per type.  Rows carry the SOG and COG of dead
    reckoning; the actual track adds the type's drift at every step."""
    rng = np.random.default_rng(11)
    frames = []
    for i in range(60):
        kind = "ABC"[i % 3]
        n, dt = 40, 5.0
        speed = rng.uniform(3.0, 8.0)  # m/s
        course = rng.uniform(0.0, 360.0)
        t = dt * np.arange(n)
        step = speed * dt
        sx, sy = np.sin(np.radians(course)), np.cos(np.radians(course))
        k = np.arange(n)
        x = 4e5 + rng.uniform(-1e3, 1e3) + k * (step * sx + _DRIFT[kind][0])
        y = 5.3e6 + rng.uniform(-1e3, 1e3) + k * (step * sy + _DRIFT[kind][1])
        frames.append(_frame(t, x, y, speed / KNOT, course,
                             vessel_type=kind, ident=f"v{i}"))
    samples = _make(pd.concat(frames), context_column="type", by="id")
    return split_by_trajectory(samples, test_size=0.2, seed=0)


def _predictor(**kw):
    kw.setdefault("device", "cpu")
    return ClstmPredictor(**kw)


def _tiny_fit(**kw):
    train, _ = _synthetic_fleet()
    kw.setdefault("max_epochs", 3)
    return _predictor(**kw).fit(train)


@pytest.mark.torch
def test_clstm_learns_each_types_deviation():
    pytest.importorskip("torch")
    # A batch of 32 instead of the default 256: this tiny fleet has about 600
    # samples a group, so 256 gives 2-3 optimiser steps per epoch and the
    # networks stop far short of convergence.  Real data has millions of
    # samples.  Passes both margins on the 10 fleet seeds 11-20 measured.
    train, test = _synthetic_fleet()
    clstm = _predictor(max_epochs=300, batch_size=32).fit(train)
    lstm = _predictor(context=False, max_epochs=300, batch_size=32).fit(train)
    assert clstm.groups_ == ["A", "B", "C"] and clstm.dropped_groups_ == []
    assert lstm.groups_ == ["all"]
    errors = {
        "curvilinear": prediction_errors(test.curvilinear, test),
        "lstm": prediction_errors(lstm.predict(test), test),
        "clstm": prediction_errors(clstm.predict(test), test),
    }
    mean = {k: v.loc["all", "mean_error"] for k, v in errors.items()}
    print("mean point-wise error", mean)
    print("r1 = clstm/curvilinear %.4f (< 0.3), r2 = clstm/lstm %.4f (< 0.5)"
          % (mean["clstm"] / mean["curvilinear"], mean["clstm"] / mean["lstm"]))
    assert mean["clstm"] < 0.3 * mean["curvilinear"]
    assert mean["clstm"] < 0.5 * mean["lstm"]


@pytest.mark.torch
def test_target_defaults_to_the_deviation_and_is_validated():
    pytest.importorskip("torch")
    assert _predictor().target == "deviation"
    assert _predictor(target="absolute").target == "absolute"
    with pytest.raises(ValueError, match="target"):
        _predictor(target="relative")


@pytest.mark.torch
def test_deviation_on_dead_reckoning_data_predicts_the_curvilinear_positions():
    pytest.importorskip("torch")
    # The true motion IS dead reckoning here: the deviation is zero, so the
    # prediction must be T' (within a metre), whatever the training.
    rng = np.random.default_rng(5)
    frames = []
    for i in range(20):
        n, dt = 40, 5.0
        speed, course = rng.uniform(3.0, 8.0), rng.uniform(0.0, 360.0)
        k = np.arange(n)
        x = 4e5 + rng.uniform(-1e3, 1e3) + k * dt * speed * np.sin(np.radians(course))
        y = 5.3e6 + rng.uniform(-1e3, 1e3) + k * dt * speed * np.cos(np.radians(course))
        frames.append(_frame(dt * k, x, y, speed / KNOT, course, vessel_type="A", ident=f"v{i}"))
    samples = _make(pd.concat(frames), context_column="type", by="id")
    train, test = split_by_trajectory(samples, test_size=0.25, seed=0)
    fitted = _predictor(max_epochs=20, batch_size=32).fit(train)
    pred = fitted.predict(test)
    assert np.abs(pred - test.curvilinear).max() < 1.0
    absolute = _predictor(max_epochs=20, batch_size=32, target="absolute").fit(train)
    # the absolute network cannot match that precision on coordinates of ~5e6 m
    assert np.abs(absolute.predict(test) - test.curvilinear).max() > 1.0


@pytest.mark.torch
def test_absolute_target_still_works_and_differs_from_the_deviation():
    pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    a = _tiny_fit(target="absolute")
    d = _tiny_fit()
    pa, pd_ = a.predict(test), d.predict(test)
    assert pa.shape == pd_.shape == test.targets.shape and np.isfinite(pa).all()
    assert not np.array_equal(pa, pd_)
    assert "absolute" in a.summary() and "deviation" in d.summary()


@pytest.mark.torch
def test_fit_is_deterministic_on_cpu():
    pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    a, b = _tiny_fit(), _tiny_fit()
    np.testing.assert_array_equal(a.predict(test), b.predict(test))
    c = _tiny_fit(seed=1)
    assert not np.array_equal(a.predict(test), c.predict(test))
    assert a.history_ == b.history_


@pytest.mark.torch
def test_save_and_load_give_the_same_predictions(tmp_path):
    pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    fitted = _tiny_fit(min_samples=300)
    path = tmp_path / "clstm.pt"
    fitted.save(path)
    loaded = ClstmPredictor.load(path, device="cpu")
    np.testing.assert_array_equal(fitted.predict(test), loaded.predict(test))
    assert loaded.groups_ == fitted.groups_
    assert loaded.dropped_groups_ == fitted.dropped_groups_
    assert loaded.history_ == fitted.history_
    assert loaded.summary() == fitted.summary()
    assert loaded.hidden_size == 64 and loaded.min_samples == 300
    assert loaded.target == fitted.target == "deviation"


@pytest.mark.torch
def test_save_and_load_keep_the_absolute_target_and_read_old_files(tmp_path):
    torch = pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    fitted = _tiny_fit(target="absolute", min_samples=300)
    path = tmp_path / "abs.pt"
    fitted.save(path)
    loaded = ClstmPredictor.load(path, device="cpu")
    assert loaded.target == "absolute"
    np.testing.assert_array_equal(fitted.predict(test), loaded.predict(test))
    # a file written before the option existed has no "target": absolute
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["config"]["target"]
    old = tmp_path / "old.pt"
    torch.save(payload, old)
    assert ClstmPredictor.load(old, device="cpu").target == "absolute"


@pytest.mark.torch
def test_small_groups_get_no_network():
    pytest.importorskip("torch")
    train, test = _synthetic_fleet()
    counts = {g: int((train.group == g).sum()) for g in "ABC"}
    limit = max(counts.values()) + 1  # every group is dropped but none trains
    assert _predictor(max_epochs=1, min_samples=limit).fit(train).groups_ == []

    limit = sorted(counts.values())[1]  # drops the smallest group only
    fitted = _predictor(max_epochs=2, min_samples=limit).fit(train)
    dropped = sorted(g for g, c in counts.items() if c < limit)
    assert dropped
    assert fitted.dropped_groups_ == dropped
    assert fitted.groups_ == sorted(set("ABC") - set(dropped))
    pred = fitted.predict(test)
    gone = np.isin(test.group, dropped)
    assert gone.any() and np.isnan(pred[gone]).all()
    assert np.isfinite(pred[~gone]).all()
    table = prediction_errors(pred, test)
    for g in dropped:
        assert table.loc[g, "missing"] == (test.group == g).sum()
        assert table.loc[g, "samples"] == 0
    assert table.loc["all", "missing"] == gone.sum()
    assert set(fitted.history_) == set(fitted.groups_)
    for g in dropped:
        assert g in fitted.summary().splitlines()[-1]


@pytest.mark.torch
def test_one_trajectory_trains_without_validation():
    pytest.importorskip("torch")
    train, _ = _synthetic_fleet()
    one = train.subset(train.trajectory == train.trajectory[0])
    fitted = _predictor(max_epochs=4, min_samples=10).fit(one)
    (hist,) = fitted.history_.values()
    assert len(hist) == 4 and all(np.isnan(h[2]) for h in hist)
    assert "none" in fitted.summary()


@pytest.mark.torch
def test_early_stopping_restores_the_best_epoch():
    pytest.importorskip("torch")
    train, _ = _synthetic_fleet()
    fitted = _predictor(max_epochs=60, patience=2, lr=0.05).fit(train)
    stopped = 0
    for hist in fitted.history_.values():
        val = [h[2] for h in hist]
        assert [h[0] for h in hist] == list(range(1, len(hist) + 1))
        assert not np.isnan(val).any()
        if len(hist) < 60:
            stopped += 1
            assert len(hist) - 1 - int(np.argmin(val)) == 2  # patience ran out
        assert f"{min(val):.5f}" in fitted.summary()
    assert stopped


@pytest.mark.torch
def test_predict_before_fit_raises(tmp_path):
    pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    with pytest.raises(RuntimeError, match="fit"):
        _predictor().predict(test)
    with pytest.raises(RuntimeError, match="fit"):
        _predictor().save(tmp_path / "nowhere.pt")


@pytest.mark.torch
def test_predict_keeps_the_sample_shape_and_float64():
    pytest.importorskip("torch")
    _, test = _synthetic_fleet()
    fitted = _tiny_fit()
    pred = fitted.predict(test)
    assert pred.shape == test.targets.shape and pred.dtype == np.float64
    assert np.isfinite(pred).all()
    assert fitted.predict(test.subset([])).shape == (0, 3, 2)


def test_missing_torch_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"catraj\[predict\]"):
        ClstmPredictor()


def test_import_does_not_need_torch():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import catraj, catraj.prediction as p\n"
        "from catraj.prediction import ClstmPredictor\n"
        "import numpy as np\n"
        "p.curvilinear_predict(np.zeros(1), np.zeros(1), np.ones(1),"
        " np.zeros(1), 1.0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr


# -------------------------------------------------------------- sample data


@pytest.mark.data
@pytest.mark.torch
def test_clstm_runs_on_the_sample_fleet():
    pytest.importorskip("torch")
    from catraj.datasets import load_ais, load_coastline

    ais = load_ais()    # sorted by (mmsi, unix_time)
    land = load_coastline()
    small = two_stage_cpls_compress(
        ais, 1_000, 1.54, land, status_column="navigational_status",
        stay_radius=100, stay_duration=600, time_column="unix_time", by="mmsi",
    )
    samples = make_samples(
        small, speed_column="speed_over_ground",
        course_column="course_over_ground", context_column="vessel_type",
        time_column="unix_time", by="mmsi",
    )
    train, test = split_by_trajectory(samples)
    fitted = ClstmPredictor(max_epochs=3, min_samples=50, device="cpu").fit(train)
    assert fitted.groups_ and fitted.dropped_groups_
    pred = fitted.predict(test)
    assert pred.shape == test.targets.shape
    trained = np.isin(test.group, fitted.groups_)
    assert trained.any() and np.isfinite(pred[trained]).all()
    assert np.isnan(pred[~trained]).all()
    assert prediction_errors(pred, test).loc["all", "samples"] == trained.sum()
