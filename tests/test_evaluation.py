import numpy as np
import pytest

from catraj.annotation import (
    DLT,
    Spline,
    default_candidates,
    evaluate_spatial,
    evaluate_temporal,
    k_fold,
    leave_one_station_out,
)

from .conftest import linear_field

# --------------------------------------------------------------- spatial


def test_loso_needs_three_stations():
    metrics, unreachable, n = leave_one_station_out([0, 1], [0, 1], [1, 2], "idw")
    assert (metrics.n, unreachable, n) == (0, 0, 2)


def test_loso_counts_unreachable_hull_stations(grid_stations):
    x, y = grid_stations
    metrics, unreachable, n = leave_one_station_out(x, y, linear_field(x, y), "tin")
    assert n == 16
    assert unreachable >= 4  # at least the four corners
    assert metrics.rmse == pytest.approx(0.0, abs=1e-9)  # exact where reachable


def test_evaluate_spatial_picks_tin_on_a_plane(station_frame):
    f = station_frame
    table = evaluate_spatial(f.x, f.y, f.value, f.t, methods=["nearest", "tin"])
    assert table.methods == ["nearest", "tin"]
    assert len(table.epochs) == 6
    assert table.select() == "tin"
    per_epoch = table.select(per_epoch=True)
    assert set(per_epoch) == set(table.epochs)
    assert set(per_epoch.values()) == {"tin"}
    assert len(table.to_frame()) == 12
    assert "tin" in table.summary()


def test_evaluate_spatial_passes_method_kwargs(station_frame):
    f = station_frame
    table = evaluate_spatial(
        f.x, f.y, f.value, f.t, methods=["idw"], method_kwargs={"idw": {"power": 1}}
    )
    assert table.methods == ["idw"]


def test_evaluate_spatial_rejects_bad_input(station_frame):
    f = station_frame
    with pytest.raises(ValueError):
        evaluate_spatial(f.x, f.y, f.value, f.t, methods=["no_such_method"])
    with pytest.raises(ValueError):
        evaluate_spatial(f.x[:3], f.y, f.value, f.t)
    table = evaluate_spatial(f.x, f.y, f.value, f.t, methods=["idw"])
    with pytest.raises(ValueError):
        table.select(rule="r2")


# -------------------------------------------------------------- temporal

T = np.arange(0.0, 25.0) * 3600.0
V = 8.0 + 3.0 * np.sin(2 * np.pi * T / (24 * 3600.0))


class _Spy(DLT):
    seen = []

    def fit(self, t, values):
        _Spy.seen.append(np.asarray(t, float).copy())
        return super().fit(t, values)


@pytest.mark.parametrize("k", [2, 5, 10])
def test_kfold_never_holds_out_the_endpoints(k):
    _Spy.seen = []
    k_fold(T, V, _Spy, k)
    assert len(_Spy.seen) == k
    for train in _Spy.seen:
        assert T[0] in train and T[-1] in train


def test_kfold_too_little_data():
    metrics, n_train = k_fold(T[:3], V[:3], DLT, 2)
    assert (metrics.n, n_train) == (0, 0)
    metrics, n_train = k_fold(T[:5], V[:5], DLT, 10)  # fewer interior points than k
    assert n_train == 0


def test_kfold_is_reproducible_with_a_seed():
    a, _ = k_fold(T, V, Spline, 4, seed=7)
    b, _ = k_fold(T, V, Spline, 4, seed=7)
    assert a == b


def test_evaluate_temporal_prefers_spline_on_a_smooth_series():
    table = evaluate_temporal(T, V)
    names = [name for name, _, _ in default_candidates()]
    assert table.methods == names
    assert table.ks == list(range(2, 11))
    assert table.best_row("rmse").method in {"spline"} | {n for n in names if n.startswith("pp_")}
    best_dlt = min(r.metrics.rmse for r in table.results if r.method == "dlt")
    best_spline = min(r.metrics.rmse for r in table.results if r.method == "spline")
    assert best_spline < best_dlt


def test_select_returns_a_model_fitted_to_every_observation():
    table = evaluate_temporal(T, V, ks=[2, 3])
    model = table.select(method="spline")
    assert isinstance(model, Spline)
    assert model(T) == pytest.approx(V)
    assert isinstance(table.select(method="dlt"), DLT)
    with pytest.raises(ValueError):
        table.select(method="nope")
    with pytest.raises(ValueError):
        table.select(rule="r2")


def test_temporal_frame_and_summary():
    table = evaluate_temporal(T, V, ks=[2, 3])
    frame = table.to_frame()
    assert {"method", "k", "n_train", "rmse", "mae"} <= set(frame.columns)
    assert len(frame) == len(default_candidates()) * 2
    assert "best k=" in table.summary()
