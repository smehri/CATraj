import numpy as np
import pytest

from catraj.annotation import (
    IDW,
    RBF,
    SPATIAL_METHODS,
    TIN,
    CloughTocher,
    Kriging,
    Nearest,
    angular_residual,
    to_surface,
)

from helpers import linear_field

ALL = [Nearest, IDW, TIN, CloughTocher, RBF, Kriging]
INSIDE = (np.array([5_000.0, 17_000.0, 23_500.0]), np.array([5_000.0, 12_000.0, 28_000.0]))


@pytest.mark.parametrize("cls", ALL)
def test_constant_field_is_reproduced(cls, grid_stations):
    x, y = grid_stations
    fitted = cls().fit(x, y, np.full(x.size, 7.5))
    assert fitted(*INSIDE) == pytest.approx(7.5, abs=1e-6)


@pytest.mark.parametrize("cls", [TIN, RBF])
def test_linear_field_is_reproduced(cls, grid_stations):
    x, y = grid_stations
    fitted = cls().fit(x, y, linear_field(x, y))
    assert fitted(*INSIDE) == pytest.approx(linear_field(*INSIDE), abs=1e-6)


@pytest.mark.parametrize("cls", [IDW, TIN, CloughTocher, RBF, Kriging])
def test_exact_at_stations(cls, grid_stations):
    x, y = grid_stations
    values = linear_field(x, y)
    assert cls().fit(x, y, values)(x, y) == pytest.approx(values, abs=1e-4)


def test_idw_midpoint_of_two_stations_is_their_mean():
    fitted = IDW().fit([0.0, 10.0], [0.0, 0.0], [2.0, 6.0])
    assert fitted(5.0, 0.0) == pytest.approx(4.0)


def test_idw_with_k1_is_nearest(grid_stations):
    x, y = grid_stations
    v = np.arange(x.size, dtype=float)
    q = (np.array([1_000.0, 26_000.0]), np.array([2_000.0, 9_000.0]))
    assert IDW(k=1).fit(x, y, v)(*q) == pytest.approx(Nearest().fit(x, y, v)(*q))


@pytest.mark.parametrize("cls", [TIN, CloughTocher])
def test_triangulation_is_nan_outside_hull(cls, grid_stations):
    x, y = grid_stations
    fitted = cls().fit(x, y, linear_field(x, y))
    assert cls.extrapolates is False
    assert np.isnan(fitted(-5_000.0, -5_000.0))


@pytest.mark.parametrize("cls", [Nearest, IDW, RBF, Kriging])
def test_extrapolating_methods_answer_outside_hull(cls, grid_stations):
    x, y = grid_stations
    assert np.isfinite(cls().fit(x, y, linear_field(x, y))(-5_000.0, -5_000.0))


@pytest.mark.parametrize("cls", ALL)
def test_circular_interpolation_goes_through_north(cls):
    # Two stations at 350 and 10 degrees; halfway between them is north.
    x = np.array([0.0, 10_000.0, 0.0, 10_000.0])
    y = np.array([0.0, 0.0, 10_000.0, 10_000.0])
    v = np.array([350.0, 10.0, 350.0, 10.0])
    out = cls(circular=True).fit(x, y, v)(5_000.0, 5_000.0)
    if cls is Nearest:  # piecewise constant: lands on one of them
        assert out in (pytest.approx(350.0), pytest.approx(10.0))
    else:
        assert abs(angular_residual(out, 0.0)) < 1.0


def test_missing_stations_are_dropped(grid_stations):
    x, y = grid_stations
    v = linear_field(x, y)
    v[3] = np.nan
    fitted = IDW().fit(x, y, v)
    assert fitted.n_stations == x.size - 1
    assert np.isfinite(fitted(5_000.0, 5_000.0))


def test_rbf_on_collinear_stations_yields_nan_rather_than_raising():
    # A thin-plate spline cannot fit its linear term to stations on one line.
    x = np.array([0.0, 10_000.0, 20_000.0, 30_000.0])
    fitted = RBF().fit(x, 2 * x, [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(fitted(5_000.0, 5_000.0))


def test_leave_one_out_survives_a_degenerate_epoch():
    from catraj.annotation import evaluate_spatial

    # Epoch 0: four stations on a line (singular for RBF); epoch 1: a square.
    x = np.r_[0.0, 1e4, 2e4, 3e4, 0.0, 1e4, 0.0, 1e4]
    y = np.r_[0.0, 1e4, 2e4, 3e4, 0.0, 0.0, 1e4, 1e4]
    table = evaluate_spatial(x, y, np.arange(8.0), np.r_[0, 0, 0, 0, 1, 1, 1, 1],
                             methods=["rbf", "idw"])
    assert set(table.epochs) == {0, 1}


def test_empty_epoch_yields_nan_rather_than_raising():
    fitted = IDW().fit([np.nan], [0.0], [1.0])
    assert np.isnan(fitted(0.0, 0.0))


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        IDW()(0.0, 0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        IDW().fit([0.0, 1.0], [0.0], [1.0, 2.0])


def test_scalar_in_scalar_out_and_shape_kept(grid_stations):
    x, y = grid_stations
    fitted = IDW().fit(x, y, linear_field(x, y))
    assert isinstance(fitted(1.0, 2.0), float)
    assert fitted(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)


def test_kriging_std_is_small_at_stations_and_grows_away(grid_stations):
    x, y = grid_stations
    k = Kriging().fit(x, y, np.sin(x / 9_000.0) + np.cos(y / 7_000.0))
    at_station = k.predict_std(x[:1], y[:1])[0]
    far_away = k.predict_std([200_000.0], [200_000.0])[0]
    assert at_station < 1e-2 < far_away


def test_kriging_std_undefined_for_bearings(grid_stations):
    x, y = grid_stations
    with pytest.raises(NotImplementedError):
        Kriging(circular=True).fit(x, y, np.full(x.size, 90.0)).predict_std(0, 0)


def test_to_surface_shape_and_validation(grid_stations):
    x, y = grid_stations
    fitted = IDW().fit(x, y, linear_field(x, y))
    grid, xs, ys = to_surface(fitted, (0, 0, 30_000, 20_000), 10_000)
    assert grid.shape == (len(ys), len(xs)) == (3, 4)
    with pytest.raises(ValueError):
        to_surface(fitted, (0, 0, 1, 1), 0)
    with pytest.raises(ValueError):
        to_surface(fitted, (1, 1, 0, 0), 1)


def test_registry_names_every_method():
    assert set(SPATIAL_METHODS.values()) == set(ALL)
