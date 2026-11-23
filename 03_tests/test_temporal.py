import numpy as np
import pytest

from catraj.annotation import DLT, PiecewisePolynomial, Spline, angular_residual

T = np.arange(0.0, 11.0)


def test_dlt_is_linear_between_epochs_and_holds_beyond():
    f = DLT().fit([0.0, 10.0, 20.0], [0.0, 10.0, 0.0])
    assert f([5.0, 15.0]) == pytest.approx([5.0, 5.0])
    assert f(30.0) == pytest.approx(0.0)
    assert f(-5.0) == pytest.approx(0.0)


def test_spline_reproduces_a_cubic():
    cubic = lambda t: 0.5 * t**3 - 2.0 * t**2 + t - 4.0
    f = Spline().fit(T, cubic(T))
    q = np.array([0.3, 2.5, 7.75])
    assert f(q) == pytest.approx(cubic(q))


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_piecewise_reproduces_its_own_degree(degree):
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=degree + 1)
    # 1 + step * m observations, so the windows tile the series exactly and
    # every piece has its full degree.
    t = np.arange(0.0, 1 + (degree + 1) * 6)
    f = PiecewisePolynomial(degree=degree).fit(t, np.polyval(coeffs, t))
    q = np.array([1.5, t[-1] / 2 + 0.3, t[-1] - 0.1])
    assert f(q) == pytest.approx(np.polyval(coeffs, q), rel=1e-6, abs=1e-6)


def test_piecewise_short_tail_drops_its_degree():
    # Documented choice: a final window holding fewer than degree + 1
    # observations is fitted at the highest degree it can support.
    t = np.arange(0.0, 7.0)  # degree 2 -> windows [0..3], [3..6]: exact
    pp = PiecewisePolynomial(degree=2).fit(np.r_[t, 7.0], np.r_[t**2, 49.0])
    lo, hi, _, coeffs = pp._state[-1]
    assert (lo, hi) == (6.0, 7.0)
    assert len(coeffs) == 2  # a line through the last two observations


def test_piecewise_is_well_conditioned_over_a_long_record():
    # Six months of hourly UNIX seconds: windows far from the series start
    # must not make the least-squares fit ill-conditioned.
    import warnings

    n = 4387
    t = 1.4436e9 + np.arange(n) * 3600.0
    v = 8 + 3 * np.sin(np.arange(n) / 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # a RankWarning would fail the test
        for degree in range(2, 8):
            fitted = PiecewisePolynomial(degree=degree).fit(t, v)(t)
            assert np.all(np.isfinite(fitted))
            assert np.max(np.abs(fitted - v)) < 0.05


def test_piecewise_is_least_squares_not_interpolating():
    rng = np.random.default_rng(0)
    v = np.sin(T) + rng.normal(scale=0.3, size=T.size)
    pp = PiecewisePolynomial(degree=2).fit(T, v)
    spline = Spline().fit(T, v)
    assert np.max(np.abs(pp(T) - v)) > 1e-3       # passes near, not through
    assert spline(T) == pytest.approx(v)            # passes through


def test_piecewise_survives_unix_epoch_times():
    # Raw UNIX seconds to the 7th power overflow an uncentred Vandermonde fit.
    t = 1.4546e9 + np.arange(0.0, 25.0) * 3600.0
    f = PiecewisePolynomial(degree=7).fit(t, np.sin(np.arange(25.0) / 4))
    assert np.all(np.isfinite(f(t)))


def test_piecewise_window_defaults():
    pp = PiecewisePolynomial(degree=3)
    assert (pp.window, pp.step) == (5, 4)


@pytest.mark.parametrize("kwargs", [{"degree": 0}, {"degree": 3, "window": 3}, {"step": 0}])
def test_piecewise_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        PiecewisePolynomial(**kwargs)


@pytest.mark.parametrize("cls", [DLT, PiecewisePolynomial, Spline])
def test_unsorted_and_missing_observations_are_handled(cls):
    order = np.random.default_rng(3).permutation(T.size)
    v = 2.0 * T + 1.0
    t_in, v_in = T[order].copy(), v[order].copy()
    t_in[0] = np.nan
    f = cls().fit(t_in, v_in)
    assert f(4.5) == pytest.approx(10.0, abs=1e-6)


def test_duplicate_timestamps_collapse_to_their_mean():
    f = DLT().fit([0.0, 0.0, 10.0], [0.0, 2.0, 10.0])
    assert f(0.0) == pytest.approx(1.0)


def test_duplicate_bearings_collapse_to_circular_mean():
    f = DLT(circular=True).fit([0.0, 0.0, 10.0], [350.0, 10.0, 0.0])
    assert abs(angular_residual(f(0.0), 0.0)) < 1e-9


@pytest.mark.parametrize("cls", [DLT, Spline, PiecewisePolynomial])
def test_circular_series_turns_through_north(cls):
    t = np.arange(0.0, 8.0)
    bearings = np.mod(340.0 + 5.0 * t, 360.0)  # 340, 345, ..., 15
    out = cls(circular=True).fit(t, bearings)(3.5)
    assert abs(angular_residual(out, 357.5)) < 0.5


def test_too_few_observations_yield_nan():
    assert np.isnan(Spline().fit([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])(1.5))
    assert np.isnan(DLT().fit([0.0], [1.0])(0.0))


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        DLT()(0.0)


def test_scalar_in_scalar_out():
    assert isinstance(DLT().fit(T, T)(3.0), float)
