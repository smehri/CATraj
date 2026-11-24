import numpy as np
import pytest

from catraj.annotation import COMPASS_STEP, angular_residual, decode_compass
from catraj.annotation._circular import from_components, to_components


@pytest.mark.parametrize("deg", [0.0, 45.0, 179.0, 270.0, 359.5])
def test_components_round_trip(deg):
    u, v = to_components(deg)
    assert from_components(u, v) == pytest.approx(deg)


def test_from_components_wraps_into_0_360():
    out = from_components(np.array([1.0, 0.0]), np.array([-1e-12, -1.0]))
    assert np.all((out >= 0) & (out < 360))
    assert out[1] == pytest.approx(270.0)


def test_mean_of_350_and_10_is_north_not_south():
    u, v = to_components(np.array([350.0, 10.0]))
    mean = from_components(u.mean(), v.mean())
    assert abs(angular_residual(mean, 0.0)) < 1e-9


@pytest.mark.parametrize(
    "pred, obs, expected",
    [(1.0, 359.0, 2.0), (359.0, 1.0, -2.0), (90.0, 270.0, -180.0), (10.0, 10.0, 0.0)],
)
def test_angular_residual_takes_short_arc(pred, obs, expected):
    assert angular_residual(pred, obs) == pytest.approx(expected)


def test_decode_compass():
    out = decode_compass([1, 5, 9, 16, 17])
    assert COMPASS_STEP == 22.5
    assert out[:4] == pytest.approx([0.0, 90.0, 180.0, 337.5])  # 1 is north
    assert np.isnan(out[4])  # 17: calm, not a direction


def test_decode_compass_follows_the_archive_table():
    codes = np.arange(0, 21, dtype=float)
    out = decode_compass(codes)
    assert np.isnan(out[0]) and np.isnan(out[17:]).all()  # 0, calm, variable, no data
    assert out[1:17] == pytest.approx((codes[1:17] - 1) * 22.5)
    assert np.isnan(decode_compass([np.nan])[0])


def test_decode_compass_can_start_at_zero():
    out = decode_compass([0, 4, 8, 15, 16], north=0)
    assert out[:4] == pytest.approx([0.0, 90.0, 180.0, 337.5])
    assert np.isnan(out[4])
