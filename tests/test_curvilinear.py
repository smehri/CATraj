import numpy as np
import pytest

from catraj.prediction.curvilinear import KNOT, curvilinear_predict, mean_interval


def test_straight_track_is_predicted_exactly():
    px, py = curvilinear_predict([100.0], [200.0], [5.0], [90.0], 10.0, steps=3)
    assert px.shape == py.shape == (1, 3)
    np.testing.assert_allclose(px, [[150, 200, 250]], atol=1e-9)
    np.testing.assert_allclose(py, [[200, 200, 200]], atol=1e-9)

    px, py = curvilinear_predict([0.0], [0.0], [5.0], [0.0], 10.0)
    np.testing.assert_allclose(px, 0, atol=1e-9)
    np.testing.assert_allclose(py, [[50, 100, 150]], atol=1e-9)

    px, py = curvilinear_predict([0.0], [0.0], [5.0], [180.0], 10.0)
    np.testing.assert_allclose(px, 0, atol=1e-9)
    np.testing.assert_allclose(py, [[-50, -100, -150]], atol=1e-9)


def test_dt_can_differ_per_point():
    px, py = curvilinear_predict(
        [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [90.0, 90.0], [10.0, 20.0], steps=2
    )
    np.testing.assert_allclose(px, [[10, 20], [20, 40]], atol=1e-9)
    np.testing.assert_allclose(py, 0, atol=1e-9)


def test_mean_interval_is_eq_12():
    assert mean_interval([0, 10, 30, 60]) == 20.0


def test_knot_is_one_nautical_mile_per_hour():
    assert KNOT == pytest.approx(0.514444, abs=1e-6)


def test_curvilinear_rejects_bad_input():
    with pytest.raises(ValueError):
        curvilinear_predict([0, 1], [0], [1, 1], [0, 0], 1.0)
    with pytest.raises(ValueError):
        curvilinear_predict([0], [0], [1], [0], -1.0)
    with pytest.raises(ValueError):
        curvilinear_predict([0, 0], [0, 0], [1, 1], [0, 0], [1.0, -1.0])
    with pytest.raises(ValueError):
        curvilinear_predict([0, 0], [0, 0], [1, 1], [0, 0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        mean_interval([5])
    with pytest.raises(ValueError):
        mean_interval([0, 10, 10, 20])
    with pytest.raises(ValueError):
        mean_interval([0, 10, 5])
