import numpy as np
import pytest

from catraj.annotation import Metrics, residuals, score


def test_perfect_prediction():
    m = score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert (m.rmse, m.mae, m.me, m.max_ae, m.r2, m.n) == (0, 0, 0, 0, 1, 3)


def test_known_values():
    m = score([3.0, 1.0], [1.0, 1.0])  # errors +2, 0
    assert m.rmse == pytest.approx(np.sqrt(2.0))
    assert m.mae == pytest.approx(1.0)
    assert m.me == pytest.approx(1.0)
    assert m.max_ae == pytest.approx(2.0)


def test_me_reveals_bias_that_rmse_hides():
    biased = score([3.0, 3.0], [1.0, 1.0])
    scattered = score([3.0, -1.0], [1.0, 1.0])
    assert biased.rmse == pytest.approx(scattered.rmse)
    assert biased.me == pytest.approx(2.0)
    assert scattered.me == pytest.approx(0.0)


def test_missing_pairs_are_skipped():
    m = score([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
    assert m.n == 1


def test_nothing_scorable_is_nan_not_an_error():
    m = score([], [])
    assert m.n == 0
    assert np.isnan(m.rmse) and np.isnan(m.r2)


def test_circular_error_crosses_north():
    m = score([359.0], [1.0], circular=True)
    assert m.rmse == pytest.approx(2.0)
    assert residuals([359.0], [1.0], circular=True)[0] == pytest.approx(-2.0)


def test_as_dict_has_every_field():
    assert set(score([1.0], [1.0]).as_dict()) == set(Metrics.__dataclass_fields__)
