import pytest
from abito.utils import *


@pytest.fixture()
def sample():
    return np.array([1, 2, 3])


@pytest.fixture()
def weights():
    return np.array([3, 3, 3])


def test_argtrim(sample):
    t = sample[argtrim(sample, 0.34, 0.34)]
    assert t.shape == (1,)
    assert not 1 in t
    assert not 3 in t
    assert 2 in t


def test_argtrimw(sample, weights):
    ind, w = argtrimw(sample, weights, 0.23, 0.23)
    t = sample[ind]
    assert t.shape == (3,)
    np.testing.assert_array_equal(w, [1, 3, 1])

    ind, w = argtrimw([1, 2, 3], [1, 200, 1], 0.01, 0.01)
    np.testing.assert_array_equal(w, [198])


def test_utils_exceptions():
    np.testing.assert_array_equal(argtrim([1, 2, 3]), [0, 1, 2])
    assert argtrim([1, 2, 3], ltrim=1.01) == []
    assert argtrim([1, 2, 3], rtrim=1.01) == []

    res, weights = argtrimw([1, 2, 3], [2, 2, 2])
    np.testing.assert_array_equal(res, [0, 1, 2])
    np.testing.assert_array_equal(weights, [2, 2, 2])

    assert argtrimw([1, 2], [2, 2], ltrim=1.01) == ([], [])
    assert argtrimw([1, 2], [2, 2], rtrim=1.01) == ([], [])
