import pytest
from abito.utils import *


@pytest.fixture()
def sample():
    return np.array([1, 5, 3, 4, 2])


@pytest.fixture()
def weights():
    return np.array([1, 1, 99, 100, 99])


def test_argtrim(sample):
    t = sample[argtrim(sample, 0.2, 0.2)]
    np.testing.assert_array_equal(t, [2, 3, 4])


def test_argtrimw(sample, weights):
    ind, w = argtrimw(sample, weights, 0.2, 0.2)
    t = sample[ind]
    np.testing.assert_array_equal(t, [2, 3, 4])
    np.testing.assert_array_equal(w, [40, 99, 41])


def test_utils_exceptions():
    np.testing.assert_array_equal(argtrim([1, 2, 3]), [0, 1, 2])
    assert argtrim([1, 2, 3], ltrim=1.01) == []
    assert argtrim([1, 2, 3], rtrim=1.01) == []

    res, weights = argtrimw([1, 2, 3], [2, 2, 2])
    np.testing.assert_array_equal(res, [0, 1, 2])
    np.testing.assert_array_equal(weights, [2, 2, 2])

    assert argtrimw([1, 2], [2, 2], ltrim=1.01) == ([], [])
    assert argtrimw([1, 2], [2, 2], rtrim=1.01) == ([], [])


def test_compress_sample():
    num = [1, 2, 1, 3]
    den = [1, 1, 2, 2]
    weights = [3, 2, 2, 2]

    num_c, den_c, weights_c = compress_sample(num, None, None)
    np.testing.assert_array_equal(num_c, [1, 2, 3])
    np.testing.assert_array_equal(den_c, [1, 1, 1])
    np.testing.assert_array_equal(weights_c, [2, 1, 1])

    num_c, den_c, weights_c = compress_sample(num, den, None)
    np.testing.assert_array_equal(num_c, [1, 1, 2, 3])
    np.testing.assert_array_equal(den_c, [1, 2, 1, 2])
    np.testing.assert_array_equal(weights_c, [1, 1, 1, 1])

    num_c, den_c, weights_c = compress_sample(num, None, weights)
    np.testing.assert_array_equal(num_c, [1, 2, 3])
    np.testing.assert_array_equal(den_c, [1, 1, 1])
    np.testing.assert_array_equal(weights_c, [5, 2, 2])

    num_c, den_c, weights_c = compress_sample(num, den, weights)
    np.testing.assert_array_equal(num_c, [1, 1, 2, 3])
    np.testing.assert_array_equal(den_c, [1, 2, 1, 2])
    np.testing.assert_array_equal(weights_c, [3, 2, 2, 2])
