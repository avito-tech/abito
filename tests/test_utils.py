import pytest
import numpy as np
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
    assert np.array_equal(w, [1, 3, 1])

