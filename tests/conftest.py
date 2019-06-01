import pytest
from abito import LinearSample, RatioSample
import numpy as np


@pytest.fixture()
def treatment():
    np.random.seed(1)
    return LinearSample(np.random.normal(loc=102, scale=10, size=100))


@pytest.fixture()
def control():
    np.random.seed(2)
    return LinearSample(np.random.normal(loc=100, scale=10, size=100))


@pytest.fixture()
def trimmed_treatment():
    np.random.seed(1)
    return LinearSample(
        np.random.normal(loc=102, scale=10, size=200),
        rtrim=0.01, ltrim=0.01)


@pytest.fixture()
def trimmed_ratio_treatment():
    np.random.seed(1)
    return RatioSample(
        np.random.normal(loc=102, scale=10, size=200),
        np.random.normal(loc=200, scale=10, size=200),
        rtrim=0.01, ltrim=0.01)


@pytest.fixture()
def weighted_treatment():
    return RatioSample([1, 2, 3], weights=[200, 200, 200])


@pytest.fixture()
def weighted_trimmed_treatment():
    return RatioSample([1, 2, 3], weights=[200, 200, 200], rtrim=0.01, ltrim=0.01)


@pytest.fixture()
def weighted_trimmed_ratio_treatment():
    return RatioSample([1, 2, 3], den=[5, 4, 3], weights=[200, 200, 200], rtrim=0.01, ltrim=0.01)


@pytest.fixture()
def ratio_treatment():
    np.random.seed(1)
    num = np.random.normal(500, 1000, size=200)
    den = np.random.normal(1000, 1000, size=200)
    return RatioSample(num, den, bootstrap_n_threads=2)


@pytest.fixture()
def ratio_control():
    np.random.seed(1)
    num = np.random.normal(50100, 1000, size=200)
    den = np.random.normal(100000, 1000, size=200)
    return RatioSample(num, den)


@pytest.fixture()
def treatment_length2():
    return LinearSample([1, 2])


@pytest.fixture()
def treatment_equalobs():
    return LinearSample(np.ones(100))


@pytest.fixture()
def control_equalobs():
    return RatioSample(np.ones(100))


@pytest.fixture()
def ratio_sample_naive_linstrat():
    return RatioSample([1, 2], [1, 2], linstrat='naive')
