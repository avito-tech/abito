import pytest
from abito import RatioSample
import numpy as np


@pytest.fixture()
def treatment():
    np.random.seed(1)
    return RatioSample(np.random.normal(loc=102, scale=10, size=10000))


@pytest.fixture()
def control():
    np.random.seed(2)
    return RatioSample(np.random.normal(loc=100, scale=10, size=10000))


@pytest.fixture()
def trimmed_treatment():
    np.random.seed(1)
    return RatioSample(np.random.normal(loc=102, scale=10, size=200), rtrim=0.01)


@pytest.fixture()
def ratio_treatment():
    np.random.seed(1)
    num = np.random.normal(50000, 1000, size=200)
    den = np.random.normal(100000, 1000, size=200)
    return RatioSample(num, den)


@pytest.fixture()
def ratio_control():
    np.random.seed(1)
    num = np.random.normal(50100, 1000, size=200)
    den = np.random.normal(100000, 1000, size=200)
    return RatioSample(num, den)


def test_sample_stats(treatment):
    assert treatment.mean == pytest.approx(102, 0.02)
    assert treatment.var == pytest.approx(100, 0.02)
    assert treatment.std == pytest.approx(10, 0.02)
    assert treatment.mean_std == pytest.approx(0.1, 0.02)
    assert treatment.nobs == 10000


def test_significance(treatment, control):
    res = treatment.t_test(control)
    assert res.p_value == pytest.approx(0, 0.0001)
    assert res.statistic == pytest.approx(16.19, 0.01)

    res = treatment.mann_whitney_u_test(control)
    assert res.p_value == pytest.approx(0, 0.0001)
    assert res.u_statistic == 43551416.0
    assert res.z_statistic == pytest.approx(15.79, 0.01)

    res = treatment.shapiro_test()
    assert res.statistic == pytest.approx(0.99, 0.01)
    assert res.p_value == pytest.approx(0.59, 0.01)

    res = treatment.median_test(control)
    assert res.p_value == pytest.approx(0, 0.0001)
    assert res.statistic == pytest.approx(172.6082, 0.01)
    assert res.grand_median == pytest.approx(100.944, 0.01)

    res = treatment.levene_test(control)
    assert res.p_value == pytest.approx(0.893442, 0.0001)
    assert res.statistic == pytest.approx(0.0179429, 0.01)

    res = treatment.mood_test(control)
    assert res.p_value == pytest.approx(0.872408, 0.0001)
    assert res.statistic == pytest.approx(0.160599, 0.01)


def test_trimmed_treatment(trimmed_treatment):
    assert trimmed_treatment.nobs == 198


def test_bootstrap(ratio_treatment, ratio_control):
    assert ratio_treatment.bootstrap_mean_dist.shape == (10000,)
    assert ratio_treatment.bootstrap_mean == pytest.approx(ratio_treatment.mean, 0.02)
    assert ratio_treatment.bootstrap_mean_std == pytest.approx(ratio_treatment.mean_std, 0.02)

    t = ratio_treatment.t_test(ratio_control)
    bs = ratio_treatment.bootstrap_test(ratio_control)

    assert t.statistic == pytest.approx(bs.statistic, 0.01)
    assert t.p_value == pytest.approx(bs.p_value, 0.03)
    assert t.mean_diff == pytest.approx(bs.mean_diff, 0.01)
    assert t.mean_diff_std == pytest.approx(bs.mean_diff_std, 0.01)
