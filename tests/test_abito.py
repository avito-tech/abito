import pytest
import numpy as np
from abito import RatioSample
import abito.utils as utils


def test_sample_stats(treatment):
    assert treatment.mean == pytest.approx(102, 0.02)
    assert treatment.var == pytest.approx(80, 0.02)
    assert treatment.std == pytest.approx(9, 0.02)
    assert treatment.mean_std == pytest.approx(0.9, 0.02)
    assert treatment.nobs == 100


def test_significance(treatment, control):
    res = treatment.t_test(control, equal_var=True)
    assert res.p_value == pytest.approx(0.0085, 0.01)
    assert res.statistic == pytest.approx(2.65, 0.01)

    res = treatment.t_test(control, equal_var=False)
    assert res.p_value == pytest.approx(0.0085, 0.01)
    assert res.statistic == pytest.approx(2.65, 0.01)

    res = treatment.t_test_1samp(101)
    assert res.p_value == pytest.approx(0.074, 0.01)
    assert res.statistic == pytest.approx(1.805, 0.01)

    res = treatment.mann_whitney_u_test(control)
    assert res.p_value == pytest.approx(0.0217, 0.01)
    assert res.u_statistic == 4061.0
    assert res.z_statistic == pytest.approx(2.295, 0.01)

    res = treatment.shapiro_test()
    assert res.statistic == pytest.approx(0.992, 0.01)
    assert res.p_value == pytest.approx(0.821, 0.01)

    res = treatment.median_test(control)
    assert res.p_value == pytest.approx(0.119, 0.01)
    assert res.statistic == pytest.approx(2.42, 0.01)
    assert res.grand_median == pytest.approx(100, 0.01)

    res = treatment.levene_test(control)
    assert res.p_value == pytest.approx(0.232, 0.01)
    assert res.statistic == pytest.approx(1.433, 0.01)

    res = treatment.mood_test(control)
    assert res.p_value == pytest.approx(0.320, 0.01)
    assert res.statistic == pytest.approx(-0.992, 0.01)


def test_exceptions(treatment_length2, treatment_equalobs, control_equalobs):
    res = treatment_length2.shapiro_test()
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)

    res = treatment_equalobs.median_test(control_equalobs)
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)
    assert np.isnan(res.grand_median)

    with pytest.raises(ValueError):
        RatioSample([1, 2], [1, 2], linstrat='')


def test_ratio_naive_linstrat(ratio_sample_naive_linstrat):
    np.testing.assert_array_equal(ratio_sample_naive_linstrat.obs, [1, 1])


def test_trimmed_weighted_samples(
        trimmed_treatment,
        trimmed_ratio_treatment,
        weighted_treatment,
        weighted_trimmed_treatment,
        weighted_trimmed_ratio_treatment
):
    assert weighted_treatment.nobs == 600
    assert weighted_treatment.full.shape[0] == 600
    assert weighted_treatment.sum == 1200
    assert weighted_treatment.demeaned_sumsquares == 400

    assert trimmed_treatment.nobs == 196
    assert trimmed_ratio_treatment.nobs == 196
    assert weighted_trimmed_treatment.nobs == 588
    assert weighted_trimmed_ratio_treatment.nobs == 588


def test_bootstrap(ratio_treatment, ratio_control):
    assert ratio_treatment.bootstrap_mean_dist.shape == (10000,)
    assert ratio_treatment.bootstrap_mean == pytest.approx(ratio_treatment.mean, 0.02)
    assert ratio_treatment.bootstrap_mean_std == pytest.approx(ratio_treatment.mean_std, 0.02)

    t = ratio_treatment.t_test(ratio_control)
    bs = ratio_treatment.bootstrap_test(ratio_control)

    assert t.statistic == pytest.approx(bs.statistic, 0.05)
    assert t.p_value == pytest.approx(bs.p_value, 0.2)
    assert t.mean_diff == pytest.approx(bs.mean_diff, 0.01)
    assert t.mean_diff_std == pytest.approx(bs.mean_diff_std, 0.01)
