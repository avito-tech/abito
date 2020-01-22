import pytest
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu, shapiro, median_test, levene, mood
import abito as ab


@pytest.fixture()
def normal_obs():
    np.random.seed(1)
    return np.random.normal(loc=102, scale=10, size=1000)


@pytest.fixture()
def normal_obs_control():
    np.random.seed(2)
    return np.random.normal(loc=100, scale=10, size=1000)


@pytest.fixture()
def poisson_obs():
    np.random.seed(1)
    return np.random.poisson(1, size=1000)


@pytest.fixture()
def poisson_obs_control():
    np.random.seed(2)
    return np.random.poisson(1, size=1000)


def _subtest_compare_sample_vs_obs(sample, obs):
    assert sample.nobs == obs.shape[0]
    assert sample.sum() == obs.sum()
    assert sample.mean() == obs.mean()
    assert sample.var() == pytest.approx(obs.var(ddof=1), 1e-6)
    assert sample.std() == obs.std(ddof=1)
    assert sample.mean_std() == obs.std(ddof=1) / np.sqrt(sample.nobs)
    assert sample.median() == np.median(obs)
    assert sample.quantile(0.5) == sample.median()
    np.testing.assert_array_equal(sample.quantile([0.4, 0.6]), np.quantile(obs, [0.4, 0.6]))


def test_sample_stats(normal_obs):
    np.random.seed(1)
    treatment = ab.sample(normal_obs)
    _subtest_compare_sample_vs_obs(treatment, normal_obs)


def test_sample_weighted(poisson_obs):
    np.random.seed(1)
    treatment = ab.sample(poisson_obs).reweigh()
    shuffle_ind = np.arange(treatment.obs.shape[0])
    np.random.shuffle(shuffle_ind)
    treatment.obs = treatment.obs[shuffle_ind]
    treatment.weights = treatment.weights[shuffle_ind]
    _subtest_compare_sample_vs_obs(treatment, poisson_obs)


def test_ratio(poisson_obs, poisson_obs_control):
    s = ab.sample(poisson_obs, poisson_obs_control).reweigh()
    assert s.nobs == 1000
    assert s.ratio() == poisson_obs.sum() / poisson_obs_control.sum()
    np.testing.assert_array_equal(s.weights, s.numsamp.weights)
    np.testing.assert_array_equal(s.weights, s.densamp.weights)
    np.testing.assert_array_equal(s.num, s.numsamp.obs)
    np.testing.assert_array_equal(s.den, s.densamp.obs)


def test_linearize(poisson_obs, poisson_obs_control):
    s = ab.sample(poisson_obs, poisson_obs_control + 1).reweigh()
    lin = s.linearize(strategy='taylor')
    assert s.ratio() == pytest.approx(lin.mean(), 1e-6)

    lin = s.linearize(strategy='naive')
    assert lin.mean() != s.ratio()


def test_significance_tests(normal_obs, normal_obs_control):
    treatment = ab.sample(normal_obs)
    control = ab.sample(normal_obs_control)
    res = treatment.t_test(control, equal_var=True)
    res_expected = ttest_ind(normal_obs, normal_obs_control, equal_var=True)
    assert res.p_value == res_expected.pvalue
    assert res.statistic == res_expected.statistic

    res = treatment.t_test(control, equal_var=False)
    res_expected = ttest_ind(normal_obs, normal_obs_control, equal_var=False)
    assert res.p_value == res_expected.pvalue
    assert res.statistic == res_expected.statistic

    res = treatment.t_test_1samp(101)
    res_expected = ttest_1samp(normal_obs, 101)
    assert res.p_value == res_expected.pvalue
    assert res.statistic == res_expected.statistic

    res = treatment.mann_whitney_u_test(control)
    res_expected = mannwhitneyu(normal_obs_control, normal_obs, alternative='two-sided')
    assert res.p_value == pytest.approx(res_expected.pvalue, 1e-6)
    assert res.u_statistic == res_expected.statistic

    res = treatment.shapiro_test()
    res_expected = shapiro(normal_obs)
    assert res.statistic == res_expected[0]
    assert res.p_value == res_expected[1]

    res = treatment.median_test(control)
    res_expected = median_test(normal_obs, normal_obs_control)
    assert res.p_value == res_expected[1]
    assert res.statistic == res_expected[0]
    assert res.grand_median == res_expected[2]

    res = treatment.levene_test(control)
    res_expected = levene(normal_obs, normal_obs_control)
    assert res.p_value == res_expected.pvalue
    assert res.statistic == res_expected.statistic

    res = treatment.mood_test(control)
    res_expected = mood(normal_obs, normal_obs_control)
    assert res.p_value == res_expected[1]
    assert res.statistic == res_expected[0]


def _subtest_equality(sample1, sample2):
    assert sample1.mean() == pytest.approx(sample2.mean(), 1e-6)
    assert sample1.var() == pytest.approx(sample2.var(), 0.02)
    assert sample1.std() == pytest.approx(sample2.std(), 0.01)
    assert sample1.mean_std() == pytest.approx(sample2.mean_std(), 0.01)
    assert sample1.nobs == sample2.nobs
    assert sample1.median() == sample2.median()
    assert sample1.fullobs.sum() == pytest.approx(sample2.fullobs.sum(), 1e-6)


def test_reweigh(poisson_obs):
    s = ab.sample(poisson_obs)
    sc = ab.sample(poisson_obs).reweigh()
    sc.reweigh(inplace=True)
    _subtest_equality(s, sc)


def test_compress(poisson_obs, poisson_obs_control):
    s = ab.sample(poisson_obs)
    sc = s.compress(n_buckets=100)
    _subtest_equality(s, sc)

    sc = ab.sample(poisson_obs)
    sc.compress(n_buckets=100, inplace=True)
    _subtest_equality(s, sc)

    s = ab.sample(poisson_obs, poisson_obs_control + 1)
    sc = s.compress(n_buckets=100, sort_by='den')
    assert s.ratio() == pytest.approx(sc.ratio(), 1e-6)

    sc = s.compress(n_buckets=100, sort_by='num', weights_dist='multinomial')
    assert s.ratio() == pytest.approx(sc.ratio(), 1e-6)

    sc = s.copy()
    sc.compress(n_buckets=100, sort_by='taylor', reweigh=True)
    assert s.ratio() == pytest.approx(sc.ratio(), 1e-6)

    with pytest.raises(ValueError):
        sc = s.compress(n_buckets=100, sort_by='num', weights_dist='')

    with pytest.raises(ValueError):
        sc = s.compress(n_buckets=100, sort_by='num', stat='sum')


def test_trim(normal_obs, poisson_obs, poisson_obs_control):
    s = ab.sample(normal_obs)
    assert s.trim(rtrim=0.01, ltrim=0.01).nobs == 980

    s = ab.sample(poisson_obs, poisson_obs_control).reweigh()
    assert s.trim(rtrim=0.01, ltrim=0.01, sort_by='num').nobs == 980
    assert s.trim(rtrim=0.01, ltrim=0.01, sort_by='den').nobs == 980
    assert s.trim(rtrim=0.01, ltrim=0.01, sort_by='taylor').nobs == 980


def test_exceptions():
    res = ab.sample([1, 2]).shapiro_test()
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)

    res = ab.sample(np.ones(100)).median_test(ab.sample(np.ones(100)))
    assert np.isnan(res.statistic)
    assert np.isnan(res.p_value)
    assert np.isnan(res.grand_median)

    with pytest.raises(ValueError):
        s = ab.sample([1, 2], [1, 1]).linearize('')

    ar = np.array([1, 2, 3], dtype='float')
    rw = ab.compress.reweigh(ar)


def _subtest_bootstrap(sample, sample_control):
    n_iters = 1000
    bs = sample.bootstrap_estimates('mean', n_iters)
    assert bs.size == n_iters

    np.random.seed(3)
    bst = sample.bootstrap_test(sample_control, 'mean', n_iters, n_threads=2)
    res_expected = sample.t_test(sample_control)
    assert bst.t_statistic == pytest.approx(res_expected.statistic, 0.1)
    assert bst.t_p_value == pytest.approx(res_expected.p_value, 0.5)
    assert bst.est_p_value == pytest.approx(res_expected.p_value, 0.7)

    bs = sample.bootstrap_estimates('mean', n_iters, cache_result=True)
    assert sample._get_from_cache(stat='mean', n_iters=n_iters) is not None
    assert sample.bootstrap_estimates('mean', n_iters, cache_result=True) is not None
    sample._del_from_cache(stat='mean', n_iters=n_iters)
    assert sample._get_from_cache(stat='mean', n_iters=n_iters) is None
    del sample.cache
    assert sample.cache == {}

    bs = sample.bootstrap_estimates('quantile', n_iters, q=0.5)
    assert bs.size == n_iters


def test_bootstrap(poisson_obs, poisson_obs_control):
    st = ab.sample(poisson_obs)
    sc = ab.sample(poisson_obs_control)
    _subtest_bootstrap(st, sc)

    st = st.reweigh()
    _subtest_bootstrap(st, sc)


def test_sample_factory():
    s = ab.sample([1, 2], weights=[2, 5])
    assert s.mean() == 12 / 7
