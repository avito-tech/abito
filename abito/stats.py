from typing import Union, Iterable
from cached_property import cached_property
import numpy as np
from scipy.stats import rankdata, tiecorrect, shapiro, distributions, median_test, levene, mood

from .utils import argtrimw, compress_1d, compress_2d
from .result_tuples import *
from .bootstrap import get_bootstrap_dist


np.warnings.filterwarnings('ignore')


def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide='ignore', invalid='ignore'):
        df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)
    return df, denom


def _equal_var_ttest_denom(v1, n1, v2, n2):
    df = n1 + n2 - 2.0
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    return df, denom


def _t_test_from_stats(mean1, mean2, denom, df):
    d = mean1 - mean2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, denom)
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail

    return TTestResult(p_value=prob, statistic=t, mean_diff=d, mean_diff_std=denom)


def t_test_from_stats(mean1, std1, nobs1, mean2, std2, nobs2,
                      equal_var=False):
    """
    :return: TTestResult(statistic, p_value, mean_diff, mean_diff_std)
    """
    if equal_var:
        df, denom = _equal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2)
    else:
        df, denom = _unequal_var_ttest_denom(std1**2, nobs1,
                                             std2**2, nobs2)

    return _t_test_from_stats(mean1, mean2, denom, df)


def _mann_whitney_u_statistic(x: Iterable, y: Iterable, use_continuity: bool = True):

    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x

    t = tiecorrect(ranked)
    std = np.sqrt(t * n1 * n2 * (n1 + n2 + 1) / 12.0)

    mean = n1 * n2 / 2.0 + 0.5 * use_continuity

    return u1, mean, std


def mann_whitney_u_test_from_stats(u1: float, mean: float, std: float) -> MannWhitneyUTestResult:
    """
    :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
    """
    z = np.divide(-(u1 - mean), std)
    p = 2 * distributions.norm.sf(abs(z))
    return MannWhitneyUTestResult(p_value=p, u_statistic=u1, z_statistic=z)


def mann_whitney_u_test(x: Iterable, y: Iterable, use_continuity: bool = True):
    """
    :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
    """
    stats = _mann_whitney_u_statistic(x, y, use_continuity)
    return mann_whitney_u_test_from_stats(*stats)


class LinearSample:
    """
    Statistics and significance tests for 1d-sample.
    """
    def __init__(self, obs: Iterable[Union[int, float]]):
        obs = np.asarray(obs)
        self.obs = obs

    @property
    def full(self):
        """
        :return: full sample
        """
        return self.obs

    @property
    def nobs(self):
        """
        :return: number of observations
        """
        return self.obs.shape[0]

    @property
    def sum(self) -> np.float:
        """
        :return: sample sum
        """
        return self.obs.sum()

    @property
    def mean(self) -> np.float:
        """
        :return: sample mean
        """
        return np.divide(self.sum, self.nobs)

    @property
    def demeaned(self) -> np.ndarray:
        """
        :return: demeaned observations
        """
        return self.obs - self.mean

    @property
    def demeaned_sumsquares(self) -> np.ndarray:
        """
        :return: sum of squares of demeaned observations
        """
        return (self.demeaned**2).sum()

    @property
    def var(self) -> np.float:
        """
        :return: sample variance, unbiased
        """
        return self.demeaned_sumsquares / (self.nobs - 1)

    @property
    def std(self) -> np.float:
        """
        :return: sample standard deviation, square root of the sample variance
        """
        return np.sqrt(self.var)

    @property
    def mean_std(self):
        """
        :return: sample standard deviation of the sample mean
        """
        return self.std / np.sqrt(self.nobs)

    def t_test_1samp(self, popmean: float) -> TTestResult:
        """
        :param popmean: population mean
        :return: TTestResult(statistic, p_value, mean_diff, mean_diff_std)
        """
        return _t_test_from_stats(self.mean, popmean, self.mean_std, self.nobs - 1)

    def t_test(self, control: Union['LinearSample', 'LinearSampleW'], equal_var: bool = False) -> TTestResult:
        """
        :param control: control sample
        :param equal_var:
        If True (default), perform a standard independent 2 sample test that assumes equal population
        variances [https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test].
        If False, perform Welch’s t-test, which does not assume equal population variance
        [https://en.wikipedia.org/wiki/Welch%27s_t-test].
        :return: TTestResult(statistic, p_value, mean_diff, mean_diff_std)
        """
        return t_test_from_stats(self.mean, self.std, self.nobs,
                                 control.mean, control.std, control.nobs,
                                 equal_var=equal_var)

    def mann_whitney_u_test(self, control: Union['LinearSample', 'LinearSampleW'],
                            use_continuity=True) -> MannWhitneyUTestResult:
        """
        :param control: control sample
        :param use_continuity: use continuity correction
        :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
        """
        res = mann_whitney_u_test(self.full, control.full, use_continuity)  # TODO: support weighted samples
        return res

    def shapiro_test(self) -> ShapiroTestResult:
        """
        Shapiro-Wilk test of normality
        :return: ShapiroTestResult(statistic, p_value)
        """
        if self.nobs < 3:
            return ShapiroTestResult(np.nan, np.nan)
        res = shapiro(self.full)
        return ShapiroTestResult(p_value=res[1], statistic=res[0])

    def median_test(self, control: Union['LinearSample', 'LinearSampleW']) -> MedianTestResult:
        """
        :param control: control sample
        :return: MedianTestResult(statistic, p_value, grand_median)
        """
        if self.std == 0:
            return MedianTestResult(np.nan, np.nan, np.nan)
        res = median_test(self.full, control.full)
        return MedianTestResult(p_value=res[1], statistic=res[0], grand_median=res[2])

    def levene_test(self, control: Union['LinearSample', 'LinearSampleW']) -> LeveneTestResult:
        """
        :param control: control sample
        :return: LeveneTestResult(statistic, p_value)
        """
        res = levene(self.full, control.full)
        return LeveneTestResult(p_value=res[1], statistic=res[0])

    def mood_test(self, control: Union['LinearSample', 'LinearSampleW']) -> MoodTestResult:
        """
        :param control: control sample
        :return: MoodTestResult(statistic, p_value)
        """
        res = mood(self.full, control.full)
        return MoodTestResult(statistic=res[0], p_value=res[1])


class LinearSampleW(LinearSample):
    """
    Statistics and significance tests for 1d-sample. Weighted version.
    """
    def __init__(
            self,
            obs: Iterable[Union[int, float]],
            weights: Iterable[Union[int, float]],
    ):
        super().__init__(obs)
        weights = np.floor(weights).astype(int)
        self.weights = weights

    @property
    def full(self):
        """
        :return: full sample after weights applied
        """
        return np.repeat(self.obs, self.weights, axis=0)

    @property
    def nobs(self):
        """
        :return: number of observations, sum of weights
        """
        return self.weights.sum()

    @property
    def sum(self) -> np.float:
        """
        :return: sample sum
        """
        return np.float(np.dot(self.obs.T, self.weights))

    @property
    def demeaned_sumsquares(self) -> np.ndarray:
        """
        :return: sum of squares of demeaned observations
        """
        return np.dot((self.demeaned**2).T, self.weights)


class RatioSample(LinearSample):
    bootstrap_n_threads: int = 1
    num: Union['LinearSample', 'LinearSampleW'] = None
    den: Union['LinearSample', 'LinearSampleW'] = None

    def __init__(self, obs):
        super().__init__(obs)

    @cached_property
    def bootstrap_mean_dist(self) -> np.ndarray:
        """
        :return: bootstrap distribution of sample mean
        """
        # TODO: support weighted samples
        return get_bootstrap_dist(self.num.full, self.den.full, 10000, 1, self.bootstrap_n_threads)

    @cached_property
    def bootstrap_mean(self) -> np.ndarray:
        """
        :return: median of bootstrap distribution of sample mean
        """
        return np.median(self.bootstrap_mean_dist)

    @cached_property
    def bootstrap_mean_std(self) -> np.ndarray:
        """
        :return: standard deviation of bootstrap distribution of sample mean
        """
        return np.std(self.bootstrap_mean_dist)

    def bootstrap_test(self, control: Union['RatioSample', 'RatioSampleW']) -> BootstrapTestResult:
        """
        :param control: control sample
        :return: BootstrapTestResult(statistic, p_value, mean_diff, mean_diff_std)
        """
        dist = self.bootstrap_mean_dist - control.bootstrap_mean_dist
        p = min(
            (dist <= 0).mean(axis=0),
            (dist > 0).mean(axis=0),
        ) * 2
        md = np.median(dist)
        mds = np.std(dist)
        s = np.divide(md, mds)
        return BootstrapTestResult(statistic=s, p_value=p, mean_diff=md, mean_diff_std=mds)


class RatioSampleW(LinearSampleW, RatioSample):
    pass


def linear_sample_factory(
    obs: Iterable[Union[int, float]],
    weights: Iterable[Union[int, float]] = None,
    ltrim: float = None,
    rtrim: float = None,
    compress: bool = False,
):
    obs = np.asarray(obs)

    if weights is None:
        if compress:
            obs, weights = compress_1d(obs)

    if ltrim or rtrim:
        ind, weights = argtrimw(obs, weights, ltrim, rtrim)
        obs = obs[ind]

    if weights is not None:
        return LinearSampleW(obs, weights)
    else:
        return LinearSample(obs)


def prepare_ratio_components(num, den, linstrat, weights, ltrim, rtrim):
    num = linear_sample_factory(num, weights)
    den = linear_sample_factory(den, weights)

    if linstrat == 'taylor':
        r = num.sum / den.sum
        obs = (num.obs - r * den.obs) / den.mean + r
    elif linstrat == 'naive':
        obs = num.obs / den.obs
    else:
        raise ValueError("linearizarion strategy must be 'taylor' or 'naive'")

    if ltrim or rtrim:
        ind, weights = argtrimw(obs, weights, ltrim, rtrim)
        obs = obs[ind]
        num = linear_sample_factory(num.obs[ind], weights)
        den = linear_sample_factory(den.obs[ind], weights)

    return obs, num, den, weights


def ratio_sample_factory(
        num: Iterable[Union[int, float]],
        den: Iterable[Union[int, float]],
        linstrat: str,
        weights: Iterable[Union[int, float]] = None,
        ltrim: float = None,
        rtrim: float = None,
        compress: bool = False,
        bootstrap_n_threads: int = 1,
):
    if weights is None:
        if compress:
            u, weights = compress_2d(np.array([num, den]).T)
            num, den = u[:, 0], u[:, 1]

    obs, num, den, weights = prepare_ratio_components(num, den, linstrat, weights, ltrim, rtrim)

    if weights is not None:
        r = RatioSampleW(obs, weights)
    else:
        r = RatioSample(obs)

    r.num = num
    r.den = den
    r.bootstrap_n_threads = bootstrap_n_threads

    return r


def sample_factory(
        num: Iterable[Union[int, float]],
        den: Iterable[Union[int, float]] = None,
        weights: Iterable[Union[int, float]] = None,
        linstrat: str = 'taylor',
        ltrim: float = None,
        rtrim: float = None,
        compress: bool = False,
        bootstrap_n_threads: int = 1,
):
    """
    :param num: numerator of observed data
    :param den: denominator of observed data
    :param weights: weights of observations. Length must be the same as obs's
    :param linstrat:
    lineariztion strategy. 'naive' when ratio computes element-wize. 'taylor' to use Taylor's expansion at mean.
    :param ltrim: proportion of data to cut from left tail
    :param rtrim: proportion of data to cut from right tail
    :param compress: turn not-weighted sample into weighted version
    :param bootstrap_n_threads: number of processes to use in bootstrapping methods
    """
    if den is None:
        return linear_sample_factory(num, weights, ltrim, rtrim, compress)
    else:
        return ratio_sample_factory(num, den, linstrat, weights, ltrim, rtrim, compress, bootstrap_n_threads)


Sample = sample_factory
