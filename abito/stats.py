import logging
from typing import Union, Iterable
from cached_property import cached_property
import numpy as np
from scipy.stats import rankdata, tiecorrect, shapiro, distributions, median_test, levene, mood

from .utils import argtrim, argtrimw
from .result_tuples import *
from .bootstrap import get_bootstrap_dist


np.warnings.filterwarnings('ignore')

logger = logging.getLogger()


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
    if t == 0:
        raise ValueError('All numbers are identical in mannwhitneyu')
    std = np.sqrt(t * n1 * n2 * (n1 + n2 + 1) / 12.0)

    mean = n1 * n2 / 2.0 + 0.5 * use_continuity

    return u1, mean, std


def mann_whitney_u_test_from_stats(u1: float, mean: float, std: float) -> MannWhitneyUTestResult:
    """
    :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
    """
    z = - (u1 - mean) / std
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
    def __init__(
            self,
            obs: Iterable[Union[int, float]],
            weights: Iterable[Union[int, float]] = None,
            ltrim: float = None,
            rtrim: float = None
    ):
        """
        :param obs: sample of observations
        :param weights: weights of observations. Length must be the same as obs's
        :param ltrim: proportion of data to cut from left tail
        :param rtrim: proportion of data to cut from right tail
        """
        obs = np.asarray(obs)

        if weights is not None:
            weights = np.asarray(weights)
            is_weighted = True
        else:
            weights = np.ones(obs.shape[0])
            is_weighted = False

        if ltrim or rtrim:
            if is_weighted:
                ind, weights = argtrimw(obs, weights, ltrim, rtrim)
                obs = obs[ind]
            else:
                ind = argtrim(obs, ltrim, rtrim)
                obs = obs[ind]
                weights = weights[ind]

        self.obs = obs
        self.weights = weights
        self.is_weighted = is_weighted

    @cached_property
    def full(self):
        """
        :return: full sample after weights applied
        """
        if self.is_weighted:
            w_int = np.floor(self.weights).astype(int)
            return np.repeat(self.obs, w_int, axis=0)
        else:
            return self.obs

    @cached_property
    def nobs(self):
        """
        :return: number of observations, sum of weights
        """
        return self.weights.sum()

    @cached_property
    def sum(self) -> np.float:
        """
        :return: sample sum
        """
        res = np.dot(self.obs.T, self.weights)
        return res

    @cached_property
    def mean(self) -> np.float:
        """
        :return: sample mean
        """
        return self.sum / self.nobs

    @cached_property
    def demeaned(self) -> np.ndarray:
        """
        :return: demeaned observations
        """
        return self.obs - self.mean

    @cached_property
    def demeaned_sumsquares(self) -> np.ndarray:
        """
        :return: sum of squares of demeaned observations
        """
        return np.dot((self.demeaned**2).T, self.weights)

    @cached_property
    def var(self) -> np.float:
        """
        :return: sample variance, unbiased
        """
        return self.demeaned_sumsquares / (self.nobs - 1)

    @cached_property
    def std(self) -> np.float:
        """
        :return: sample standard deviation, square root of the sample variance
        """
        return np.sqrt(self.var)

    @cached_property
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

    def t_test(self, control: Union["LinearSample", "RatioSample"], equal_var: bool = False) -> TTestResult:
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

    def mann_whitney_u_test(self, control: Union["LinearSample", "RatioSample"],
                            use_continuity=True) -> MannWhitneyUTestResult:
        """
        :param control: control sample
        :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
        """
        try:
            res = mann_whitney_u_test(self.full, control.full, use_continuity)
        except Exception as e:
            if str(e) == 'All numbers are identical in mannwhitneyu':
                return MannWhitneyUTestResult(np.nan, np.nan, np.nan)
            raise e
        return res

    def shapiro_test(self) -> ShapiroTestResult:
        """
        Shapiro-Wilk test of normality
        :return: ShapiroTestResult(statistic, p_value)
        """
        try:
            res = shapiro(self.full)
        except Exception as e:
            if str(e) == 'Data must be at least length 3.':
                return ShapiroTestResult(np.nan, np.nan)
            raise e
        return ShapiroTestResult(p_value=res[1], statistic=res[0])

    def median_test(self, control: Union["LinearSample", "RatioSample"]) -> MedianTestResult:
        """
        :param control: control sample
        :return: MedianTestResult(statistic, p_value, grand_median)
        """
        try:
            res = median_test(self.full, control.full)
        except Exception as e:
            logger.debug(e)
            return MedianTestResult(np.nan, np.nan, np.nan)
        return MedianTestResult(p_value=res[1], statistic=res[0], grand_median=res[2])

    def levene_test(self, control: Union["LinearSample", "RatioSample"]) -> LeveneTestResult:
        """
        :param control: control sample
        :return: LeveneTestResult(statistic, p_value)
        """
        try:
            res = levene(self.full, control.full)
        except Exception as e:
            logger.debug(e)
            return LeveneTestResult(np.nan, np.nan)
        return LeveneTestResult(p_value=res[1], statistic=res[0])

    def mood_test(self, control: Union["LinearSample", "RatioSample"]) -> MoodTestResult:
        """
        :param control: control sample
        :return: MoodTestResult(statistic, p_value)
        """
        try:
            res = mood(self.full, control.full)
        except Exception as e:
            logger.debug(e)
            return MoodTestResult(np.nan, np.nan)
        return MoodTestResult(statistic=res[0], p_value=res[1])


class RatioSample(LinearSample):
    """
    Statistics and significance tests for ratio sample.
    """
    def __init__(
            self,
            num: Iterable[Union[int, float]],
            den: Iterable[Union[int, float]] = None,
            weights: Iterable[Union[int, float]] = None,
            linstrat: str = 'taylor',
            ltrim: float = None,
            rtrim: float = None,
    ):
        """
        :param num: numerator of observed data
        :param den: denominator of observed data
        :param weights: weights of observations. Length must be the same as obs's
        :param linstrat:
        lineariztion strategy. 'naive' when ratio computes element-wize. 'taylor' to use Taylor's expansion at mean.
        :param ltrim: proportion of data to cut from left tail
        :param rtrim: proportion of data to cut from right tail
        """
        num = LinearSample(num, weights)

        if den is not None:
            den = LinearSample(den, weights)
            is_ratio = True
            if linstrat == 'taylor':
                r = num.sum / den.sum
                s = (num.obs - r * den.obs) / den.mean + r
            elif linstrat == 'naive':
                s = num.obs / den.obs
            else:
                raise ValueError("linearizarion strategy must be 'taylor' or 'naive'")

            if ltrim or rtrim:
                if num.is_weighted:
                    ind, weights = argtrimw(s, weights, ltrim, rtrim)
                    s = s[ind]
                    num = LinearSample(num.obs[ind], weights)
                    den = LinearSample(den.obs[ind], weights)
                else:
                    ind = argtrim(s, ltrim, rtrim)
                    s = s[ind]
                    num = LinearSample(num.obs[ind])
                    den = LinearSample(den.obs[ind])
            super().__init__(s, weights)
        else:
            super().__init__(num.obs, weights, ltrim, rtrim)
            num = LinearSample(num.obs, weights, ltrim, rtrim)
            weights = self.weights if self.is_weighted else None
            den = LinearSample(np.ones(self.obs.shape[0]), weights)
            is_ratio = False

        self.num = num
        self.den = den
        self.is_ratio = is_ratio

    @cached_property
    def bootstrap_mean_dist(self) -> np.ndarray:
        """
        :return: bootstrap distribution of sample mean
        """
        return get_bootstrap_dist(self.num.full, self.den.full, 10000, 1, 1)

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

    def bootstrap_test(self, control: "RatioSample") -> BootstrapTestResult:
        """
        :param control: control sample
        :return: BootstrapTestResult(statistic, p_value, mean_diff, mean_diff_std)
        """
        dist = self.bootstrap_mean_dist - control.bootstrap_mean_dist
        p = min(
            (dist <= 0).mean(axis=0),
            (dist >= 0).mean(axis=0),
        ) * 2
        md = np.median(dist)
        mds = np.std(dist)
        s = md / mds
        return BootstrapTestResult(statistic=s, p_value=p, mean_diff=md, mean_diff_std=mds)
