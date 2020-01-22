import numpy as np
from typing import Iterable, Tuple
from collections import namedtuple
import scipy.stats as stats
from abito.lib.stats.plain import *


__all__ = [
    't_test_from_stats',
    't_test',
    't_test_1samp',
    'mann_whitney_u_test_from_stats',
    'mann_whitney_u_test',
    'bootstrap_test',
    'shapiro_test',
    'median_test',
    'levene_test',
    'mood_test',
]


def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide='ignore', invalid='ignore'):
        df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not np.nan.
    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)
    return df, denom


def _equal_var_ttest_denom(v1, n1, v2, n2):
    df = n1 + n2 - 2.0
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    return df, denom


TTestResult = namedtuple('TTestResult', [
    'statistic',
    'p_value',
    'mean_diff',
    'mean_diff_std',
])
TTestResult.__new__.__defaults__ = (np.nan,) * len(TTestResult._fields)


def _t_test_from_stats(mean1, mean2, denom, df) -> TTestResult:
    d = mean1 - mean2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, denom)
    prob = stats.distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail

    return TTestResult(p_value=prob, statistic=t, mean_diff=d, mean_diff_std=denom)


def t_test_from_stats(mean1: float, std1: float, nobs1: int, mean2: float, std2: float, nobs2: float,
                      equal_var: bool = False) -> TTestResult:
    """
    :return: TTestResult(statistic, p_value, mean_diff, mean_diff_std)
    """
    if equal_var:
        df, denom = _equal_var_ttest_denom(std1**2, nobs1, std2**2, nobs2)
    else:
        df, denom = _unequal_var_ttest_denom(std1**2, nobs1,
                                             std2**2, nobs2)

    return _t_test_from_stats(mean1, mean2, denom, df)


def t_test(obs, obs_control, equal_var=False) -> TTestResult:
    mean1 = mean(obs)
    std1 = std(obs)
    nobs1 = obs.shape[0]
    mean2 = mean(obs_control)
    std2 = std(obs_control)
    nobs2 = obs_control.shape[0]
    return t_test_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var)


def t_test_1samp(obs, popmean: float) -> TTestResult:
    """
    :param popmean: population mean
    :return: TTestResult(statistic, p_value, mean_diff, mean_diff_std)
    """
    return _t_test_from_stats(mean(obs), popmean, mean_std(obs), obs.shape[0] - 1)


def _mann_whitney_u_statistic(obs, obs_control, use_continuity: bool = True) -> Tuple:

    obs = np.asarray(obs)
    obs_control = np.asarray(obs_control)
    n1 = len(obs)
    n2 = len(obs_control)
    ranked = stats.rankdata(np.concatenate((obs, obs_control)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x

    t = stats.tiecorrect(ranked)
    std = np.sqrt(t * n1 * n2 * (n1 + n2 + 1) / 12.0)

    mean = n1 * n2 / 2.0 + 0.5 * use_continuity

    return u1, mean, std


MannWhitneyUTestResult = namedtuple('MannWhitneyUTestResult', [
    'u_statistic',
    'z_statistic',
    'p_value',
])
MannWhitneyUTestResult.__new__.__defaults__ = (np.nan,) * len(MannWhitneyUTestResult._fields)


def mann_whitney_u_test_from_stats(u1: float, mean: float, std: float) -> MannWhitneyUTestResult:
    """
    :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
    """
    z = np.divide(-(u1 - mean), std)
    p = 2 * stats.distributions.norm.sf(abs(z))
    return MannWhitneyUTestResult(p_value=p, u_statistic=u1, z_statistic=z)


def mann_whitney_u_test(obs: Iterable, obs_control: Iterable, use_continuity: bool = True) -> MannWhitneyUTestResult:
    """
    :param use_continuity: use continuity correction
    :return: MannWhitneyUTestResult(u_statistic, z_statistic, p_value)
    """
    stats = _mann_whitney_u_statistic(obs, obs_control, use_continuity)
    return mann_whitney_u_test_from_stats(*stats)


BootstrapTestResult = namedtuple('BootstrapTestResult', [
    'estimates_diff_std',

    'est_p_value',
    'est_t_statistic',

    'median_est_diff',
    'median_est_t_statistic',
    'median_est_t_p_value',

    'stat_diff',
    't_statistic',
    't_p_value',
])
BootstrapTestResult.__new__.__defaults__ = (np.nan,) * len(BootstrapTestResult._fields)


def bootstrap_test(
        stat_val,
        bootstrap_estimates,
        nobs,
        stat_val_control,
        bootstrap_estimates_control,
        nobs_control
) -> BootstrapTestResult:
    """
    :param stat_val: sample value of statistic in treatment group
    :param bootstrap_estimates: bootstrap estimates (10000 or so) of statistic
    :param nobs: number of observations in initial sample (needed only for degrees of freedom for t-distribution)
    :param stat_val_control: sample value of statistic in control group
    :param bootstrap_estimates_control: same as above
    :param nobs_control: same as above
    :return: BootstrapTestResult(estimates_diff_std, est_p_value, est_t_statistic, median_est_diff,
                                 median_est_t_statistic, median_est_t_p_value, stat_diff, t_statistic, t_p_value)
    """
    estimates_diff = bootstrap_estimates - bootstrap_estimates_control

    median_est_diff = np.median(estimates_diff, axis=0)
    estimates_diff_std = np.std(estimates_diff, axis=0)

    est_p_value = np.min([
        (estimates_diff <= 0).mean(axis=0),
        (estimates_diff > 0).mean(axis=0)
    ], axis=0) * 2

    df = nobs + nobs_control - 2
    est_t_statistic = stats.distributions.t.isf(est_p_value / 2, df=df) * np.sign(median_est_diff)

    median_est_t_statistic = np.divide(median_est_diff, estimates_diff_std)
    median_est_t_p_value = stats.distributions.t.sf(np.abs(median_est_t_statistic), df) * 2

    stat_diff = stat_val - stat_val_control

    t_statistic = np.divide(stat_diff, estimates_diff_std)
    t_p_value = stats.distributions.t.sf(np.abs(t_statistic), df) * 2  # use np.abs to get upper tail

    return BootstrapTestResult(estimates_diff_std, est_p_value, est_t_statistic, median_est_diff,
                               median_est_t_statistic, median_est_t_p_value, stat_diff, t_statistic, t_p_value)


ShapiroTestResult = namedtuple('ShapiroTestResult', [
    'statistic',
    'p_value',
])
ShapiroTestResult.__new__.__defaults__ = (np.nan,) * len(ShapiroTestResult._fields)


def shapiro_test(obs) -> ShapiroTestResult:
    """
    Shapiro-Wilk test of normality
    :return: ShapiroTestResult(statistic, p_value)
    """
    if obs.size < 3:
        return ShapiroTestResult(np.nan, np.nan)
    res = stats.shapiro(obs)
    return ShapiroTestResult(p_value=res[1], statistic=res[0])


MedianTestResult = namedtuple('MedianTestResult', [
    'statistic',
    'p_value',
    'grand_median',
])
MedianTestResult.__new__.__defaults__ = (np.nan,) * len(MedianTestResult._fields)


def median_test(obs, obs_control) -> MedianTestResult:
    """
    :return: MedianTestResult(statistic, p_value, grand_median)
    """
    if std(obs) == 0:
        return MedianTestResult(np.nan, np.nan, np.nan)
    res = stats.median_test(obs, obs_control)
    return MedianTestResult(p_value=res[1], statistic=res[0], grand_median=res[2])


LeveneTestResult = namedtuple('LeveneTestResult', [
    'statistic',
    'p_value',
])
LeveneTestResult.__new__.__defaults__ = (np.nan,) * len(LeveneTestResult._fields)


def levene_test(obs, obs_control) -> LeveneTestResult:
    """
    :return: LeveneTestResult(statistic, p_value)
    """
    res = stats.levene(obs, obs_control)
    return LeveneTestResult(p_value=res[1], statistic=res[0])


MoodTestResult = namedtuple('MoodTestResult', [
    'statistic',
    'p_value',
])
MoodTestResult.__new__.__defaults__ = (np.nan,) * len(MoodTestResult._fields)


def mood_test(obs, obs_control) -> MoodTestResult:
    """
    :return: MoodTestResult(statistic, p_value)
    """
    res = stats.mood(obs, obs_control)
    return MoodTestResult(statistic=res[0], p_value=res[1])