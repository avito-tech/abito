from typing import Union, Iterable, Optional, NamedTuple
import numpy as np
from numpy.lib import recfunctions as rfn

from abito.lib.stats import weighted, plain
from abito.lib.significance import *
from abito.lib.significance import _t_test_from_stats
from abito.lib.bootstrap import generate_bootstrap_estimates
from abito.lib.prepare.trim import argtrim
from abito.lib.prepare.compress import reweigh, compress, _validate_compress_stat
from abito.lib.utils import _return_or_inplace


__all__ = ['sample', 'Sample', 'LinearSample', 'RatioSample']


class SampleBase:
    """
    Base class with shared methods
    """
    def __init__(self, obs: np.ndarray, weights: np.ndarray = np.empty(0, dtype='int')):
        """
        Parameters
        ----------
        obs : ndarray
            If RatioSample is being initialized, `obs` should be structured array with field names ('num', 'den')
        weights : ndarray, optional
            Weights. If not specified `weights` assigned to empty ndarray
        """
        self.obs = obs
        self.weights = weights

    def copy(self):
        """
        Return copy of the object.

        Returns
        -------
        LinearSample or RatioSample
        """
        return self.__class__(self.obs.copy(), self.weights.copy())

    @property
    def is_weighted(self) -> bool:
        """
        Is Sample has weights.
        Returns
        -------
        bool
        """
        return self.weights.shape[0] > 0

    @property
    def fullobs(self) -> np.ndarray:
        """
        Full observations vector after weights applied

        Returns
        -------
        out : ndarray

        """
        if self.is_weighted:
            return np.repeat(self.obs, self.weights)
        else:
            return self.obs

    @property
    def _stat_funcs_module(self):
        if self.is_weighted:
            return weighted
        else:
            return plain

    @property
    def nobs(self) -> np.int:
        """
        Number of observations in sample.

        Returns
        -------
        out : int

        """
        if self.is_weighted:
            return self.weights.sum()
        else:
            return self.obs.shape[0]

    def reweigh(self, inplace=False):
        """
        Make sample weighted. Initialize weights by grouping unique values of observations.
        Reinitialise if sample already has weights (may be used in case of duplicates).

        Parameters
        ----------
        inplace : bool, optional
            If ``True``, reinitialize obs / weights of the instance, default is ``False``.

        Returns
        -------
        out : LinearSample or RatioSample
            If `inplace` is ``True``, returns None.
        """
        obs, weights = reweigh(self.obs, self.weights)
        return _return_or_inplace(self, obs, weights, inplace)

    @property
    def cache(self):
        """
        Dict property that stores results of bootstrap_estimates.

        Returns
        -------
        out : dict of ndarrays
        """
        if not hasattr(self, '_cache'):
            self.cache = {}
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @cache.deleter
    def cache(self):
        self._cache = {}

    @staticmethod
    def _build_cache_key(**key_args):
        key_args = {k: tuple(v) if isinstance(v, Iterable) else v for k, v in key_args.items()}
        return tuple(sorted(key_args.items()))

    def _store_in_cache(self, value, **key_args):
        key = self._build_cache_key(**key_args)
        self.cache[key] = value

    def _get_from_cache(self, **key_args):
        key = self._build_cache_key(**key_args)
        return self.cache.get(key, None)

    def _del_from_cache(self, **key_args):
        key = self._build_cache_key(**key_args)
        if key in self.cache:
            del self.cache[key]

    def stat(self, stat: str, **stat_args) -> Union[np.float, np.int, np.ndarray]:
        """
        Calculate statistic using name and parameters.

        Parameters
        ----------
        stat : str
            {'sum', 'mean', 'var', 'std', 'mean_std', 'quantile', 'median'} for LinearSample, 'ratio' for RatioSample
        stat_args : dict
            Additional argumants that needed by corresponding method. At this time only for quantile.

        Returns
        -------
        out : number or array_like
            array_like is returned only in case of quantile and iterable `q`

        """
        stat_func = getattr(self._stat_funcs_module, stat)
        if self.is_weighted:
            return stat_func(self.obs, self.weights, **stat_args)
        else:
            return stat_func(self.obs, **stat_args)

    def bootstrap_estimates(self, stat, n_iters, n_threads=1, cache_result=False, **stat_args) -> np.ndarray:
        """

        Parameters
        ----------
        stat : str
            Name of statistics.
        n_iters : int
            Number of bootstrap iterations, size of output result.
        n_threads : int
            Number of processes to use in parallel calculation.
        cache_result : bool
            Cache results to save time in future analysis.
        stat_args : dict
            Additional arguments to pass in stat func.

        Returns
        -------
        out : array
            Array of bootstrap estimates
        """
        from_cache = self._get_from_cache(stat=stat, n_iters=n_iters, **stat_args)
        if from_cache is not None:
            return from_cache
        stat_func = getattr(self._stat_funcs_module, stat)
        res = generate_bootstrap_estimates(self.obs, stat_func, n_iters, self.weights, n_threads, **stat_args)
        if cache_result:
            self._store_in_cache(res, stat=stat, n_iters=n_iters, **stat_args)
        return res

    def bootstrap_test(
            self,
            control: 'SampleBase',
            stat: str,
            n_iters: int,
            n_threads: int = 1,
            cache_result: bool = False,
            **stat_args
    ) -> NamedTuple:
        """
        Significance test based on bootstrap estimates.

        Parameters
        ----------
        control : LinearSample or RatioSample
            Control sample.
        stat : str
            Name of statistics.
        n_iters : int
            Number of bootstrap iterations, size of output result.
        n_threads : int
            Number of processes to use in parallel calculation.
        cache_result : bool
            Cache results to save time in future analysis.
        stat_args : dict
            Additional arguments to pass in stat func.

        Returns
        -------
            estimates_diff_std : float
                Standard deviation of bootstrap estimates.
            est_p_value : float
                P-value calculated as share of estimates above / below zero.
            est_t_statistic : float
                t-statistic derived from est_p_value using inverted survival function of t-distribution.
            median_est_diff : float
                Difference between estimates' median.
            median_est_t_statistic : float
                Equal to median_est_diff / estimates_diff_std
            median_est_t_p_value : float
                P-value derived from median_est_t_statistic using survival function of t-distribution.
            stat_diff : float
                Difference between statistics if treatment and control.
            t_statistic : float
                Equal to stat_diff / estimates_diff_std.
            t_p_value : float
                P-value derived from t_statistic using survival function of t-distribution

        """
        t_stat = self.stat(stat, **stat_args)
        c_stat = control.stat(stat, **stat_args)
        t_bootstrap_estimates = self.bootstrap_estimates(stat, n_iters, n_threads, cache_result, **stat_args)
        c_bootstrap_estimates = control.bootstrap_estimates(stat, n_iters, n_threads, cache_result, **stat_args)
        return bootstrap_test(t_stat, t_bootstrap_estimates, self.nobs, c_stat, c_bootstrap_estimates, control.nobs)


class LinearSample(SampleBase):

    def compress(
            self,
            n_buckets: int,
            stat: str = 'mean',
            weights_dist: str = 'uniform',
            reweigh: bool = False,
            inplace=False,
    ):
        """

        Parameters
        ----------
        n_buckets
        stat
        weights_dist
        reweigh
        inplace

        Returns
        -------

        """
        _validate_compress_stat(stat)
        obs = self.fullobs
        obs.sort()
        compressed, weights = compress(obs, n_buckets, stat, weights_dist, reweigh)
        return _return_or_inplace(self, compressed, weights, inplace)

    def trim(self, ltrim=0, rtrim=0, inplace=False) -> Optional['LinearSample']:
        """

        Parameters
        ----------
        ltrim
        rtrim
        inplace

        Returns
        -------

        """

        trim_ind, weights = argtrim(self.obs, self.weights, ltrim=ltrim, rtrim=rtrim)

        trimmed = self.obs[trim_ind]
        samp = LinearSample(trimmed, weights)

        return _return_or_inplace(self, samp.obs, samp.weights, inplace)

    def sum(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('sum')

    def mean(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('mean')

    def var(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('var')

    def std(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('std')

    def mean_std(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('mean_std')

    def quantile(self, q) -> np.float:
        """

        Parameters
        ----------
        q

        Returns
        -------

        """
        return self.stat('quantile', q=q)

    def median(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('median')

    def t_test_1samp(self, popmean: float) -> NamedTuple:
        """

        Parameters
        ----------
        popmean

        Returns
        -------

        """
        return _t_test_from_stats(self.mean(), popmean, self.mean_std(), self.nobs - 1)

    def t_test(self, control: 'LinearSample', equal_var: bool = False) -> NamedTuple:
        """

        Parameters
        ----------
        control
        equal_var

        Returns
        -------

        """
        return t_test_from_stats(self.mean(), self.std(), self.nobs,
                                 control.mean(), control.std(), control.nobs,
                                 equal_var=equal_var)

    def mann_whitney_u_test(self, control: 'LinearSample',
                            use_continuity=True) -> NamedTuple:
        """

        Parameters
        ----------
        control
        use_continuity

        Returns
        -------

        """
        return mann_whitney_u_test(self.fullobs, control.fullobs, use_continuity)  # TODO: support weighted samples

    def shapiro_test(self) -> NamedTuple:
        """

        Returns
        -------

        """
        return shapiro_test(self.fullobs)

    def median_test(self, control: 'LinearSample') -> NamedTuple:
        """

        Parameters
        ----------
        control

        Returns
        -------

        """
        return median_test(self.fullobs, control.fullobs)

    def levene_test(self, control: 'LinearSample') -> NamedTuple:
        """

        Parameters
        ----------
        control

        Returns
        -------

        """
        return levene_test(self.fullobs, control.fullobs)

    def mood_test(self, control: 'LinearSample') -> NamedTuple:
        """

        Parameters
        ----------
        control

        Returns
        -------

        """
        return mood_test(self.fullobs, control.fullobs)


class RatioSample(SampleBase):
    """
    RatioSample
    """

    _obs_dtype = np.dtype({
        'names': ['num', 'den'],
        'formats': ['float', 'float'],
    })

    @classmethod
    def from_numden(
            cls,
            num: np.ndarray,
            den: np.ndarray,
            weights: np.ndarray = None,
    ) -> 'RatioSample':
        """

        Parameters
        ----------
        num
        den
        weights

        Returns
        -------

        """
        obs = np.rec.fromarrays([num, den], dtype=cls._obs_dtype)
        return cls(obs, weights)

    @property
    def num(self):
        """

        Returns
        -------

        """
        return self.obs['num']

    @property
    def den(self):
        """

        Returns
        -------

        """
        return self.obs['den']

    @property
    def numsamp(self):
        """

        Returns
        -------

        """
        return LinearSample(self.num, self.weights)
    
    @property
    def densamp(self):
        """

        Returns
        -------

        """
        return LinearSample(self.den, self.weights)

    def linearize(self, strategy: str = 'taylor'):
        """
        Lineariztion strategy.

        Parameters
        ----------
        strategy : str, {'taylor', 'naive'}
            When 'naive' ratio is computed element-wize. 'taylor' to use Taylor's expansion at mean point.
        Returns
        -------
        out : LinearSample
        """
        if strategy == 'taylor':
            r = self.numsamp.sum() / self.densamp.sum()
            obs = (self.num - r * self.den) / self.densamp.mean() + r
        elif strategy == 'naive':
            obs = self.num / self.den
        else:
            raise ValueError("linearizarion strategy must be 'taylor' or 'naive'")
        return LinearSample(obs, self.weights)

    def compress(
            self,
            n_buckets: int,
            sort_by: str = 'den',
            stat: str = 'mean',
            weights_dist: str = 'uniform',
            reweigh: bool = False,
            inplace=False
    ):
        """
        Compress sample into ntile-buckets

        Parameters
        ----------
        n_buckets : int
            Number of buckets / unique values in the new sample.
        sort_by : str, {'num', 'den', 'taylor', 'naive'}
        stat
        weights_dist
        reweigh
        inplace

        Returns
        -------

        """
        _validate_compress_stat(stat)

        fullobs = self.fullobs
        if sort_by == 'num':
            sort_ind = np.argsort(fullobs, order=('num', 'den'))
        elif sort_by == 'den':
            sort_ind = np.argsort(fullobs, order=('den', 'num'))
        else:
            obs_to_sort = self.linearize(strategy=sort_by).fullobs
            sort_ind = np.argsort(obs_to_sort)

        fullobs = rfn.structured_to_unstructured(fullobs[sort_ind])
        compressed, weights = compress(fullobs, n_buckets, stat, weights_dist, reweigh)

        obs = rfn.unstructured_to_structured(compressed, dtype=self._obs_dtype)
        return _return_or_inplace(self, obs, weights, inplace)

    def trim(self, ltrim=0, rtrim=0, sort_by: str = 'taylor', inplace=False):
        """

        Parameters
        ----------
        ltrim
        rtrim
        sort_by
        inplace

        Returns
        -------

        """
        if sort_by == 'num':
            trim_ind, weights = argtrim(self.obs, self.weights, ltrim=ltrim, rtrim=rtrim, order=('num', 'den'))
        elif sort_by == 'den':
            trim_ind, weights = argtrim(self.obs, self.weights, ltrim=ltrim, rtrim=rtrim, order=('den', 'num'))
        else:
            obs_to_trim = self.linearize(strategy=sort_by).obs
            trim_ind, weights = argtrim(obs_to_trim, self.weights, ltrim=ltrim, rtrim=rtrim)

        trimmed = self.obs[trim_ind]
        samp = RatioSample(trimmed, weights)

        return _return_or_inplace(self, samp.obs, samp.weights, inplace)

    def ratio(self) -> np.float:
        """

        Returns
        -------

        """
        return self.stat('ratio')


def sample(
        obs: Iterable[Union[int, float]],
        den: Iterable[Union[int, float]] = None,
        weights: Iterable[Union[int, float]] = None,
):
    """
    Create a sample — instance of LinearSample or RatioSample.
    The most convenient way to calculate statistics of sample observations.

    `obs` is 1D-observations of some random variable. If `den` is specified, returns 'RatioSample'.
    `weights` are the counts of each observation. All three parameters must be of same shape.

    Parameters
    ----------
    obs : array_like
        Observations or Numerator.
    den : array_like or None
        Denominator. Same shape as obs.
    weights : array_like or None
        Weights. Same shape as obs.
    Returns
    -------
    LinearSample or RatioSample
    """
    obs = np.asanyarray(obs)
    if weights is not None:
        weights = np.asanyarray(weights, dtype='int')
    else:
        weights = np.empty(0, dtype='int')
    if den is None:
        return LinearSample(obs, weights)
    else:
        den = np.asanyarray(den)
        return RatioSample.from_numden(obs, den, weights)


Sample = sample
