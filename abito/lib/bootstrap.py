import numpy as np
import multiprocessing
from abito.lib.stats.weighted import _quantile_sorted, _sort_obs


__all__ = ['generate_bootstrap_estimates']


def _do_bootstrap_plain(obs, stat_func, stat_args, n_iters, seed):
    if seed is not None:
        np.random.seed(seed)

    nobs = obs.shape[0]
    result = []
    for i in range(n_iters):
        new_ind = np.random.choice(nobs, nobs, replace=True)
        obs_new = obs[new_ind]
        result.append(stat_func(obs_new, **stat_args))
    return result


def _do_bootstrap_weighted(obs, weights, stat_func, stat_args, n_iters, seed):
    if seed is not None:
        np.random.seed(seed)

    if stat_func.__name__ == 'quantile':
        obs, weights = _sort_obs(obs, weights)
        stat_func = _quantile_sorted

    nobs = weights.sum()
    ps = weights / nobs

    result = []
    for i in range(n_iters):
        new_weights = np.random.multinomial(nobs, ps)
        weights = new_weights
        result.append(stat_func(weights=weights, obs=obs, **stat_args))
    return result


def _prepare_bootstrap_procedure(obs, weights, stat_func, n_iters, **stat_args):
    if weights.shape[0] == 0:
        func = _do_bootstrap_plain
        args = (obs, stat_func, stat_args, n_iters)
    else:
        func = _do_bootstrap_weighted
        args = (obs, weights, stat_func, stat_args, n_iters)
    return func, args


def generate_bootstrap_estimates(obs, stat_func, n_iters, weights=np.empty(0), n_threads=1, **stat_args):
    n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads
    if n_threads <= 1:
        func, args = _prepare_bootstrap_procedure(obs, weights, stat_func, n_iters, **stat_args)
        results = np.asarray(func(*args, 0))
    else:
        with multiprocessing.Pool(n_threads) as pool:
            n_iters_per_thread = int(n_iters / n_threads)
            pool_results = []
            seeds = np.random.randint(2**32, size=n_threads)
            for seed in seeds:
                func, args = _prepare_bootstrap_procedure(obs, weights, stat_func, n_iters_per_thread, **stat_args)
                r = pool.apply_async(func, (*args, seed))
                pool_results.append(r)
            results = []
            [results.extend(r.get()) for r in pool_results]
            results = np.asarray(results)
    return results
