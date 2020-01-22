import numpy as np
from abito.lib.stats import plain


def _array_binary_view(ar: np.ndarray):
    ar = np.ascontiguousarray(ar)
    ar = ar.view('|S%d' % (ar.itemsize * ar[0].size))
    return ar


def _reweigh_plain_general(ar: np.ndarray):
    ar_binary = _array_binary_view(ar)
    _, ind, counts = np.unique(ar_binary, return_index=True, return_counts=True)
    return ar[ind], counts


def _reweigh_plain_int_fast(ar: np.ndarray):
    try:
        y = np.bincount(ar)
    except (ValueError, TypeError):
        ar_int = ar.astype('int', casting='unsafe')
        assert (ar == ar_int).all()
        y = np.bincount(ar_int)
    ii = np.nonzero(y)[0]
    u, counts = ii, y[ii]
    return u, counts


def _reweigh_plain(ar: np.ndarray):
    try:
        return _reweigh_plain_int_fast(ar)
    except (ValueError, TypeError, AssertionError):
        return _reweigh_plain_general(ar)


def _reweigh_weighted(ar: np.ndarray, weights: np.ndarray):
    ar_binary = _array_binary_view(ar)
    if ar_binary.ndim > 1:
        ar_binary = ar_binary.flatten()
    unique_values, ind = np.unique(ar_binary, return_index=True)
    new_weights = np.array([weights[ar_binary == e].sum() for e in unique_values])
    return ar[ind], new_weights


def reweigh(ar: np.ndarray, weights: np.ndarray = np.empty(0)):
    if weights.shape[0] == 0:
        compressed, weights = _reweigh_plain(ar)
    else:
        compressed, weights = _reweigh_weighted(ar, weights)
    return compressed, weights


def _calc_bucket_weights(nobs: int, n_buckets: int, dist: str = 'uniform') -> np.ndarray:
    if dist == 'uniform':
        nobs_per_bucket = nobs // n_buckets
        nobs_left = nobs % n_buckets
        leftovers = np.array((n_buckets - nobs_left) * [0] + nobs_left * [1])
        np.random.shuffle(leftovers)
        bucket_weights = np.array([nobs_per_bucket] * n_buckets) + leftovers
    elif dist == 'multinomial':
        bucket_weights = np.random.multinomial(nobs, [1 / n_buckets] * n_buckets)
    else:
        raise ValueError('dist must be uniform or multinomial')
    return bucket_weights.astype('int')


def _split_into_buckets(
        ar: np.ndarray,
        n_buckets: int,
        weights_dist: str = 'uniform',
):
    nobs = ar.shape[0]
    bucket_weights = _calc_bucket_weights(nobs, n_buckets, weights_dist)
    bucket_split_indices = bucket_weights.cumsum()[:-1]

    return np.split(ar, bucket_split_indices), bucket_weights


def _validate_compress_stat(stat):
    if stat not in ('mean', 'median'):
        raise ValueError('Statistic for compression can only be mean or median')


def compress(
        ar: np.ndarray,
        n_buckets: int,
        stat,
        weights_dist: str = 'uniform',
        reweigh: bool = False,
):
    split, weights = _split_into_buckets(ar, n_buckets, weights_dist)

    stat_func = getattr(plain, stat)
    compressed = np.array([stat_func(s) for s in split])

    if reweigh:
        return _reweigh_weighted(compressed, weights)
    else:
        return compressed, weights
