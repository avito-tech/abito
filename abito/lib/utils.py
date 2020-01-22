import numpy as np


__all__ = ['is_array_sorted']


def is_array_sorted(ar):
    for i in range(ar.size-1):
        if ar[i + 1] < ar[i]:
            return False
    return True


def _quantile_is_valid(q):
    res = True
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                res = False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            res = False
    if not res:
        raise ValueError("Quantiles must be in the range [0, 1]")
    return True


def _return_or_inplace(obj, obs, weights, inplace):
    if inplace:
        obj.__init__(obs, weights)
    else:
        return obj.__class__(obs, weights)
