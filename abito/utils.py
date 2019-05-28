import numpy as np
from typing import Iterable


def argtrim(a: Iterable, ltrim: float = None, rtrim: float = None) -> np.ndarray:
    a = np.asarray(a)

    if not rtrim and not ltrim:
        return np.arange(a.shape[0])

    nobs = a.shape[0]

    lowercut = 0
    if ltrim is not None:
        if ltrim >= 1:
            return []
        lowercut = int(ltrim * nobs)

    uppercut = nobs
    if rtrim is not None:
        if rtrim >= 1:
            return []
        uppercut = nobs - int(rtrim * nobs)

    args = np.argpartition(a, (lowercut, uppercut - 1))
    return args[lowercut:uppercut]


def argtrimw(a: Iterable, weights: Iterable, ltrim: float = None, rtrim: float = None) -> np.ndarray:
    a = np.asarray(a)
    weights = np.asarray(weights)

    assert a.shape == weights.shape

    if not rtrim and not ltrim:
        return np.arange(a.shape[0]), weights

    nobs = weights.sum()

    ind_sorted = a.argsort()
    weights_sorted = weights[ind_sorted]

    lowercut_a = 0
    if ltrim is not None:
        if ltrim >= 1:
            return []
        lowercut_w = int(ltrim * nobs)
        l = 0
        while lowercut_w > 0:
            w = weights_sorted[l]
            if w <= lowercut_w:
                lowercut_a += 1
            else:
                weights_sorted[l] = w - lowercut_w
            lowercut_w -= w

    uppercut_a = a.shape[0]
    if rtrim is not None:
        if rtrim >= 1:
            return []
        uppercut_w = int(rtrim * nobs)
        u = uppercut_a - 1
        while uppercut_w > 0:
            w = weights_sorted[u]
            if w <= uppercut_w:
                uppercut_a -= 1
            else:
                weights_sorted[u] = w - uppercut_w
            uppercut_w -= w

    ind_sorted = ind_sorted[lowercut_a:uppercut_a]
    weights_sorted = weights_sorted[lowercut_a:uppercut_a]
    return ind_sorted, weights_sorted
