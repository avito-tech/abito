import numpy as np
from typing import Iterable, Tuple, Union


def argtrim(a: Iterable, ltrim: float = None, rtrim: float = None) -> np.ndarray:
    a = np.asarray(a)

    if not rtrim and not ltrim:
        return np.arange(a.shape[0])

    nobs = a.shape[0]

    lowercut = 0
    if ltrim is not None:
        if ltrim >= 1:
            return np.asarray([])
        lowercut = int(ltrim * nobs)

    uppercut = nobs
    if rtrim is not None:
        if rtrim >= 1:
            return np.asarray([])
        uppercut = nobs - int(rtrim * nobs)

    args = np.argpartition(a, (lowercut, uppercut - 1))
    return args[lowercut:uppercut]


def argtrimw(
        a: Iterable,
        weights: Iterable = None,
        ltrim: float = None,
        rtrim: float = None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:

    a = np.asarray(a)

    if weights is None:
        return argtrim(a, ltrim, rtrim), None

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
            return np.asarray([]), np.asarray([])
        lowercut_w = int(ltrim * nobs)
        while lowercut_w > 0:
            w = weights_sorted[lowercut_a]
            if w <= lowercut_w:
                weights_sorted[lowercut_a] = 0
                lowercut_a += 1
            else:
                weights_sorted[lowercut_a] = w - lowercut_w
            lowercut_w -= w

    uppercut_a = a.shape[0]
    if rtrim is not None:
        if rtrim >= 1:
            return np.asarray([]), np.asarray([])
        uppercut_w = int(rtrim * nobs)
        while uppercut_w > 0:
            w = weights_sorted[uppercut_a - 1]
            if w <= uppercut_w:
                weights_sorted[uppercut_a - 1] = 0
                uppercut_a -= 1
            else:
                weights_sorted[uppercut_a - 1] = w - uppercut_w
            uppercut_w -= w

    ind_sorted = ind_sorted[lowercut_a:uppercut_a]
    weights_sorted = weights_sorted[lowercut_a:uppercut_a]
    return ind_sorted, weights_sorted


def compress_1d(ar: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    ar = np.asarray(ar).flatten()
    try:
        y = np.bincount(ar)
        ii = np.nonzero(y)[0]
        u, counts = ii, y[ii]
    except (ValueError, TypeError):
        u, counts = np.unique(ar, return_counts=True)
    return u, counts


def compress_2d(ar2d: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    ar2d = np.asarray(ar2d)
    if ar2d.ndim > 2:
        raise ValueError('ar2d must be 2-dimensional')
    elif ar2d.ndim == 1 or min(ar2d.shape) == 1:
        return compress_1d(ar2d)
    ar = np.ascontiguousarray(ar2d)
    ar = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
    _, ind, counts = np.unique(ar, return_index=True, return_counts=True)
    return ar2d[ind], counts
