import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Union


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


def argtrimw(a: Iterable, weights: Iterable, ltrim: float = None, rtrim: float = None) -> Tuple[np.ndarray, np.ndarray]:
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
            return [], []
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
            return [], []
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


def compress_sample(
        num: Iterable[Union[int, float]],
        den: Iterable[Union[int, float]] = None,
        weights: Iterable[Union[int, float]] = None,
) -> Tuple[np.array, np.array, np.array]:
    df = pd.DataFrame({'num': num})
    df['den'] = den if den else 1
    df['weights'] = weights if weights else 1
    grp = df.groupby(['num', 'den']).sum().reset_index(drop=False)
    return grp.num.values, grp.den.values, grp.weights.values
