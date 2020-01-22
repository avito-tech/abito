from typing import Tuple, Union
import numpy as np
from abito.lib.utils import _quantile_is_valid


def _argquantile_weighted(
        weights_sorted: np.ndarray,
        q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    inds = []
    cuts = []
    q = np.asanyarray(q)
    weight_qs = [int(q * weights_sorted.sum()) for q in q.flatten()]
    cur_weight_q = 0
    weight_q = weight_qs.pop(0)
    for cur_weight_ind, cur_weight in enumerate(weights_sorted):
        cur_weight_q += cur_weight
        while weight_q <= cur_weight_q:
            inds.append(cur_weight_ind)
            cuts.append(cur_weight - (cur_weight_q - weight_q))
            if weight_qs:
                weight_q = weight_qs.pop(0)
            else:
                weight_q = -1
                break
        if weight_q < 0:
            break
    inds = np.asanyarray(inds).reshape(q.shape)
    cuts = np.asanyarray(cuts).reshape(q.shape)
    return inds, cuts


def _argtrim_plain(
        ar: np.ndarray,
        q: np.ndarray,
        **kwargs
):
    trim_inds = (q * ar.shape[0]).astype('int')
    ind = np.argpartition(ar, trim_inds, **kwargs)
    return ind[trim_inds[0]:trim_inds[1]]


def _argtrim_weighted(
        ar: np.ndarray,
        weights: np.ndarray,
        q: np.ndarray,
        **kwargs
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:

    weights = weights.copy()

    ind_sorted = ar.argsort(**kwargs)
    weights = weights[ind_sorted]

    (lowercut_a, uppercut_a), (lowercut_w, uppercut_w) = _argquantile_weighted(weights, q)

    weights[uppercut_a] = uppercut_w
    weights[lowercut_a] -= lowercut_w

    ind_sorted = ind_sorted[lowercut_a:uppercut_a + 1]
    weights = weights[lowercut_a:uppercut_a + 1]
    return ind_sorted, weights


def argtrim(
        ar: np.ndarray,
        weights: np.ndarray = np.empty(0),
        ltrim: float = 0,
        rtrim: float = 0,
        **kwargs
):
    q = np.array([ltrim, 1 - rtrim])
    _quantile_is_valid(q)
    if weights.shape[0] == 0:
        ind = _argtrim_plain(ar, q, **kwargs)
    else:
        ind, weights = _argtrim_weighted(ar, weights, q, **kwargs)
    return ind, weights
