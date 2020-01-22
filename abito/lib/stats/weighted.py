import numpy as np
from typing import Union
from abito.lib.utils import is_array_sorted, _quantile_is_valid
from abito.lib.prepare.trim import _argquantile_weighted


__all__ = ['sum', 'mean', 'var', 'std', 'mean_std', 'quantile', 'median', 'ratio']


def sum(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return np.dot(obs.T, weights).sum()


def mean(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return np.divide(sum(obs, weights), weights.sum())


def demeaned(obs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return obs - mean(obs, weights)


def demeaned_sumsquares(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return np.dot((demeaned(obs, weights) ** 2).T, weights).sum()


def var(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return demeaned_sumsquares(obs, weights) / (weights.sum() - 1)


def std(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return np.sqrt(var(obs, weights))


def mean_std(obs: np.ndarray, weights: np.ndarray):
    return std(obs, weights) / np.sqrt(weights.sum())


def ratio(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return sum(obs['num'], weights) / sum(obs['den'], weights)


def _quantile_sorted(obs: np.ndarray, weights: np.ndarray, q) -> Union[np.ndarray, np.float]:
    indices, _ = _argquantile_weighted(weights, q)
    return obs[indices]


def _sort_obs(obs: np.ndarray, weights: np.ndarray):
    if not is_array_sorted(obs):
        ind_sorted = np.argsort(obs)
        obs = obs[ind_sorted]
        weights = weights[ind_sorted]
    return obs, weights


def quantile(obs: np.ndarray, weights: np.ndarray, q) -> Union[np.ndarray, np.float]:
    obs = np.asarray(obs)
    weights = np.asarray(weights)
    q = np.asanyarray(q)
    _quantile_is_valid(q)
    obs, weights = _sort_obs(obs, weights)
    return _quantile_sorted(obs, weights, q)


def median(obs: np.ndarray, weights: np.ndarray) -> np.float:
    return quantile(obs, weights, 0.5)


