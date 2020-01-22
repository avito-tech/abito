import numpy as np
from typing import Union


__all__ = ['sum', 'mean', 'var', 'std', 'mean_std', 'quantile', 'median', 'ratio']


def sum(obs: np.ndarray) -> np.float:
    return obs.sum(axis=0)


def mean(obs: np.ndarray) -> np.float:
    return np.divide(obs.sum(axis=0), obs.shape[0])


def demeaned(obs: np.ndarray) -> np.ndarray:
    return obs - mean(obs)


def demeaned_sumsquares(obs: np.ndarray) -> np.float:
    return (demeaned(obs) ** 2).sum(axis=0)


def var(obs: np.ndarray) -> np.float:
    return demeaned_sumsquares(obs) / (obs.shape[0] - 1)


def std(obs: np.ndarray) -> np.float:
    return np.sqrt(var(obs))


def mean_std(obs: np.ndarray) -> np.float:
    return std(obs) / np.sqrt(obs.shape[0])


def quantile(obs: np.ndarray, q: float) -> Union[np.ndarray, np.float]:
    return np.quantile(obs, q, axis=0)


def median(obs: np.ndarray) -> np.float:
    return quantile(obs, 0.5)


def ratio(obs: np.ndarray) -> np.float:
    return sum(obs['num']) / sum(obs['den'])
