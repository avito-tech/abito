import pytest
import numpy as np
from abito.lib.utils import _quantile_is_valid


def test_quantile_is_valid():
    with pytest.raises(ValueError):
        _quantile_is_valid(np.array([-0.1]))

    with pytest.raises(ValueError):
        _quantile_is_valid(np.linspace(1, 1.01, 1000))
