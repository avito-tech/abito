import pytest
import numpy as np
from abito.lib.significance import *


def test_t_test():
    np.random.seed(0)
    treatment = np.random.normal(100, size=100)
    control = np.random.normal(100, size=100)
    r = t_test(treatment, control)
    assert r.p_value == pytest.approx(0.9, 0.1)

    r = t_test_1samp(treatment, 100)
    assert r.p_value == pytest.approx(0.6, 0.1)
