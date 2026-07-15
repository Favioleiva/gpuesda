import numpy as np
import pytest


@pytest.fixture
def coordinates():
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)


@pytest.fixture
def values():
    return np.array([1.0, 2.5, 4.0, 7.0, 3.0])
