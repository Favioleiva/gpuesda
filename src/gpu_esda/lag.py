"""Observation-first spatial lag for vectors, matrices, and panels."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import BackendName, select_backend, to_backend, to_numpy
from .validation import validate_square_weights, validate_values


def spatial_lag(
    weights: Any, y: Any, backend: BackendName = "auto", *, return_native: bool = False
) -> Any:
    n = validate_square_weights(weights)
    arr = validate_values(y, n, allow_constant=True)
    original_shape = arr.shape
    matrix = arr.reshape(n, -1) if arr.ndim > 1 else arr
    info = select_backend(backend)
    w_native = to_backend(weights, info.name)
    y_native = to_backend(matrix, info.name)
    result = w_native @ y_native
    if arr.ndim > 1:
        result = result.reshape(original_shape)
    if return_native:
        return result
    result = to_numpy(result)
    return np.asarray(result)
