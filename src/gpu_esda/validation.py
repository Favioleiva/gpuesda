"""Input validation shared across numerical modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import to_numpy


def validate_values(y: Any, n: int | None = None, *, allow_constant: bool = False) -> np.ndarray:
    arr = np.asarray(to_numpy(y))
    if arr.ndim < 1 or arr.ndim > 3:
        raise ValueError("values must have shape (n,), (n, columns), or (n, variables, periods)")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"values have {arr.shape[0]} observations but weights have {n}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("values must be numeric")
    if not np.isfinite(arr).all():
        raise ValueError(
            "values contain NaN or infinity; missing values are never converted to zero"
        )
    flat = arr.reshape(arr.shape[0], -1)
    if not allow_constant and np.any(np.ptp(flat, axis=0) == 0):
        raise ValueError("constant or zero-variance variables are not valid for Moran statistics")
    return arr


def validate_square_weights(weights: Any) -> int:
    if (
        not hasattr(weights, "shape")
        or len(weights.shape) != 2
        or weights.shape[0] != weights.shape[1]
    ):
        raise ValueError("weights must be a square two-dimensional matrix")
    return int(weights.shape[0])
