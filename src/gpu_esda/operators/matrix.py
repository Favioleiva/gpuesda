"""Adapter exposing existing dense/CSR weights as a spatial operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import BackendName, _cupy, select_backend, to_backend
from ..validation import validate_square_weights
from .base import SpatialOperator


class MatrixWeightsOperator(SpatialOperator):
    """Wrap an explicit square matrix without changing vector-MVP behavior."""

    def __init__(self, weights: Any, backend: BackendName = "auto") -> None:
        n = validate_square_weights(weights)
        self.backend = select_backend(backend).name
        self._xp = np if self.backend == "cpu" else _cupy()
        self.weights = to_backend(weights, self.backend)
        self.shape = (n,)
        self.mask = self._xp.ones(n, dtype=bool)
        self._row_sums = self._xp.asarray(self.weights.sum(axis=1)).ravel()

    @property
    def xp(self) -> Any:
        return self._xp

    def apply(self, values: Any) -> Any:
        native = self._xp.asarray(values)
        if native.shape[0] != self.shape[0]:
            raise ValueError("values and matrix operator have incompatible observation counts")
        return self.weights @ native

    def row_sums(self) -> Any:
        return self._row_sums
