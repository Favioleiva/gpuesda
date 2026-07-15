"""Weight-matrix diagnostics and memory estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .backend import to_numpy


@dataclass
class WeightDiagnostics:
    n: int
    shape: tuple[int, int]
    nnz: int
    density: float
    min_nonzero: float | None
    max_weight: float | None
    row_sum_min: float
    row_sum_max: float
    row_sum_mean: float
    islands: int
    dense_bytes: int
    csr_bytes: int
    construction_seconds: float = 0.0
    normalization_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def matrix_diagnostics(
    weights: Any, construction_seconds: float = 0.0, normalization_seconds: float = 0.0
) -> WeightDiagnostics:
    matrix = to_numpy(weights)
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        values = matrix.data[matrix.data != 0]
        nnz = matrix.nnz
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        csr_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        itemsize = matrix.dtype.itemsize
    else:
        matrix = np.asarray(matrix)
        values = matrix[matrix != 0]
        nnz = int(np.count_nonzero(matrix))
        row_sums = matrix.sum(axis=1)
        itemsize = matrix.dtype.itemsize
        csr_bytes = nnz * (itemsize + 4) + (matrix.shape[0] + 1) * 4
    n = matrix.shape[0]
    return WeightDiagnostics(
        n=n,
        shape=matrix.shape,
        nnz=nnz,
        density=nnz / (matrix.shape[0] * matrix.shape[1]),
        min_nonzero=float(values.min()) if values.size else None,
        max_weight=float(values.max()) if values.size else None,
        row_sum_min=float(row_sums.min()),
        row_sum_max=float(row_sums.max()),
        row_sum_mean=float(row_sums.mean()),
        islands=int(np.count_nonzero(row_sums == 0)),
        dense_bytes=n * n * itemsize,
        csr_bytes=int(csr_bytes),
        construction_seconds=construction_seconds,
        normalization_seconds=normalization_seconds,
    )
