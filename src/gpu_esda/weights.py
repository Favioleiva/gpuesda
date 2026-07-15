"""Independent inverse-distance weight construction."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from .backend import BackendName, _cupy, memory_diagnostics, select_backend
from .diagnostics import WeightDiagnostics, matrix_diagnostics


def _validate_coordinates(coordinates: Any, dtype: Any) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=dtype)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 1:
        raise ValueError("coordinates must have shape (n, dimensions) with n >= 2")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or infinity")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("duplicated coordinates create zero distances and are not allowed")
    return coords


def _choose_format(
    n: int, predicted_nnz: int, itemsize: int, free_bytes: int, requested: str
) -> str:
    if requested not in {"auto", "dense", "csr"}:
        raise ValueError("output_format must be 'auto', 'dense', or 'csr'")
    if requested != "auto":
        return requested
    density = predicted_nnz / (n * n)
    dense_bytes = n * n * itemsize
    csr_bytes = predicted_nnz * (itemsize + 4) + (n + 1) * 4
    return (
        "dense"
        if dense_bytes < free_bytes * 0.6 and (density >= 0.35 or dense_bytes <= csr_bytes)
        else "csr"
    )


def _normalize(matrix: Any, xp: Any, sparse: bool) -> Any:
    if sparse:
        sums = xp.asarray(matrix.sum(axis=1)).ravel()
        inv = xp.zeros_like(sums)
        nz = sums != 0
        inv[nz] = 1 / sums[nz]
        if xp is np:
            return sp.diags(inv) @ matrix
        import cupyx.scipy.sparse as cpsp

        return cpsp.diags(inv) @ matrix
    sums = matrix.sum(axis=1)
    safe = xp.where(sums == 0, 1, sums)
    return matrix / safe[:, None]


def inverse_distance_weights(
    coordinates: Any,
    power: float = 2,
    threshold: float | None = None,
    k: int | None = None,
    row_standardize: bool = True,
    binary: bool = False,
    output_format: Literal["auto", "dense", "csr"] = "auto",
    dtype: str | np.dtype = "float64",
    block_size: int | None = None,
    backend: BackendName = "auto",
    return_diagnostics: bool = False,
) -> Any | tuple[Any, WeightDiagnostics]:
    """Construct inverse-distance weights on CPU or GPU.

    Threshold and KNN restrictions are applied before weighting. The diagonal is
    always excluded. Block processing limits the temporary pairwise allocation.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    if threshold is not None and threshold <= 0:
        raise ValueError("threshold must be positive")
    coords_np = _validate_coordinates(coordinates, dtype)
    n = coords_np.shape[0]
    if k is not None and not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < n")
    info = select_backend(backend)
    xp = np if info.name == "cpu" else _cupy()
    coords = xp.asarray(coords_np)
    free = int(memory_diagnostics(info.name)["free_bytes"])
    predicted = n * (k if k is not None else (n - 1))
    fmt = _choose_format(n, predicted, coords.dtype.itemsize, free, output_format)
    # Building rows as dense blocks avoids a full temporary distance tensor.
    if block_size is None:
        bytes_per_row = max(1, n * coords.shape[1] * coords.dtype.itemsize)
        block_size = max(1, min(n, int(free * 0.15 / bytes_per_row)))
    if fmt == "dense":
        matrix = xp.zeros((n, n), dtype=coords.dtype)
    else:
        row_parts: list[Any] = []
        col_parts: list[Any] = []
        data_parts: list[Any] = []
    if info.name == "gpu":
        xp.cuda.Stream.null.synchronize()
    construct_start = time.perf_counter()
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        delta = coords[start:stop, None, :] - coords[None, :, :]
        dist2 = xp.sum(delta * delta, axis=2)
        local_rows = xp.arange(stop - start)
        dist2[local_rows, xp.arange(start, stop)] = xp.inf
        keep = xp.ones(dist2.shape, dtype=bool)
        if threshold is not None:
            keep &= dist2 <= threshold * threshold
        if k is not None:
            selected = xp.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
            kmask = xp.zeros_like(keep)
            kmask[local_rows[:, None], selected] = True
            keep &= kmask
        values = xp.where(keep, 1 if binary else dist2 ** (-power / 2), 0)
        if fmt == "dense":
            matrix[start:stop] = values
        else:
            rr, cc = xp.nonzero(values)
            row_parts.append(rr + start)
            col_parts.append(cc)
            data_parts.append(values[rr, cc])
    if fmt == "csr":
        rows = xp.concatenate(row_parts) if row_parts else xp.array([], dtype=xp.int32)
        cols = xp.concatenate(col_parts) if col_parts else xp.array([], dtype=xp.int32)
        data = xp.concatenate(data_parts) if data_parts else xp.array([], dtype=coords.dtype)
        if info.name == "cpu":
            matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            import cupyx.scipy.sparse as cpsp

            matrix = cpsp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if info.name == "gpu":
        xp.cuda.Stream.null.synchronize()
    construction_seconds = time.perf_counter() - construct_start
    if info.name == "gpu":
        xp.cuda.Stream.null.synchronize()
    norm_start = time.perf_counter()
    if row_standardize:
        matrix = _normalize(matrix, xp, fmt == "csr")
    if info.name == "gpu":
        xp.cuda.Stream.null.synchronize()
    normalization_seconds = time.perf_counter() - norm_start
    diag = matrix_diagnostics(matrix, construction_seconds, normalization_seconds)
    return (matrix, diag) if return_diagnostics else matrix
