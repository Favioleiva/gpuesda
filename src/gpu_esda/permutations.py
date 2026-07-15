"""Batched global and conditional-local permutation engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .backend import BackendName, _cupy, select_backend, to_backend, to_numpy


@dataclass
class PermutationSummary:
    pvalue: Any
    mean: Any
    std: Any
    minimum: Any
    maximum: Any
    simulations: np.ndarray | None
    timings: dict[str, float]


def choose_batch_size(
    n: int, permutations: int, dtype: Any, free_bytes: int, max_cardinality: int = 1
) -> int:
    """Choose one of the documented batch sizes using a conservative estimate."""
    itemsize = np.dtype(dtype).itemsize
    candidates = (1024, 512, 256, 128, 64)
    for batch in candidates:
        required = batch * n * max(2, max_cardinality + 1) * itemsize
        if batch <= permutations and required < free_bytes * 0.25:
            return batch
    return min(64, permutations)


def _directed_p(count_greater: Any, permutations: int, xp: Any) -> Any:
    extreme = xp.minimum(count_greater, permutations - count_greater)
    return (extreme + 1.0) / (permutations + 1.0)


def global_permutations(
    z: Any,
    weights: Any,
    observed: float,
    permutations: int,
    seed: int,
    backend: BackendName = "auto",
    batch_size: int | None = None,
    keep_simulations: bool = False,
) -> PermutationSummary:
    if permutations < 1:
        raise ValueError("permutations must be positive")
    info = select_backend(backend)
    xp = np if info.name == "cpu" else _cupy()
    z_native = xp.asarray(z)
    w_native = to_backend(weights, info.name)
    n = z_native.size
    s0 = float(np.asarray(to_numpy(weights.sum())).item())
    denom = float(np.asarray(to_numpy((z_native * z_native).sum())).item())
    if batch_size is None:
        free = 2**63 if info.name == "cpu" else int(xp.cuda.Device().mem_info[0])
        batch_size = choose_batch_size(n, permutations, z_native.dtype, free)
    rng = np.random.default_rng(seed) if info.name == "cpu" else xp.random.default_rng(seed)
    total = total_sq = 0.0
    minimum, maximum = np.inf, -np.inf
    greater = 0
    saved: list[np.ndarray] = []
    timings = {
        "random_indices": 0.0,
        "rearrangement": 0.0,
        "spatial_lag": 0.0,
        "statistic": 0.0,
        "reduction": 0.0,
        "host_transfer": 0.0,
    }
    completed = 0
    while completed < permutations:
        size = min(batch_size, permutations - completed)
        t = time.perf_counter()
        order = xp.argsort(rng.random((size, n)), axis=1)
        timings["random_indices"] += time.perf_counter() - t
        t = time.perf_counter()
        permuted = z_native[order]
        timings["rearrangement"] += time.perf_counter() - t
        t = time.perf_counter()
        lagged = w_native @ permuted.T
        timings["spatial_lag"] += time.perf_counter() - t
        t = time.perf_counter()
        sims = (n / s0) * xp.sum(permuted.T * lagged, axis=0) / denom
        timings["statistic"] += time.perf_counter() - t
        if info.name == "gpu":
            xp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        host = np.asarray(to_numpy(sims))
        timings["host_transfer"] += time.perf_counter() - t
        total += float(host.sum())
        total_sq += float(np.square(host).sum())
        minimum = min(minimum, float(host.min()))
        maximum = max(maximum, float(host.max()))
        greater += int(np.count_nonzero(host >= observed))
        if keep_simulations:
            saved.append(host)
        completed += size
    mean = total / permutations
    std = max(0.0, total_sq / permutations - mean * mean) ** 0.5
    pvalue = float(_directed_p(greater, permutations, np))
    return PermutationSummary(
        pvalue, mean, std, minimum, maximum, np.concatenate(saved) if saved else None, timings
    )


def _padded_neighbors(weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = to_numpy(weights)
    matrix = sp.csr_matrix(matrix)
    diagonal = matrix.diagonal().copy()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    degrees = np.diff(matrix.indptr)
    max_card = int(degrees.max(initial=0))
    padded = np.zeros((matrix.shape[0], max_card), dtype=matrix.dtype)
    for i in range(matrix.shape[0]):
        row = matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]]
        padded[i, : row.size] = row
    return padded, degrees, diagonal


def local_conditional_permutations(
    z: Any,
    weights: Any,
    observed: Any,
    permutations: int,
    seed: int,
    backend: BackendName = "auto",
    batch_size: int | None = None,
    keep_simulations: bool = False,
) -> PermutationSummary:
    """Conditional randomization holding each focal observation fixed."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    info = select_backend(backend)
    xp = np if info.name == "cpu" else _cupy()
    z_native = xp.asarray(z)
    observed_np = np.asarray(observed)
    n = z_native.size
    padded_np, degrees, diagonal_np = _padded_neighbors(weights)
    max_card = padded_np.shape[1]
    padded = xp.asarray(padded_np)
    diagonal = xp.asarray(diagonal_np)
    scaling = (n - 1) / float(np.square(np.asarray(to_numpy(z_native))).sum())
    if batch_size is None:
        free = 2**63 if info.name == "cpu" else int(xp.cuda.Device().mem_info[0])
        batch_size = choose_batch_size(n, permutations, z_native.dtype, free, max_card)
    rng = np.random.default_rng(seed) if info.name == "cpu" else xp.random.default_rng(seed)
    sums = np.zeros(n)
    sums_sq = np.zeros(n)
    mins = np.full(n, np.inf)
    maxs = np.full(n, -np.inf)
    greater = np.zeros(n, dtype=np.int64)
    saved: list[np.ndarray] = []
    timings = {
        "random_indices": 0.0,
        "rearrangement": 0.0,
        "spatial_lag": 0.0,
        "statistic": 0.0,
        "reduction": 0.0,
        "host_transfer": 0.0,
    }
    focal = xp.arange(n)[None, :, None]
    completed = 0
    while completed < permutations:
        size = min(batch_size, permutations - completed)
        t = time.perf_counter()
        ids = xp.argsort(rng.random((size, n - 1)), axis=1)[:, :max_card]
        timings["random_indices"] += time.perf_counter() - t
        t = time.perf_counter()
        adjusted = ids[:, None, :] + (ids[:, None, :] >= focal)
        sampled = z_native[adjusted]
        timings["rearrangement"] += time.perf_counter() - t
        t = time.perf_counter()
        neighbor_lag = xp.sum(sampled * padded[None, :, :], axis=2) + diagonal[None, :] * z_native
        timings["spatial_lag"] += time.perf_counter() - t
        t = time.perf_counter()
        sims = scaling * z_native[None, :] * neighbor_lag
        timings["statistic"] += time.perf_counter() - t
        if info.name == "gpu":
            xp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        host = np.asarray(to_numpy(sims))
        timings["host_transfer"] += time.perf_counter() - t
        sums += host.sum(axis=0)
        sums_sq += np.square(host).sum(axis=0)
        mins = np.minimum(mins, host.min(axis=0))
        maxs = np.maximum(maxs, host.max(axis=0))
        greater += np.count_nonzero(host >= observed_np[None, :], axis=0)
        if keep_simulations:
            saved.append(host)
        completed += size
    mean = sums / permutations
    std = np.sqrt(np.maximum(0, sums_sq / permutations - mean * mean))
    pvalue = _directed_p(greater, permutations, np)
    simulations = np.concatenate(saved, axis=0) if saved else None
    return PermutationSummary(pvalue, mean, std, mins, maxs, simulations, timings)
