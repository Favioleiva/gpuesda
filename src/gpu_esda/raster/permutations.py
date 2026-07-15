"""Streaming raster permutation inference without n-by-R result storage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..operators.raster import RasterStencilOperator


@dataclass
class RasterPermutationSummary:
    pvalue: Any
    mean: Any
    std: Any
    minimum: Any
    maximum: Any
    batch_size: int
    permutations: int
    timings: dict[str, float]
    cell_chunk_size: int | None = None


def choose_raster_batch_size(
    operator: RasterStencilOperator, permutations: int, arrays_per_permutation: int = 3
) -> int:
    """Choose a conservative permutation batch from current free memory."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if operator.backend == "cpu":
        return min(permutations, 4)
    free = int(operator.xp.cuda.Device().mem_info[0])
    per = int(np.prod(operator.shape)) * operator.dtype.itemsize * arrays_per_permutation
    safe = max(1, int(free * 0.2 // max(1, per)))
    return min(permutations, 16, safe)


def _directed_p(greater: Any, permutations: int, xp: Any) -> Any:
    return (xp.minimum(greater, permutations - greater) + 1.0) / (permutations + 1.0)


def global_raster_permutations(
    z: Any,
    operator: RasterStencilOperator,
    observed: float,
    permutations: int,
    seed: int,
    batch_size: int | None = None,
) -> RasterPermutationSummary:
    """Unconditional global permutation of values over valid raster positions."""
    xp = operator.xp
    z = xp.asarray(z)
    valid_indices = xp.flatnonzero(operator.mask.ravel())
    valid_values = z.ravel()[valid_indices]
    n = int(valid_values.size)
    denominator = xp.sum(valid_values * valid_values, dtype=xp.float64)
    s0 = operator.s0()
    batch = batch_size or choose_raster_batch_size(operator, permutations)
    if batch < 1:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed) if operator.backend == "cpu" else xp.random.RandomState(seed)
    total = xp.asarray(0, dtype=xp.float64)
    total_sq = xp.asarray(0, dtype=xp.float64)
    minimum = xp.asarray(xp.inf, dtype=xp.float64)
    maximum = xp.asarray(-xp.inf, dtype=xp.float64)
    greater = xp.asarray(0, dtype=xp.int64)
    completed = 0
    timings = {"randomization": 0.0, "spatial_lag": 0.0, "reduction": 0.0}
    while completed < permutations:
        current = min(batch, permutations - completed)
        try:
            started = time.perf_counter()
            shuffled = xp.stack([rng.permutation(valid_values) for _ in range(current)])
            grids = xp.zeros((current, *operator.shape), dtype=z.dtype)
            grids.reshape(current, -1)[:, valid_indices] = shuffled
            if operator.backend == "gpu":
                xp.cuda.Stream.null.synchronize()
            timings["randomization"] += time.perf_counter() - started
            started = time.perf_counter()
            lag = operator.apply(grids)
            if operator.backend == "gpu":
                xp.cuda.Stream.null.synchronize()
            timings["spatial_lag"] += time.perf_counter() - started
            started = time.perf_counter()
            sims = (n / s0) * xp.sum(grids * lag, axis=(-2, -1), dtype=xp.float64) / denominator
            total += sims.sum(dtype=xp.float64)
            total_sq += xp.sum(sims * sims, dtype=xp.float64)
            minimum = xp.minimum(minimum, sims.min())
            maximum = xp.maximum(maximum, sims.max())
            greater += xp.count_nonzero(sims >= observed)
            if operator.backend == "gpu":
                xp.cuda.Stream.null.synchronize()
            timings["reduction"] += time.perf_counter() - started
            completed += current
            del shuffled, grids, lag, sims
        except Exception as exc:
            oom_type = getattr(getattr(xp, "cuda", None), "memory", None)
            is_oom = oom_type is not None and isinstance(exc, oom_type.OutOfMemoryError)
            if not is_oom or current == 1:
                raise
            batch = max(1, current // 2)
            xp.get_default_memory_pool().free_all_blocks()
    mean = total / permutations
    std = xp.sqrt(xp.maximum(0, total_sq / permutations - mean * mean))
    return RasterPermutationSummary(
        float(_directed_p(greater, permutations, xp).item()),
        float(mean.item()),
        float(std.item()),
        float(minimum.item()),
        float(maximum.item()),
        batch,
        permutations,
        timings,
    )


def _draw_without_replacement(rng: Any, xp: Any, high: int, shape: tuple[int, int, int]) -> Any:
    """Draw ordered small samples without replacement using duplicate resampling."""
    draws = rng.integers(0, high, size=shape, dtype=xp.int64)
    for column in range(1, shape[2]):
        duplicate = xp.any(draws[:, :, column, None] == draws[:, :, :column], axis=2)
        while bool(xp.any(duplicate).item()):
            draws[:, :, column] = xp.where(
                duplicate,
                rng.integers(0, high, size=duplicate.shape, dtype=xp.int64),
                draws[:, :, column],
            )
            duplicate = xp.any(draws[:, :, column, None] == draws[:, :, :column], axis=2)
    return draws


def local_raster_conditional_permutations(
    z: Any,
    operator: RasterStencilOperator,
    observed: Any,
    permutations: int,
    seed: int,
    batch_size: int | None = None,
    cell_chunk_size: int = 250_000,
) -> RasterPermutationSummary:
    """PySAL-compatible conditional randomization with chunked focal cells.

    The focal value is held fixed. For each focal cell and replicate, neighbor-count
    values are sampled without replacement from all other valid cells, then combined
    with that focal cell's actual normalized stencil weights.
    """
    xp = operator.xp
    z = xp.asarray(z)
    observed = xp.asarray(observed)
    valid_indices = xp.flatnonzero(operator.mask.ravel())
    z_valid = z.ravel()[valid_indices]
    observed_valid = observed.ravel()[valid_indices]
    n = int(z_valid.size)
    weights = operator.neighbor_weights().reshape(len(operator.stencil.offsets), -1)
    weights = weights[:, valid_indices].T
    cardinality = weights.shape[1]
    if cardinality > n - 1:
        raise ValueError("stencil cardinality cannot exceed the conditional population")
    batch = batch_size or min(8, choose_raster_batch_size(operator, permutations, 2))
    chunk_size = min(cell_chunk_size, n)
    rng = np.random.default_rng(seed) if operator.backend == "cpu" else xp.random.default_rng(seed)
    p_valid = xp.empty(n, dtype=xp.float32)
    mean_valid = xp.empty(n, dtype=xp.float32)
    std_valid = xp.empty(n, dtype=xp.float32)
    min_valid = xp.empty(n, dtype=xp.float32)
    max_valid = xp.empty(n, dtype=xp.float32)
    scaling = xp.asarray((n - 1) / xp.sum(z_valid * z_valid, dtype=xp.float64), dtype=z_valid.dtype)
    timings = {"randomization": 0.0, "statistic": 0.0, "reduction": 0.0}
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        focal = xp.arange(start, stop, dtype=xp.int64)
        focal_z = z_valid[start:stop]
        focal_weights = weights[start:stop]
        greater = xp.zeros(stop - start, dtype=xp.int32)
        sums = xp.zeros(stop - start, dtype=xp.float64)
        sums_sq = xp.zeros(stop - start, dtype=xp.float64)
        mins = xp.full(stop - start, xp.inf, dtype=xp.float32)
        maxs = xp.full(stop - start, -xp.inf, dtype=xp.float32)
        completed = 0
        while completed < permutations:
            current = min(batch, permutations - completed)
            try:
                started = time.perf_counter()
                draws = _draw_without_replacement(
                    rng, xp, n - 1, (current, stop - start, cardinality)
                )
                draws += draws >= focal[None, :, None]
                sampled = z_valid[draws]
                if operator.backend == "gpu":
                    xp.cuda.Stream.null.synchronize()
                timings["randomization"] += time.perf_counter() - started
                started = time.perf_counter()
                sims = (
                    scaling * focal_z[None, :] * xp.sum(sampled * focal_weights[None, :, :], axis=2)
                )
                if operator.backend == "gpu":
                    xp.cuda.Stream.null.synchronize()
                timings["statistic"] += time.perf_counter() - started
                started = time.perf_counter()
                greater += xp.count_nonzero(sims >= observed_valid[start:stop][None, :], axis=0)
                sums += sims.sum(axis=0, dtype=xp.float64)
                sums_sq += xp.sum(sims * sims, axis=0, dtype=xp.float64)
                mins = xp.minimum(mins, sims.min(axis=0))
                maxs = xp.maximum(maxs, sims.max(axis=0))
                timings["reduction"] += time.perf_counter() - started
                completed += current
                del draws, sampled, sims
            except Exception as exc:
                oom_type = getattr(getattr(xp, "cuda", None), "memory", None)
                is_oom = oom_type is not None and isinstance(exc, oom_type.OutOfMemoryError)
                if not is_oom or current == 1:
                    raise
                batch = max(1, current // 2)
                xp.get_default_memory_pool().free_all_blocks()
        mean = sums / permutations
        p_valid[start:stop] = _directed_p(greater, permutations, xp)
        mean_valid[start:stop] = mean
        std_valid[start:stop] = xp.sqrt(xp.maximum(0, sums_sq / permutations - mean * mean))
        min_valid[start:stop] = mins
        max_valid[start:stop] = maxs
    outputs = []
    for valid in (p_valid, mean_valid, std_valid, min_valid, max_valid):
        full = xp.full(operator.mask.size, xp.nan, dtype=valid.dtype)
        full[valid_indices] = valid
        outputs.append(full.reshape(operator.shape))
    return RasterPermutationSummary(
        *outputs, batch, permutations, timings, cell_chunk_size=chunk_size
    )
