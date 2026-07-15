"""Observed Global and Local Moran statistics for any spatial operator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backend import to_backend, to_numpy
from ..multiple_testing import adjust_pvalues
from ..operators.base import SpatialOperator
from .permutations import global_raster_permutations, local_raster_conditional_permutations

QUADRANT_LABELS = {0: "island", 1: "HH", 2: "LH", 3: "LL", 4: "HL"}


@dataclass
class RasterMoranGlobalResult:
    I: float  # noqa: E741
    EI: float
    n: int
    s0: float
    backend: str
    dtype: str
    timings: dict[str, float]
    p_sim: float | None = None
    EI_sim: float | None = None
    seI_sim: float | None = None
    z_sim: float | None = None
    permutations: int = 0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass
class RasterMoranLocalResult:
    local_i: Any
    z: Any
    spatial_lag: Any
    quadrant_code: Any
    island: Any
    mask: Any
    backend: str
    dtype: str
    timings: dict[str, float]
    p_sim: Any = None
    p_fdr: Any = None
    significant: Any = None
    permutation_mean: Any = None
    permutation_std: Any = None
    z_sim: Any = None
    permutations: int = 0
    seed: int | None = None

    def to_frame(self, raster: Any) -> pd.DataFrame:
        codes = raster.gather(self.quadrant_code).astype(np.int8)
        valid = raster.gather(self.mask).astype(bool)
        islands = raster.gather(self.island).astype(bool)
        labels = np.array(["not valid", "HH", "LH", "LL", "HL", "island"])
        output_codes = np.where(~valid, 0, np.where(islands, 5, codes))
        return pd.DataFrame(
            {
                "cell_id": raster.valid_cell_ids,
                "local_i": raster.gather(self.local_i),
                "spatial_lag": raster.gather(self.spatial_lag),
                "quadrant": labels[output_codes],
                "p_sim": np.nan if self.p_sim is None else raster.gather(self.p_sim),
                "p_fdr": np.nan if self.p_fdr is None else raster.gather(self.p_fdr),
                "significant": False
                if self.significant is None
                else raster.gather(self.significant),
            }
        )


def _standardize(values: Any, operator: SpatialOperator) -> tuple[Any, int]:
    xp = operator.xp
    native = xp.asarray(values)
    if tuple(native.shape) != operator.shape:
        raise ValueError(f"values shape {native.shape} does not match operator {operator.shape}")
    if not bool(xp.all(xp.isfinite(native[operator.mask])).item()):
        raise ValueError("valid cells contain NaN or infinity")
    n = int(operator.mask.sum().item())
    if n < 2:
        raise ValueError("at least two valid raster cells are required")
    valid = native[operator.mask]
    mean = valid.sum(dtype=xp.float64) / n
    mean_native = xp.asarray(mean, dtype=native.dtype)
    centered = xp.where(operator.mask, native - mean_native, xp.asarray(0, dtype=native.dtype))
    variance = xp.sum(centered * centered, dtype=xp.float64) / n
    if float(variance.item()) == 0:
        raise ValueError("constant raster values are not valid for Moran statistics")
    standard_deviation = xp.asarray(xp.sqrt(variance), dtype=native.dtype)
    return centered / standard_deviation, n


def _components(
    values: Any, operator: SpatialOperator
) -> tuple[Any, int, Any, Any, dict[str, float]]:
    xp = operator.xp
    started = time.perf_counter()
    z, n = _standardize(values, operator)
    if operator.backend == "gpu":
        xp.cuda.Stream.null.synchronize()
    standardization = time.perf_counter() - started
    started = time.perf_counter()
    lag = operator.apply(z)
    if operator.backend == "gpu":
        xp.cuda.Stream.null.synchronize()
    spatial_lag = time.perf_counter() - started
    denominator = xp.sum(z * z, dtype=xp.float64)
    return z, n, lag, denominator, {"standardization": standardization, "spatial_lag": spatial_lag}


def _global_from_components(
    z: Any,
    n: int,
    lag: Any,
    denominator: Any,
    operator: SpatialOperator,
    timings: dict[str, float],
) -> RasterMoranGlobalResult:
    started = time.perf_counter()
    xp = operator.xp
    s0 = operator.s0()
    numerator = xp.sum(z * lag, dtype=xp.float64)
    observed = float((n / s0 * numerator / denominator).item())
    reduction = time.perf_counter() - started
    return RasterMoranGlobalResult(
        observed,
        -1.0 / (n - 1),
        n,
        s0,
        operator.backend,
        str(z.dtype),
        {**timings, "global_reduction": reduction, "total": sum(timings.values()) + reduction},
    )


def _local_from_components(
    z: Any,
    n: int,
    lag: Any,
    denominator: Any,
    operator: SpatialOperator,
    timings: dict[str, float],
) -> RasterMoranLocalResult:
    started = time.perf_counter()
    xp = operator.xp
    local_scale = xp.asarray((n - 1) / denominator, dtype=z.dtype)
    local_i = local_scale * z * lag
    island = operator.islands()
    zp, lp = z > 0, lag > 0
    codes = xp.where(zp & lp, 1, xp.where(~zp & lp, 2, xp.where(~zp & ~lp, 3, 4)))
    codes = xp.where(operator.mask & ~island, codes, 0).astype(xp.int8)
    local_i = xp.where(operator.mask, local_i, xp.nan)
    lag_output = xp.where(operator.mask, lag, xp.nan)
    if operator.backend == "gpu":
        xp.cuda.Stream.null.synchronize()
    local_seconds = time.perf_counter() - started
    return RasterMoranLocalResult(
        local_i,
        z,
        lag_output,
        codes,
        island,
        operator.mask,
        operator.backend,
        str(z.dtype),
        {
            **timings,
            "local_statistic": local_seconds,
            "total": sum(timings.values()) + local_seconds,
        },
    )


def moran_global(
    values: Any,
    operator: SpatialOperator,
    permutations: int = 0,
    seed: int | None = 12345,
    batch_size: int | None = None,
) -> RasterMoranGlobalResult:
    z, n, lag, denominator, timings = _components(values, operator)
    result = _global_from_components(z, n, lag, denominator, operator, timings)
    if permutations:
        if seed is None:
            raise ValueError("seed must be explicit when permutations are requested")
        summary = global_raster_permutations(z, operator, result.I, permutations, seed, batch_size)
        result.p_sim = summary.pvalue
        result.EI_sim = summary.mean
        result.seI_sim = summary.std
        result.z_sim = (result.I - summary.mean) / summary.std if summary.std else np.nan
        result.permutations = permutations
        result.seed = seed
        result.timings["permutations"] = sum(summary.timings.values())
        result.timings.update(
            {f"permutation_{key}": value for key, value in summary.timings.items()}
        )
    return result


def moran_local(
    values: Any,
    operator: SpatialOperator,
    permutations: int = 0,
    seed: int | None = 12345,
    batch_size: int | None = None,
    cell_chunk_size: int = 250_000,
    fdr: bool = True,
    alpha: float = 0.05,
) -> RasterMoranLocalResult:
    z, n, lag, denominator, timings = _components(values, operator)
    result = _local_from_components(z, n, lag, denominator, operator, timings)
    if permutations:
        if seed is None:
            raise ValueError("seed must be explicit when permutations are requested")
        summary = local_raster_conditional_permutations(
            z, operator, result.local_i, permutations, seed, batch_size, cell_chunk_size
        )
        result.p_sim = summary.pvalue
        result.permutation_mean = summary.mean
        result.permutation_std = summary.std
        safe_std = operator.xp.where(summary.std > 0, summary.std, 1)
        result.z_sim = operator.xp.where(
            summary.std > 0, (result.local_i - summary.mean) / safe_std, operator.xp.nan
        )
        result.permutations = permutations
        result.seed = seed
        result.timings["permutations"] = sum(summary.timings.values())
        result.timings.update(
            {f"permutation_{key}": value for key, value in summary.timings.items()}
        )
        if fdr:
            mask_host = np.asarray(to_numpy(operator.mask), dtype=bool)
            p_host = np.asarray(to_numpy(result.p_sim))[mask_host]
            adjusted, reject = adjust_pvalues(p_host, alpha=alpha)
            adjusted_full = np.full(operator.shape, np.nan, dtype=np.float32)
            reject_full = np.zeros(operator.shape, dtype=bool)
            adjusted_full[mask_host] = adjusted
            reject_full[mask_host] = reject
            result.p_fdr = to_backend(adjusted_full, operator.backend)
            result.significant = to_backend(reject_full, operator.backend)
        else:
            result.significant = result.p_sim <= alpha
    return result


def moran_observed(
    values: Any, operator: SpatialOperator
) -> tuple[RasterMoranGlobalResult, RasterMoranLocalResult]:
    """Compute Global and Local Moran from one standardization and one spatial lag."""
    z, n, lag, denominator, timings = _components(values, operator)
    return (
        _global_from_components(z, n, lag, denominator, operator, timings),
        _local_from_components(z, n, lag, denominator, operator, timings),
    )
