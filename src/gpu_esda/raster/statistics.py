"""Observed Global and Local Moran statistics for any spatial operator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..operators.base import SpatialOperator

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
    permutations: int = 0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items() if key != "z"}


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
    permutations: int = 0
    seed: int | None = None

    def to_frame(self, raster: Any) -> pd.DataFrame:
        codes = raster.gather(self.quadrant_code).astype(np.int8)
        return pd.DataFrame(
            {
                "cell_id": raster.valid_cell_ids,
                "local_i": raster.gather(self.local_i),
                "spatial_lag": raster.gather(self.spatial_lag),
                "quadrant": np.array([QUADRANT_LABELS[int(code)] for code in codes]),
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
    centered = xp.where(operator.mask, native - mean, 0)
    variance = xp.sum(centered * centered, dtype=xp.float64) / n
    if float(variance.item()) == 0:
        raise ValueError("constant raster values are not valid for Moran statistics")
    return centered / xp.sqrt(variance), n


def moran_global(values: Any, operator: SpatialOperator) -> RasterMoranGlobalResult:
    started = time.perf_counter()
    xp = operator.xp
    z, n = _standardize(values, operator)
    lag_started = time.perf_counter()
    lag = operator.apply(z)
    if operator.backend == "gpu":
        xp.cuda.Stream.null.synchronize()
    lag_seconds = time.perf_counter() - lag_started
    s0 = operator.s0()
    denominator = xp.sum(z * z, dtype=xp.float64)
    numerator = xp.sum(z * lag, dtype=xp.float64)
    observed = float((n / s0 * numerator / denominator).item())
    return RasterMoranGlobalResult(
        observed,
        -1.0 / (n - 1),
        n,
        s0,
        operator.backend,
        str(z.dtype),
        {"spatial_lag": lag_seconds, "total": time.perf_counter() - started},
    )


def moran_local(values: Any, operator: SpatialOperator) -> RasterMoranLocalResult:
    started = time.perf_counter()
    xp = operator.xp
    z, n = _standardize(values, operator)
    lag_started = time.perf_counter()
    lag = operator.apply(z)
    if operator.backend == "gpu":
        xp.cuda.Stream.null.synchronize()
    lag_seconds = time.perf_counter() - lag_started
    denominator = xp.sum(z * z, dtype=xp.float64)
    local_i = (n - 1) * z * lag / denominator
    island = operator.islands()
    zp, lp = z > 0, lag > 0
    codes = xp.where(zp & lp, 1, xp.where(~zp & lp, 2, xp.where(~zp & ~lp, 3, 4)))
    codes = xp.where(operator.mask & ~island, codes, 0).astype(xp.int8)
    local_i = xp.where(operator.mask, local_i, xp.nan)
    lag = xp.where(operator.mask, lag, xp.nan)
    return RasterMoranLocalResult(
        local_i,
        z,
        lag,
        codes,
        island,
        operator.mask,
        operator.backend,
        str(z.dtype),
        {"spatial_lag": lag_seconds, "total": time.perf_counter() - started},
    )
