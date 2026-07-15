"""Global Moran statistic and batched panel API."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .backend import BackendName, select_backend, to_numpy
from .lag import spatial_lag
from .permutations import global_permutations
from .validation import validate_square_weights, validate_values


def _weight_moments(weights: Any) -> tuple[float, float, float]:
    w = sp.csr_matrix(to_numpy(weights), dtype=float)
    s0 = float(w.sum())
    s1 = float(0.5 * (w + w.T).power(2).sum())
    rows = np.asarray(w.sum(axis=1)).ravel()
    cols = np.asarray(w.sum(axis=0)).ravel()
    s2 = float(np.square(rows + cols).sum())
    if s0 == 0:
        raise ValueError("weights have zero total weight")
    return s0, s1, s2


@dataclass(init=False)
class Moran:
    I: float  # noqa: E741 - conventional public name for Moran's I
    EI: float
    VI_norm: float
    VI_rand: float
    z_norm: float
    p_norm: float
    p_sim: float | None
    EI_sim: float | None
    seI_sim: float | None
    z_sim: float | None
    permutations: int
    seed: int | None
    backend: str
    dtype: str
    timings: dict[str, Any]
    sim: np.ndarray | None

    def __init__(
        self,
        y: Any,
        weights: Any,
        permutations: int = 999,
        seed: int | None = 12345,
        backend: BackendName = "auto",
        dtype: str = "float64",
        keep_simulations: bool = False,
        batch_size: int | None = None,
    ):
        started = time.perf_counter()
        n = validate_square_weights(weights)
        values = validate_values(y, n).astype(dtype, copy=False)
        if values.ndim != 1:
            raise ValueError(
                "Moran accepts one vector; use moran_panel for multiple columns/periods"
            )
        z = values - values.mean()
        z2ss = float(z @ z)
        s0, s1, s2 = _weight_moments(weights)
        lag_start = time.perf_counter()
        lagged = spatial_lag(weights, z, backend=backend)
        lag_time = time.perf_counter() - lag_start
        self.I = float(n / s0 * (z @ lagged) / z2ss)
        self.EI = -1.0 / (n - 1)
        s02 = s0 * s0
        n2 = n * n
        self.VI_norm = (n2 * s1 - n * s2 + 3 * s02) / ((n - 1) * (n + 1) * s02) - self.EI**2
        k = (np.sum(z**4) / n) / ((np.sum(z**2) / n) ** 2)
        a = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
        b = k * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        self.VI_rand = (
            (a - b) / ((n - 1) * (n - 2) * (n - 3) * s02) - self.EI**2 if n > 3 else np.nan
        )
        self.z_norm = (self.I - self.EI) / np.sqrt(self.VI_norm)
        self.p_norm = float(2 * stats.norm.sf(abs(self.z_norm)))
        self.permutations = int(permutations)
        self.seed = seed
        self.backend = select_backend(backend).name
        self.dtype = str(values.dtype)
        self.sim = None
        self.p_sim = self.EI_sim = self.seI_sim = self.z_sim = None
        perm_timing: dict[str, float] = {}
        if permutations:
            if seed is None:
                raise ValueError("seed must be explicit when permutations are requested")
            summary = global_permutations(
                z, weights, self.I, permutations, seed, self.backend, batch_size, keep_simulations
            )
            self.p_sim = float(summary.pvalue)
            self.EI_sim = float(summary.mean)
            self.seI_sim = float(summary.std)
            self.z_sim = (self.I - self.EI_sim) / self.seI_sim if self.seI_sim else np.nan
            self.sim = summary.simulations
            perm_timing = summary.timings
        self.timings = {
            "spatial_lag": lag_time,
            "permutations": perm_timing,
            "total": time.perf_counter() - started,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("sim", None)
        return data

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        return (
            f"Moran(I={self.I:.6g}, EI={self.EI:.6g}, p_sim={self.p_sim}, backend='{self.backend}')"
        )


def moran_panel(
    y: Any,
    weights: Any,
    permutations: int = 999,
    seed: int = 12345,
    backend: BackendName = "auto",
    variable_names: list[str] | None = None,
    period_names: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    n = validate_square_weights(weights)
    values = validate_values(y, n)
    if values.ndim == 1:
        values = values[:, None]
    trailing = values.shape[1:]
    flat = values.reshape(n, -1)
    rows = []
    for index in range(flat.shape[1]):
        result = Moran(
            flat[:, index],
            weights,
            permutations=permutations,
            seed=seed + index,
            backend=backend,
            **kwargs,
        )
        position = np.unravel_index(index, trailing)
        record = result.to_dict()
        record["variable"] = variable_names[position[0]] if variable_names else position[0]
        record["period"] = (
            (period_names[position[1]] if period_names else position[1])
            if len(position) > 1
            else None
        )
        rows.append(record)
    return pd.DataFrame(rows)
