"""Local Moran/LISA statistics with conditional permutations and FDR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backend import BackendName, select_backend
from .lag import spatial_lag
from .multiple_testing import adjust_pvalues
from .permutations import local_conditional_permutations
from .validation import validate_square_weights, validate_values

QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


@dataclass(init=False)
class MoranLocal:
    def __init__(
        self,
        y: Any,
        weights: Any,
        permutations: int = 999,
        seed: int | None = 12345,
        backend: BackendName = "auto",
        dtype: str = "float64",
        fdr: bool = True,
        alpha: float = 0.05,
        keep_simulations: bool = False,
        batch_size: int | None = None,
    ):
        started = time.perf_counter()
        n = validate_square_weights(weights)
        values = validate_values(y, n).astype(dtype, copy=False)
        if values.ndim != 1:
            raise ValueError("MoranLocal accepts one vector; use local_moran_panel for panels")
        self.y = values
        self.z = (values - values.mean()) / values.std()
        lag_started = time.perf_counter()
        self.lag = spatial_lag(weights, self.z, backend=backend)
        lag_time = time.perf_counter() - lag_started
        self.Is = (n - 1) * self.z * self.lag / np.sum(self.z**2)
        zp = self.z > 0
        lp = self.lag > 0
        self.q = np.where(zp & lp, 1, np.where(~zp & lp, 2, np.where(~zp & ~lp, 3, 4)))
        self.quadrant = np.array([QUADRANT_LABELS[int(code)] for code in self.q], dtype=object)
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
            summary = local_conditional_permutations(
                self.z,
                weights,
                self.Is,
                permutations,
                seed,
                self.backend,
                batch_size,
                keep_simulations,
            )
            self.p_sim = np.asarray(summary.pvalue)
            self.EI_sim = np.asarray(summary.mean)
            self.seI_sim = np.asarray(summary.std)
            self.z_sim = np.divide(
                self.Is - self.EI_sim, self.seI_sim, out=np.full(n, np.nan), where=self.seI_sim != 0
            )
            self.sim = summary.simulations
            perm_timing = summary.timings
        self.significant = (
            self.p_sim <= alpha if self.p_sim is not None else np.zeros(n, dtype=bool)
        )
        if fdr and self.p_sim is not None:
            self.p_fdr, self.significant_fdr = adjust_pvalues(self.p_sim, alpha=alpha)
        else:
            self.p_fdr = self.p_sim.copy() if self.p_sim is not None else None
            self.significant_fdr = self.significant.copy()
        self.cluster = np.where(self.significant, self.quadrant, "not significant")
        self.cluster_fdr = np.where(self.significant_fdr, self.quadrant, "not significant")
        self.timings = {
            "spatial_lag": lag_time,
            "permutations": perm_timing,
            "total": time.perf_counter() - started,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "Is": self.Is,
            "z": self.z,
            "lag": self.lag,
            "q": self.q,
            "quadrant": self.quadrant,
            "p_sim": self.p_sim,
            "p_fdr": self.p_fdr,
            "significant": self.significant,
            "significant_fdr": self.significant_fdr,
            "backend": self.backend,
            "dtype": self.dtype,
            "timings": self.timings,
        }

    def to_frame(self, ids: Any = None) -> pd.DataFrame:
        ids = np.arange(self.Is.size) if ids is None else np.asarray(ids)
        if ids.size != self.Is.size:
            raise ValueError("ids must match the number of observations")
        return pd.DataFrame(
            {
                "id": ids,
                "local_I": self.Is,
                "z": self.z,
                "lag": self.lag,
                "quadrant_code": self.q,
                "quadrant": self.quadrant,
                "p_sim": self.p_sim,
                "p_fdr": self.p_fdr,
                "significant": self.significant,
                "significant_fdr": self.significant_fdr,
                "cluster": self.cluster,
                "cluster_fdr": self.cluster_fdr,
                "backend": self.backend,
            }
        )

    def summary(self) -> str:
        return f"MoranLocal(n={self.Is.size}, significant={self.significant.sum()}, backend='{self.backend}')"


def local_moran_panel(
    y: Any,
    weights: Any,
    permutations: int = 999,
    seed: int = 12345,
    backend: BackendName = "auto",
    ids: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    n = validate_square_weights(weights)
    values = validate_values(y, n)
    if values.ndim == 1:
        values = values[:, None]
    trailing = values.shape[1:]
    flat = values.reshape(n, -1)
    frames = []
    for index in range(flat.shape[1]):
        result = MoranLocal(
            flat[:, index],
            weights,
            permutations=permutations,
            seed=seed + index,
            backend=backend,
            **kwargs,
        )
        frame = result.to_frame(ids)
        position = np.unravel_index(index, trailing)
        frame["variable"] = position[0]
        frame["period"] = position[1] if len(position) > 1 else None
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
