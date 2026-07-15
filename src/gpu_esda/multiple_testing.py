"""Multiple-testing corrections."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.stats.multitest import multipletests

from .backend import BackendName, to_numpy


def adjust_pvalues(
    pvalues: Any, method: str = "fdr_bh", alpha: float = 0.05, backend: BackendName = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    del backend  # sorting is intentionally kept on CPU until profiling justifies a GPU path
    values = np.asarray(to_numpy(pvalues), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be a finite one-dimensional array in [0, 1]")
    reject, adjusted, _, _ = multipletests(values, alpha=alpha, method=method)
    return adjusted, reject
