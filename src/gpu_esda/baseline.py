"""Reusable PySAL/ESDA CPU reference workflows."""

from __future__ import annotations

from typing import Any

import esda
import libpysal
import numpy as np
from statsmodels.stats.multitest import multipletests


def queen_weights(frame: Any, transform: str = "r") -> Any:
    w = libpysal.weights.Queen.from_dataframe(frame, use_index=True)
    w.transform = transform
    return w


def knn_weights(coordinates: Any, k: int = 8, transform: str = "r") -> Any:
    w = libpysal.weights.KNN.from_array(np.asarray(coordinates), k=k)
    w.transform = transform
    return w


def distance_band_weights(
    coordinates: Any, threshold: float, binary: bool = True, alpha: float = -2, transform: str = "r"
) -> Any:
    w = libpysal.weights.DistanceBand(
        np.asarray(coordinates),
        threshold=threshold,
        binary=binary,
        alpha=alpha,
        silence_warnings=True,
    )
    w.transform = transform
    return w


def reference_esda(
    y: Any, weights: Any, permutations: int = 999, seed: int = 12345
) -> dict[str, Any]:
    np.random.seed(seed)
    global_result = esda.Moran(y, weights, permutations=permutations)
    local_result = esda.Moran_Local(y, weights, permutations=permutations, seed=seed)
    reject, adjusted, _, _ = multipletests(local_result.p_sim, method="fdr_bh")
    return {
        "moran": global_result,
        "local_moran": local_result,
        "local_p_fdr": adjusted,
        "local_reject_fdr": reject,
    }
