"""Export reusable PySAL CPU baselines for the official dataset."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import esda
import geopandas as gpd
import libpysal
import numpy as np
import psutil

from gpu_esda.baseline import knn_weights, queen_weights


def main() -> int:
    frame = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    y = frame.CRIME.to_numpy()
    coordinates = np.column_stack((frame.X, frame.Y))
    results: list[dict[str, object]] = []
    for name, weights in (("queen", queen_weights(frame)), ("knn8", knn_weights(coordinates, 8))):
        for permutations in (99, 999, 9999):
            np.random.seed(12345)
            start = time.perf_counter()
            global_result = esda.Moran(y, weights, permutations=permutations)
            local_result = esda.Moran_Local(
                y, weights, permutations=permutations, seed=12345, keep_simulations=False
            )
            results.append(
                {
                    "weight": name,
                    "n": len(y),
                    "nnz": weights.sparse.nnz,
                    "density": weights.sparse.nnz / len(y) ** 2,
                    "permutations": permutations,
                    "seed": 12345,
                    "global_I": float(global_result.I),
                    "global_p": float(global_result.p_sim),
                    "local_significant": int(np.sum(local_result.p_sim <= 0.05)),
                    "elapsed_seconds": time.perf_counter() - start,
                    "rss_bytes": psutil.Process().memory_info().rss,
                }
            )
    payload = {
        "hardware": platform.platform(),
        "libpysal": libpysal.__version__,
        "esda": esda.__version__,
        "results": results,
    }
    Path("results/cpu_baseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
