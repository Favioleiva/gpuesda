"""Run representative dense/CSR, transfer, lag, and Moran benchmarks."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from gpu_esda import Moran, gpu_available, inverse_distance_weights, spatial_lag
from gpu_esda.backend import synchronize, to_backend
from gpu_esda.benchmarking import benchmark_call, synthetic_coordinates


def main() -> int:
    output: list[dict[str, object]] = []
    for n in (500, 1793):
        coordinates = synthetic_coordinates(n, seed=12345)
        y = np.random.default_rng(12345).normal(size=n)
        matrices: dict[tuple[str, str], object] = {}
        for backend in ["cpu", "gpu"] if gpu_available() else ["cpu"]:
            for fmt in ("dense", "csr"):
                start = time.perf_counter()
                matrix, diagnostics = inverse_distance_weights(
                    coordinates, backend=backend, output_format=fmt, return_diagnostics=True
                )
                synchronize(backend)
                construction = time.perf_counter() - start
                matrices[(backend, fmt)] = matrix
                timing = benchmark_call(lambda m=matrix, b=backend: spatial_lag(m, y, b), backend)
                output.append(
                    {
                        "operation": "inverse_distance_and_lag",
                        "n": n,
                        "backend": backend,
                        "format": fmt,
                        "construction_total_seconds": construction,
                        **diagnostics.to_dict(),
                        **timing.to_dict(),
                    }
                )
            dense = matrices[(backend, "dense")]
            timing = benchmark_call(
                lambda m=dense, b=backend: Moran(y, m, 999, 12345, b), backend, 3
            )
            output.append(
                {
                    "operation": "global_moran_999",
                    "n": n,
                    "backend": backend,
                    "format": "dense",
                    **timing.to_dict(),
                }
            )
        if gpu_available():
            start = time.perf_counter()
            native = to_backend(matrices[("cpu", "dense")], "gpu")
            synchronize("gpu")
            output.append(
                {
                    "operation": "host_to_device",
                    "n": n,
                    "backend": "gpu",
                    "format": "dense",
                    "seconds": time.perf_counter() - start,
                }
            )
            del native
    target = Path("benchmarks/results")
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "platform": platform.platform(),
        "seed": 12345,
        "gpu": gpu_available(),
        "results": output,
    }
    (target / "mvp_benchmarks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    columns = sorted({key for row in output for key in row})
    with (target / "mvp_benchmarks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
