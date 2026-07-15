"""Run observed national Black Marble raster ESDA on the local GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import psutil

from gpu_esda import BlackMarbleRaster, RasterWeights, gpu_available, moran_observed
from gpu_esda.backend import _cupy, to_numpy
from gpu_esda.raster.io import write_json, write_local_parquet
from gpu_esda.visualization import save_figure


def _figure(values: np.ndarray, title: str, filename: str, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, interpolation="nearest")
    ax.set(title=title, xlabel="grid_col", ylabel="grid_row")
    fig.colorbar(image, ax=ax, shrink=0.7)
    save_figure(fig, filename)
    plt.close(fig)


def _memory(cp: object, process: psutil.Process) -> dict[str, int]:
    free, total = cp.cuda.Device().mem_info
    pool = cp.get_default_memory_pool()
    return {
        "cpu_rss_bytes": process.memory_info().rss,
        "gpu_device_used_bytes": int(total - free),
        "gpu_pool_used_bytes": int(pool.used_bytes()),
        "gpu_pool_total_bytes": int(pool.total_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-path", type=Path, required=True)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()
    if not gpu_available():
        raise RuntimeError("the observed national run requires a usable CuPy GPU")
    cp = _cupy()
    process = psutil.Process()
    total_started = time.perf_counter()
    raster = BlackMarbleRaster.from_parquet(args.grid_path, args.daily_path, dtype="float32")
    records: list[dict[str, object]] = []
    memory_samples = [_memory(cp, process)]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(raster.structural_mask, cmap="gray", interpolation="nearest")
    ax.set(title="Black Marble Peru grid coverage", xlabel="grid_col", ylabel="grid_row")
    save_figure(fig, "grid_coverage.png")
    plt.close(fig)
    _figure(raster.values, "NTL raw — 2024-03-21", "ntl_raw_map.png")
    log_host = np.full(raster.values.shape, np.nan, dtype=np.float32)
    log_host[raster.mask] = np.log1p(raster.values[raster.mask])
    _figure(log_host, "log1p(NTL) — 2024-03-21", "ntl_log1p_map.png")

    transfer_started = time.perf_counter()
    values_gpu = cp.asarray(raster.values, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    host_to_device = time.perf_counter() - transfer_started
    memory_samples.append(_memory(cp, process))
    exported_local = None
    exported_codes = None

    for transform in ("ntl", "log1p"):
        transform_started = time.perf_counter()
        native = values_gpu if transform == "ntl" else cp.log1p(values_gpu)
        cp.cuda.Stream.null.synchronize()
        transform_seconds = time.perf_counter() - transform_started
        for stencil in ("rook", "queen", f"inverse_distance_r{args.radius}"):
            operator_started = time.perf_counter()
            if stencil == "rook":
                operator = RasterWeights.rook(raster.mask, backend="gpu")
            elif stencil == "queen":
                operator = RasterWeights.queen(raster.mask, backend="gpu")
            else:
                operator = RasterWeights.inverse_distance(
                    raster.mask, radius=args.radius, backend="gpu"
                )
            cp.cuda.Stream.null.synchronize()
            operator_seconds = time.perf_counter() - operator_started
            global_result, local_result = moran_observed(native, operator)
            counts = cp.bincount(
                local_result.quadrant_code[local_result.mask].astype(cp.int32), minlength=5
            ).get()
            islands = int(local_result.island.sum().item())
            record = {
                "date": "2024-03-21",
                "variable": transform,
                "stencil": stencil,
                "n": global_result.n,
                "moran_i": global_result.I,
                "expected_i": global_result.EI,
                "islands": islands,
                "quadrant_counts": {
                    "HH": int(counts[1]),
                    "LH": int(counts[2]),
                    "LL": int(counts[3]),
                    "HL": int(counts[4]),
                },
                "timings_seconds": {
                    "host_to_device_once": host_to_device,
                    "transformation": transform_seconds,
                    "operator_construction": operator_seconds,
                    **global_result.timings,
                    "local_statistic": local_result.timings["local_statistic"],
                },
            }
            records.append(record)
            memory_samples.append(_memory(cp, process))
            if transform == "log1p" and stencil == "queen":
                result_transfer_started = time.perf_counter()
                exported_local = SimpleNamespace(
                    local_i=to_numpy(local_result.local_i),
                    spatial_lag=to_numpy(local_result.spatial_lag),
                    quadrant_code=to_numpy(local_result.quadrant_code),
                    island=to_numpy(local_result.island),
                    mask=to_numpy(local_result.mask),
                    p_sim=None,
                    p_fdr=None,
                    significant=None,
                )
                cp.cuda.Stream.null.synchronize()
                record["timings_seconds"]["result_transfer"] = (
                    time.perf_counter() - result_transfer_started
                )
                exported_codes = exported_local.quadrant_code
            del global_result, local_result, operator
            cp.get_default_memory_pool().free_all_blocks()

    if exported_local is not None:
        write_local_parquet(
            exported_local,
            raster,
            "results/blackmarble/blackmarble_2024-03-21_local_queen_observed.parquet",
        )
        display_codes = np.where(raster.mask, exported_codes, np.nan)
        _figure(
            display_codes,
            "Local Moran quadrants — Queen, log1p(NTL)",
            "local_moran_queen.png",
            cmap="tab10",
        )
        _figure(
            display_codes,
            "LISA clusters — Queen, log1p(NTL)",
            "lisa_clusters_queen.png",
            cmap="tab10",
        )

    peak = {key: max(sample[key] for sample in memory_samples) for key in memory_samples[0]}
    payload = {
        "dataset": "faviolc/ESDAGPU",
        "date": "2024-03-21",
        "shape": list(raster.values.shape),
        "raster_metadata": raster.metadata,
        "peak_memory": peak,
        "total_seconds": time.perf_counter() - total_started,
        "results": records,
    }
    write_json(payload, "results/blackmarble/blackmarble_2024-03-21_global_observed.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([sample["gpu_pool_used_bytes"] / 2**20 for sample in memory_samples], marker="o")
    ax.set(title="GPU memory profile", xlabel="measurement stage", ylabel="CuPy used MiB")
    save_figure(fig, "gpu_memory_profile.png")
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
