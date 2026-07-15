"""Download/read and audit the Black Marble Peru grid topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ID = "faviolc/ESDAGPU"
GRID_FILE = "data/grid/grid_peru_vnp46a2.parquet"
DAILY_FILE = "data/daily/year=2024/month=03/blackmarble_peru_2024-03-21.parquet"
EXPECTED_ROWS = 6_111_958
GRID_COLUMNS = [
    "cell_id",
    "tile_id",
    "tile_h",
    "tile_v",
    "row",
    "col",
    "grid_row",
    "grid_col",
    "lon",
    "lat",
]
DAILY_COLUMNS = [
    "cell_id",
    "ntl",
    "mandatory_quality_flag",
    "latest_high_quality_retrieval",
]


def _download(filename: str, local_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def audit(grid_path: Path, daily_path: Path) -> dict[str, Any]:
    grid_schema = pq.read_schema(grid_path)
    daily_schema = pq.read_schema(daily_path)
    missing_grid = sorted(set(GRID_COLUMNS) - set(grid_schema.names))
    missing_daily = sorted(set(DAILY_COLUMNS) - set(daily_schema.names))
    if missing_grid or missing_daily:
        raise ValueError(f"missing columns: grid={missing_grid}, daily={missing_daily}")

    grid = pq.read_table(grid_path, columns=GRID_COLUMNS).to_pandas()
    daily_ids = pq.read_table(daily_path, columns=["cell_id"]).column("cell_id").to_numpy()
    n = len(grid)
    grid_ids = grid.cell_id.to_numpy()
    rows = grid.grid_row.to_numpy(dtype=np.int64)
    cols = grid.grid_col.to_numpy(dtype=np.int64)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height, width = r1 - r0 + 1, c1 - c0 + 1
    rectangular_cells = height * width
    linear = (rows - r0) * width + (cols - c0)
    unique_positions = np.unique(linear).size

    # Integer tile coding keeps national occupancy diagnostics compact.
    tile_codes, tile_names = grid.tile_id.factorize(sort=True)
    occupancy = np.zeros(rectangular_cells, dtype=bool)
    occupancy[linear] = True
    tile_raster = np.full(rectangular_cells, -1, dtype=np.int16)
    tile_raster[linear] = tile_codes.astype(np.int16, copy=False)
    occupancy = occupancy.reshape(height, width)
    tile_raster = tile_raster.reshape(height, width)
    horizontal = occupancy[:, :-1] & occupancy[:, 1:]
    vertical = occupancy[:-1, :] & occupancy[1:, :]
    cross_h = horizontal & (tile_raster[:, :-1] != tile_raster[:, 1:])
    cross_v = vertical & (tile_raster[:-1, :] != tile_raster[1:, :])

    row_offset = (grid.grid_row - grid.row).groupby(grid.tile_id).nunique()
    col_offset = (grid.grid_col - grid.col).groupby(grid.tile_id).nunique()
    tile_summary = grid.groupby("tile_id", observed=True).agg(
        tile_h=("tile_h", "nunique"),
        tile_v=("tile_v", "nunique"),
        row_min=("row", "min"),
        row_max=("row", "max"),
        col_min=("col", "min"),
        col_max=("col", "max"),
        grid_row_min=("grid_row", "min"),
        grid_row_max=("grid_row", "max"),
        grid_col_min=("grid_col", "min"),
        grid_col_max=("grid_col", "max"),
        cells=("cell_id", "size"),
    )

    # Orientation is inferred from robust correlations, not assumed from MODIS conventions.
    sample = grid.iloc[:: max(1, n // 200_000)]
    row_lat_corr = float(sample.grid_row.corr(sample.lat, method="spearman"))
    col_lon_corr = float(sample.grid_col.corr(sample.lon, method="spearman"))
    orientation = {
        "rows": "north_to_south" if row_lat_corr < 0 else "south_to_north",
        "columns": "west_to_east" if col_lon_corr > 0 else "east_to_west",
        "grid_row_lat_spearman": row_lat_corr,
        "grid_col_lon_spearman": col_lon_corr,
    }

    sorted_grid_ids = np.sort(grid_ids)
    sorted_daily_ids = np.sort(daily_ids)
    one_to_one = bool(
        len(sorted_grid_ids) == len(sorted_daily_ids)
        and np.array_equal(sorted_grid_ids, sorted_daily_ids)
    )
    memory = {}
    for dtype in (np.float32, np.float64):
        itemsize = np.dtype(dtype).itemsize
        memory[np.dtype(dtype).name] = {
            "one_array_bytes": rectangular_cells * itemsize,
            "values_lag_local_i_bytes": rectangular_cells * itemsize * 3,
        }
    memory["boolean_mask_bytes"] = rectangular_cells
    memory["cell_id_int64_bytes"] = rectangular_cells * 8

    return {
        "repository": REPO_ID,
        "grid_file": str(grid_path),
        "daily_file": str(daily_path),
        "grid_sha256": _sha256(grid_path),
        "daily_sha256": _sha256(daily_path),
        "expected_rows": EXPECTED_ROWS,
        "grid_rows": n,
        "daily_rows": len(daily_ids),
        "grid_cell_id_unique": bool(not grid.cell_id.duplicated().any()),
        "daily_cell_id_unique": bool(np.unique(daily_ids).size == len(daily_ids)),
        "grid_position_unique": bool(unique_positions == n),
        "grid_daily_one_to_one": one_to_one,
        "bounds": {"grid_row_min": r0, "grid_row_max": r1, "grid_col_min": c0, "grid_col_max": c1},
        "shape": [height, width],
        "rectangular_positions": rectangular_cells,
        "valid_positions": int(unique_positions),
        "holes": int(rectangular_cells - unique_positions),
        "coverage_fraction": unique_positions / rectangular_cells,
        "orientation": orientation,
        "tiles": int(len(tile_names)),
        "tile_row_offset_constant": bool((row_offset == 1).all()),
        "tile_col_offset_constant": bool((col_offset == 1).all()),
        "tile_h_v_constant_within_tile": bool(
            (tile_summary.tile_h == 1).all() and (tile_summary.tile_v == 1).all()
        ),
        "cross_tile_rook_edges_horizontal": int(cross_h.sum()),
        "cross_tile_rook_edges_vertical": int(cross_v.sum()),
        "continuous_national_index": bool(unique_positions == n and one_to_one),
        "reconstruct_without_spatial_join": bool(unique_positions == n and one_to_one),
        "memory_estimates": memory,
        "tile_summary": [
            {"tile_id": str(index), **{key: _scalar(value) for key, value in row.items()}}
            for index, row in tile_summary.reset_index().set_index("tile_id").iterrows()
        ],
    }


def _report(result: dict[str, Any]) -> str:
    shape = result["shape"]
    return f"""# Black Marble Peru Grid Audit

## Verdict

- Continuous national integer index: **{result["continuous_national_index"]}**
- Reconstructable without a spatial join: **{result["reconstruct_without_spatial_join"]}**
- Grid/daily one-to-one by `cell_id`: **{result["grid_daily_one_to_one"]}**

## Dimensions and coverage

- Grid rows: {result["grid_rows"]:,}; daily rows: {result["daily_rows"]:,}
- Rectangular shape: {shape[0]:,} × {shape[1]:,}
- Valid positions: {result["valid_positions"]:,}; holes: {result["holes"]:,}
- Coverage: {result["coverage_fraction"]:.4%}
- Orientation: rows {result["orientation"]["rows"]}; columns {result["orientation"]["columns"]}

## Topology checks

- Unique `cell_id` in grid/daily: {result["grid_cell_id_unique"]} / {result["daily_cell_id_unique"]}
- Unique `(grid_row, grid_col)`: {result["grid_position_unique"]}
- Constant row/column tile offsets: {result["tile_row_offset_constant"]} / {result["tile_col_offset_constant"]}
- Cross-tile Rook edges: horizontal {result["cross_tile_rook_edges_horizontal"]:,}, vertical {result["cross_tile_rook_edges_vertical"]:,}

## Memory

- One float32 rectangular array: {result["memory_estimates"]["float32"]["one_array_bytes"] / 2**20:.1f} MiB
- Three float32 result arrays: {result["memory_estimates"]["float32"]["values_lag_local_i_bytes"] / 2**20:.1f} MiB
- One float64 rectangular array: {result["memory_estimates"]["float64"]["one_array_bytes"] / 2**20:.1f} MiB
- Boolean mask: {result["memory_estimates"]["boolean_mask_bytes"] / 2**20:.1f} MiB

The machine-readable audit, hashes, bounds, and per-tile summaries are in `results/blackmarble_grid_audit.json`. No topology is inferred from coordinates: stencil adjacency is permitted only because integer national positions are unique and tile offsets are consistent.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-path", type=Path)
    parser.add_argument("--daily-path", type=Path)
    parser.add_argument("--download-dir", type=Path, default=Path("results/blackmarble/input"))
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    grid_path = args.grid_path or _download(GRID_FILE, args.download_dir, token)
    daily_path = args.daily_path or _download(DAILY_FILE, args.download_dir, token)
    result = audit(grid_path, daily_path)
    Path("results").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    Path("results/blackmarble_grid_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    Path("docs/blackmarble_grid_audit.md").write_text(_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "grid_rows",
                    "daily_rows",
                    "shape",
                    "holes",
                    "tiles",
                    "grid_daily_one_to_one",
                    "continuous_national_index",
                )
            },
            indent=2,
        )
    )
    return 0 if result["continuous_national_index"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
