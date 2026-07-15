"""Black Marble Parquet reconstruction without spatial joins."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass
class BlackMarbleRaster:
    values: np.ndarray
    mask: np.ndarray
    structural_mask: np.ndarray
    cell_ids: np.ndarray
    valid_cell_ids: np.ndarray
    valid_rows: np.ndarray
    valid_cols: np.ndarray
    origin: tuple[int, int]
    value_column: str = "ntl"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parquet(
        cls,
        grid_path: str | Path,
        daily_path: str | Path,
        value_column: str = "ntl",
        dtype: str = "float32",
    ) -> "BlackMarbleRaster":
        started = time.perf_counter()
        grid = pq.read_table(grid_path, columns=["cell_id", "grid_row", "grid_col"])
        daily = pq.read_table(daily_path, columns=["cell_id", value_column])
        grid_ids = grid.column("cell_id").to_numpy()
        daily_ids = daily.column("cell_id").to_numpy()
        if len(grid_ids) != len(daily_ids) or np.unique(grid_ids).size != len(grid_ids):
            raise ValueError("grid and daily must contain unique, equally sized cell_id arrays")
        values = daily.column(value_column).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        if not np.array_equal(grid_ids, daily_ids):
            order_grid = np.argsort(grid_ids)
            order_daily = np.argsort(daily_ids)
            if not np.array_equal(grid_ids[order_grid], daily_ids[order_daily]):
                raise ValueError("grid and daily cell_id sets are not one-to-one")
            aligned = np.empty_like(values)
            aligned[order_grid] = values[order_daily]
            values = aligned
        rows_abs = grid.column("grid_row").to_numpy().astype(np.int64, copy=False)
        cols_abs = grid.column("grid_col").to_numpy().astype(np.int64, copy=False)
        row_min, col_min = int(rows_abs.min()), int(cols_abs.min())
        rows = rows_abs - row_min
        cols = cols_abs - col_min
        shape = (int(rows.max()) + 1, int(cols.max()) + 1)
        linear = rows * shape[1] + cols
        if np.unique(linear).size != len(linear):
            raise ValueError("grid_row/grid_col positions are not unique")
        structural = np.zeros(shape, dtype=bool)
        structural[rows, cols] = True
        valid = np.isfinite(values)
        mask = np.zeros(shape, dtype=bool)
        mask[rows[valid], cols[valid]] = True
        raster = np.full(shape, np.nan, dtype=dtype)
        raster[rows[valid], cols[valid]] = values[valid]
        cell_ids = np.full(shape, -1, dtype=np.int64)
        cell_ids[rows, cols] = grid_ids
        return cls(
            raster,
            mask,
            structural,
            cell_ids,
            grid_ids,
            rows,
            cols,
            (row_min, col_min),
            value_column,
            {
                "source_rows": len(grid_ids),
                "valid_values": int(valid.sum()),
                "missing_values": int((~valid).sum()),
                "reconstruction_seconds": time.perf_counter() - started,
            },
        )

    def transformed(self, transform: str) -> "BlackMarbleRaster":
        if transform != "log1p":
            raise ValueError("supported transform is 'log1p'")
        if np.any(self.values[self.mask] < 0):
            raise ValueError("log1p requires non-negative valid values")
        values = self.values.copy()
        values[self.mask] = np.log1p(values[self.mask])
        return BlackMarbleRaster(
            values,
            self.mask.copy(),
            self.structural_mask.copy(),
            self.cell_ids.copy(),
            self.valid_cell_ids.copy(),
            self.valid_rows.copy(),
            self.valid_cols.copy(),
            self.origin,
            f"log1p({self.value_column})",
            {**self.metadata, "transform": "log1p"},
        )

    def gather(self, array: Any) -> np.ndarray:
        """Gather a raster result in original `cell_id` order."""
        from ..backend import to_numpy

        host = np.asarray(to_numpy(array))
        return host[self.valid_rows, self.valid_cols]
