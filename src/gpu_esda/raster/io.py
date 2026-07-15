"""Machine-readable Black Marble result export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def write_local_parquet(result: Any, raster: Any, path: str | Path) -> Path:
    """Export one row per source cell without materializing string labels in Python."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    codes = raster.gather(result.quadrant_code).astype(np.int8, copy=False)
    valid = raster.gather(result.mask).astype(bool, copy=False)
    islands = raster.gather(result.island).astype(bool, copy=False)
    output_codes = np.where(~valid, 0, np.where(islands, 5, codes)).astype(np.int8)
    quadrant = pa.DictionaryArray.from_arrays(
        pa.array(output_codes), pa.array(["not valid", "HH", "LH", "LL", "HL", "island"])
    )
    n = len(raster.valid_cell_ids)
    arrays: dict[str, Any] = {
        "cell_id": pa.array(raster.valid_cell_ids),
        "local_i": pa.array(raster.gather(result.local_i).astype(np.float32, copy=False)),
        "spatial_lag": pa.array(raster.gather(result.spatial_lag).astype(np.float32, copy=False)),
        "quadrant": quadrant,
        "p_sim": pa.nulls(n, type=pa.float32())
        if result.p_sim is None
        else pa.array(raster.gather(result.p_sim)),
        "p_fdr": pa.nulls(n, type=pa.float32())
        if result.p_fdr is None
        else pa.array(raster.gather(result.p_fdr)),
        "significant": pa.nulls(n, type=pa.bool_())
        if result.significant is None
        else pa.array(raster.gather(result.significant)),
    }
    pq.write_table(pa.table(arrays), target, compression="zstd")
    return target
