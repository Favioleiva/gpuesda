"""Run the safe national global-permutation smoke case on Black Marble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gpu_esda import BlackMarbleRaster, RasterWeights, moran_global
from gpu_esda.raster.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-path", type=Path, required=True)
    parser.add_argument("--daily-path", type=Path, required=True)
    parser.add_argument("--permutations", type=int, choices=(99, 999), default=99)
    args = parser.parse_args()
    raster = BlackMarbleRaster.from_parquet(args.grid_path, args.daily_path, dtype="float32")
    values = np.full(raster.values.shape, np.nan, dtype=np.float32)
    values[raster.mask] = np.log1p(raster.values[raster.mask])
    operator = RasterWeights.queen(raster.mask, backend="gpu")
    result = moran_global(values, operator, permutations=args.permutations, seed=12345)
    payload = {
        "date": "2024-03-21",
        "variable": "log1p(ntl)",
        "stencil": "queen",
        **result.to_dict(),
    }
    write_json(payload, "results/blackmarble/blackmarble_2024-03-21_global_99.json")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
