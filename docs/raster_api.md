# Raster ESDA API

The raster engine represents weights as a validity mask plus local offsets; it never materializes an `n × n` matrix. The vector matrix/CSR engine remains unchanged.

```python
from gpu_esda import BlackMarbleRaster, RasterWeights, moran_global, moran_local

raster = BlackMarbleRaster.from_parquet("grid.parquet", "daily.parquet", value_column="ntl")
weights = RasterWeights.queen(raster.mask, normalization="row", backend="gpu")
global_result = moran_global(raster.values, weights)
local_result = moran_local(
    raster.values,
    weights,
    permutations=99,
    seed=12345,
    fdr=True,
)
```

Available stencils are Rook, Queen, and circular inverse-distance-squared with configurable integer radius. For offset `(dr, dc)`, `d^-2` uses `1/(dr²+dc²)` and excludes the center.

Row normalization is local to each focal cell: both numerator and denominator use only valid neighbors. Coast, holes, and NoData are excluded; an NTL value of zero remains valid. A valid cell with no valid neighbors is explicitly marked as an island.

The default raster storage dtype is float32 while global reductions use float64. Set `dtype="float64"` on `RasterWeights` and load values as float64 for validation.

`MatrixWeightsOperator` exposes existing dense/CSR matrices through the same `apply(values)` interface. Generic `moran_global` and `moran_local` accept either operator.

## Constructors

```python
rook = RasterWeights.rook(mask, normalization="row", backend="cpu")
queen = RasterWeights.queen(mask, normalization="row", backend="gpu")
d2 = RasterWeights.inverse_distance(
    mask,
    radius=2,
    normalization="row",
    backend="gpu",
)
```

The inverse-distance stencil includes every integer offset inside the configured circular radius and gives `(dr, dc)` weight `1 / (dr² + dc²)`. The center is excluded. A focal cell with no valid neighbors is reported through the result island mask and receives zero lag.

## Inference

`moran_global(..., permutations=R)` performs unconditional global shuffles. `moran_local(..., permutations=R)` keeps each focal value fixed and samples neighbor values without replacement from all other valid cells. The local method follows the conditional-randomization design used by ESDA, but GPU and PySAL RNG streams are not bitwise identical.

Permutation batches and local focal-cell chunks are bounded independently. `permutation_batch=None` selects a conservative size from available VRAM. If a CuPy OOM occurs, the engine frees cached blocks and retries with half the batch. Only extreme counts, sums, sums of squares, and extrema are retained; simulated `n × R` matrices are never stored.

Important keyword arguments are:

- `permutations`: `0` for observed-only, `99` for smoke, or `999` for production;
- `seed`: reproducible backend RNG seed;
- `permutation_batch`: optional explicit permutation batch size;
- `cell_chunk`: conditional-local focal cells processed at once;
- `fdr=True` and `alpha`: Benjamini–Hochberg correction and threshold.

Local results expose `local_i`, `spatial_lag`, `quadrant`, `p_sim`, `p_fdr`, `significant`, and `islands`. `write_local_parquet` gathers occupied raster positions in `cell_id` order and writes the required columns without exporting structural holes.

## Black Marble reconstruction

`BlackMarbleRaster.from_parquet` reads the grid and daily files, validates unique one-to-one `cell_id`, aligns the daily values if necessary, and reconstructs arrays directly from `grid_row` and `grid_col`. Missing daily values remain invalid; numeric zero is not NoData. `raster.log1p()` preserves the same mask and mapping.

The complete audit and local runner are available in `scripts/audit_blackmarble_grid.py` and `scripts/run_blackmarble_observed.py`. Reusable `save_figure` writes figures with `dpi=200, bbox_inches="tight"` before optional display.

## Project post-processing helpers

```python
from scripts.blackmarble_postprocessing import (
    inferential_cluster_mask,
    lisa_class_counts,
    moran_scatter_line,
)

clusters = inferential_cluster_mask(valid, island, significant, quadrant)
counts = lisa_class_counts(quadrant, clusters)
x_line, wy_line = moran_scatter_line(global_result.I, -3.0, 4.0)
```

These pure NumPy helpers are project-level post-processing, not additions to the unchanged `gpu_esda 0.2.0` wheel. `inferential_cluster_mask` formalizes `valid & ~island & significant & quadrant∈{1,2,3,4}`. It prevents the raw FDR boolean from classifying NoData or islands. `lisa_class_counts` returns auditable HH/LH/LL/HL totals for any selection. `moran_scatter_line` returns the canonical zero-intercept `WY = I × Y` reference line.
