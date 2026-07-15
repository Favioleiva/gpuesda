# Raster ESDA API

The raster engine represents weights as a validity mask plus local offsets; it never materializes an `n × n` matrix. The vector matrix/CSR engine remains unchanged.

```python
from gpu_esda import BlackMarbleRaster, RasterWeights, moran_global, moran_local

raster = BlackMarbleRaster.from_parquet("grid.parquet", "daily.parquet", value_column="ntl")
weights = RasterWeights.queen(raster.mask, normalization="row", backend="gpu")
global_result = moran_global(raster.values, weights)
local_result = moran_local(raster.values, weights)
```

Available stencils are Rook, Queen, and circular inverse-distance-squared with configurable integer radius. For offset `(dr, dc)`, `d^-2` uses `1/(dr²+dc²)` and excludes the center.

Row normalization is local to each focal cell: both numerator and denominator use only valid neighbors. Coast, holes, and NoData are excluded; an NTL value of zero remains valid. A valid cell with no valid neighbors is explicitly marked as an island.

The default raster storage dtype is float32 while global reductions use float64. Set `dtype="float64"` on `RasterWeights` and load values as float64 for validation.

`MatrixWeightsOperator` exposes existing dense/CSR matrices through the same `apply(values)` interface. Generic `moran_global` and `moran_local` accept either operator.

