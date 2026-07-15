# Methodology

Correctness is established in float64 against `libpysal`/`esda` before measuring speed. Deterministic weights, normalization, lag, observed Moran statistics, and quadrant codes use strict tolerances. Permutation algorithms use fixed seeds; CPU/GPU RNG streams are documented as backend-reproducible but not identical, so p-values are compared with Monte Carlo uncertainty rather than bitwise equality.

Missing values default to `raise`. `drop` is supported for variable arrays only when the caller also subsets weights explicitly; implicit weight reindexing is prohibited. Constant/non-finite vectors, duplicate coordinates, incompatible dimensions, and negative thresholds produce clear errors. Islands retain zero rows.

Input shape convention is observation-first: `(n,)`, `(n, columns)`, or `(n, variables, periods)`. Trailing axes are flattened for a single matrix operation and restored on return. Float64 is the default; float32 is opt-in.

Synthetic data use `numpy.random.default_rng(seed)` with fixed defaults and cover uniform, clustered, and grid coordinates; independent and spatially smoothed values; variables, periods, missing values, islands, and intentional duplicates.

## Raster engine

Raster topology is an implicit stencil over verified integer grid coordinates. No national weight matrix is constructed. Each shifted contribution is gated by both focal and neighbor masks; row denominators are recomputed from the same valid pairs. This prevents coast, holes, and missing retrievals from becoming zero-valued observations. Genuine NTL zero is retained.

The Black Marble audit established a 4,395 × 3,042 national rectangle with 6,111,958 occupied positions and 7,257,632 holes. Integer positions are unique, tile offsets are constant, and cross-tile adjacency exists in the national index, so reconstruction requires no spatial join. See `docs/blackmarble_grid_audit.md`.

### Raster permutation inference

Global inference shuffles standardized values only among valid cells, scatters each small batch into the masked rectangle, applies the same stencil, and updates scalar moments/extreme counts. No complete simulation vector is retained.

Local inference follows ESDA's conditional null method: each focal value is fixed and, per replicate, a number of values equal to the maximum stencil cardinality is sampled without replacement from all other valid cells. Actual focal neighbor weights (including coast-adjusted row normalization) zero unused slots. Focal cells are processed in chunks and only p-value/moment accumulators are retained. NumPy and CuPy streams are reproducible within backend but not sequence-identical to ESDA/Numba. The default directed pseudo-p preserves current ESDA legacy behavior.

Automatic batches use a conservative fraction of free VRAM. A CuPy OOM halves the current batch and retries; no unsafe national allocation of shape `n_cells × n_permutations` occurs. Production 999-permutation Local Moran is intended for A100-class hardware; the RTX 3060 local run is limited to observed statistics and safe global smoke inference.
