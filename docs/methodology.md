# Methodology

Correctness is established in float64 against `libpysal`/`esda` before measuring speed. Deterministic weights, normalization, lag, observed Moran statistics, and quadrant codes use strict tolerances. Permutation algorithms use fixed seeds; CPU/GPU RNG streams are documented as backend-reproducible but not identical, so p-values are compared with Monte Carlo uncertainty rather than bitwise equality.

Missing values default to `raise`. `drop` is supported for variable arrays only when the caller also subsets weights explicitly; implicit weight reindexing is prohibited. Constant/non-finite vectors, duplicate coordinates, incompatible dimensions, and negative thresholds produce clear errors. Islands retain zero rows.

Input shape convention is observation-first: `(n,)`, `(n, columns)`, or `(n, variables, periods)`. Trailing axes are flattened for a single matrix operation and restored on return. Float64 is the default; float32 is opt-in.

Synthetic data use `numpy.random.default_rng(seed)` with fixed defaults and cover uniform, clustered, and grid coordinates; independent and spatially smoothed values; variables, periods, missing values, islands, and intentional duplicates.

