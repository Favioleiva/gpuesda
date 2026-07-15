# PySAL and ESDA Technical Audit

## Scope and versions

This audit targets `libpysal 4.15.0` and `esda 2.10.0`, installed without modifying either package. Source files inspected were `libpysal/weights/{contiguity,distance,weights,spatial_lag}.py`, `esda/moran.py`, `esda/crand.py`, and `esda/significance.py`.

## Component findings

| Component | API and representation | Algorithm and behavior | GPU disposition |
|---|---|---|---|
| `Queen`, `Rook` | Geometry iterable/dataframe to `W`; neighbor/weight dictionaries, lazily cached SciPy CSR | CPU geometry topology builds adjacency from shared vertices/edges. No diagonal unless supplied; islands have empty rows and warnings. | Keep topology on Shapely/GeoPandas CPU; transfer the final CSR only. |
| `KNN` | coordinates/KDTree to `W`; directed, exactly `k` neighbors absent ties | KDTree query, with coincident points requiring larger `k` or jitter. Binary original weights. | Keep reference construction on CPU; a blocked GPU distance/top-k implementation is possible only at large `n`. |
| `DistanceBand` | coordinates, threshold, `binary`, `alpha`; dictionary `W` via `WSP` | KDTree sparse distance matrix or dense SciPy distances. Nonbinary uses `distance ** alpha`; diagonal infinities are set to zero. Islands are retained. | Blocked CuPy distances and thresholding map well; avoid a full pairwise allocation when memory estimates fail. |
| `W` | neighbor and weight dictionaries, IDs, transformations | Builds CSR through Python lists. `R` divides each non-island row by its sum; `B` sets stored links to one; `D` global-standardizes; `V` variance-stabilizes. Cached `s0`, `s1`, and `s2` support analytical moments. | Do numerical transforms directly on dense/CSR arrays. Preserve zero rows. Dictionary/API conversion remains CPU. |
| `WSP` | thin SciPy sparse wrapper plus optional ID order | Avoids dictionary overhead and exposes summary constants. | Natural interchange boundary; GPU uses `cupyx.scipy.sparse.csr_matrix`. |
| `lag_spatial` | `lag_spatial(w, y)`; vector or `n × columns` | Exactly `w.sparse @ y`; binary gives weighted sum and row-standardized gives neighbor average. | Dense `@` or CSR `@`; ideal reusable-weight GPU kernel. |
| `esda.Moran` | flattened `y`, transformed `W`, result attributes | Centers `z`; observed `I=(n/s0)(z'Wz)/(z'z)`. `EI=-1/(n-1)`. Normality and randomization variances use `s0/s1/s2` and kurtosis. Global permutations call `np.random.permutation(z)` once per replicate and use a directed/minimum-tail pseudo-p `(min(greater,R-greater)+1)/(R+1)`. Final public `z` uses population standard deviation. | Centering, lag, reductions, and batched permutations are GPU-suitable. Exact NumPy/CuPy RNG streams are not expected. |
| `esda.Moran_Local` | flattened `y`, `W`, permutations, `seed`; arrays per observation | Population-standardizes `z`. `I_i=(n-1)z_i(Wz)_i/sum(z²)`. Default codes: 1 HH, 2 LH, 3 LL, 4 HL. Conditional randomization holds focal `z_i` fixed; for each replicate it samples neighbor-cardinality values without replacement from the other `n-1` observations. ESDA generates shared sampled IDs with Numba `np.random.choice`, adjusts IDs around each focal observation, and applies that row's weights. Default legacy significance is directed. | Vectorize focal observations and batches on GPU with padded neighbor weights; avoid one Python loop per observation. RNG differs, so validate observed values exactly and inference statistically. |

## Statistical and numerical details

- `W.s0=sum(W)`. For arbitrary/asymmetric weights, `s1=0.5*sum((W+W.T)^2)` and `s2=sum_i(row_sum_i+col_sum_i)^2`.
- Self-neighbors are absent from normal constructors. ESDA conditional inference extracts any supplied diagonal, zeros it for shuffled neighbors, and keeps the self contribution fixed.
- Global simulation standard deviation uses population `ddof=0`; pseudo-p includes the observed statistic by adding one to numerator and denominator.
- Local conditional analytical moments follow Sokal (1998). The MVP reproduces observed statistics and simulated summaries; inference is the primary compatibility target.
- Islands have lag zero. Row normalization leaves zero rows unchanged. Local island output is consequently zero unless an explicit island weight policy is introduced.
- Geometry and plotting remain CPU-only. Graph construction is distinct from matrix representation, which is distinct from numerical statistics and inference.

## Complexity and memory

- Contiguity and KDTree graph creation are CPU/topology tasks; their exact complexity depends on the spatial index.
- Full inverse distance is `O(n²)` compute and memory. Threshold/KNN variants can be `O(nnz)` after neighbor selection; blocked construction caps temporary memory.
- Lag is `O(n²m)` dense or `O(nnz*m)` sparse for `m` columns.
- Global permutations are `O(R * lag_cost)`. Conditional Local Moran is approximately `O(R * nnz)` with reusable sampled indices.
- PySAL dictionary construction and global permutation list comprehensions are Python-level overhead. ESDA Local Moran moves its hot conditional loop into Numba.

## Replication risks

- CuPy and NumPy/Numba do not promise identical random streams; identical seeds mean reproducible results within a backend, not bitwise cross-backend permutations.
- Reduction order can change low bits. Validation tolerances are float64 `atol=1e-10, rtol=1e-8` and float32 `atol=1e-5, rtol=1e-4` for deterministic operations.
- Float32 loses precision for extreme distance ratios, row sums, and near-threshold classifications; float64 remains the default.
- Conditional pseudo-p-values have Monte Carlo granularity `1/(R+1)`. Cross-engine comparisons must account for simulation uncertainty rather than assert exact p-values.

## Smallest valuable numerical core

The chosen core is: independent inverse-distance construction and normalization, reusable dense/CSR lag, Moran reductions, and batched permutation evaluation. Geometry topology, dataset I/O, ID management, FDR (unless profiled as material), and visualization remain on CPU.

