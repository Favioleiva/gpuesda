# CPU–GPU Functional Mapping

| Operation | CPU/reference | GPU candidate | Benefit | Difficulty | Numerical/memory risk | Fallback |
|---|---|---|---|---|---|---|
| Array operations | NumPy | CuPy | High for large batched arrays | Low | reduction order; transfers | NumPy |
| CSR operations | SciPy sparse | `cupyx.scipy.sparse` | High for sparse repeated lag | Medium | index dtype, sparse conversion copy | SciPy CSR |
| Spatial lag | SciPy matrix multiply | CuPy dense/CSR `@` | High when weights/data stay resident | Low | dense quadratic memory | SciPy `@` |
| Pairwise distances | SciPy/NumPy | blocked CuPy broadcasting | High for full inverse-distance | Medium | `O(n²)` memory, duplicate zeros | blocked NumPy/SciPy |
| KNN search | KDTree/PySAL | GPU distance blocks/top-k | Workload-dependent | High | ties and full-distance storage | PySAL/scikit-learn CPU |
| Permutation generation | NumPy/Numba | CuPy RNG + batched indices | Very high at large `R` | High | RNG sequence differs; batch memory | NumPy generator |
| Moran reductions | NumPy | CuPy reductions | Medium alone, high when fused with lag | Low | summation order | NumPy |
| Local Moran | NumPy/Numba | CuPy elementwise + padded-neighbor batches | High at large `R` | High | heterogeneous cardinality | vectorized NumPy/Numba reference |
| FDR | statsmodels | CuPy sort/scatter | Low for typical `n` | Medium | stable ordering and ties | statsmodels/NumPy |
| Geometry topology | GeoPandas/Shapely | none initially | No demonstrated benefit | High | semantic/topological robustness | CPU only |
| Mapping | GeoPandas/Matplotlib | none | No numerical bottleneck | High | ecosystem mismatch | CPU only |

`auto` chooses dense when estimated dense storage is no worse than CSR or density is at least 0.35 and memory is safe. Sparse graph families normally remain CSR. Every explicit GPU request is validated; `auto` falls back to CPU.

