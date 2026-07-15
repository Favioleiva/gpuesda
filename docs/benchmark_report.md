# MVP Benchmark Report

## Test system and scope

Results were measured on Windows 11, Python 3.12.6, an Intel 10-core/20-thread CPU, 64 GiB RAM, NVIDIA GeForce RTX 3060 (12 GiB), driver 591.86, CuPy 14.1.1 with CUDA 13 component wheels, NumPy 2.4.4, and SciPy 1.16.3. Fixed seed: 12345. Machine-readable results are generated under `benchmarks/results/` and ignored by Git as reproducible host outputs.

This is an MVP benchmark, not a cross-hardware study. It measures `n=500` and the target-relevant `n=1,793`. Larger planned sizes and broad variable sweeps remain future measurements.

## Correctness

On the 49-observation Columbus dataset, CPU lag, Global Moran I, and Local Moran I matched PySAL exactly. GPU maximum absolute differences were `7.11e-15` for lag and `6.66e-16` for Local Moran; Global Moran matched exactly and all quadrant codes matched. At 999 permutations, mean absolute Local pseudo-p differences were 0.00737 (CPU) and 0.00765 (GPU), consistent with different reproducible RNG streams. Both project backends completed 9,999 Global and conditional Local permutations, returned global pseudo-p 0.0001, and classified 20 local observations significant before FDR.

## Representative performance

| Operation at n=1,793 | CPU | GPU | CPU/GPU |
|---|---:|---:|---:|
| Full d^-2 dense construction, end-to-end | 131.9 ms | 97.7 ms | 1.35× |
| Dense lag, warmed median | 0.888 ms | 0.474 ms | 1.87× |
| CSR lag, warmed median | 3.863 ms | 4.541 ms | 0.85× |
| Global Moran, 999 permutations, warmed median | 267.7 ms | 218.1 ms | 1.23× |
| Dense host-to-device transfer | — | 9.62 ms | — |

The matrix density was 0.99944. Dense storage was 25.72 MB versus estimated CSR storage of 38.56 MB. GPU CSR construction/normalization was dramatically slower—a useful negative result. At `n=500`, startup and transfer overhead often outweighed GPU compute benefits.

## Conclusions

Repeated dense lag and batched permutations are the clearest acceleration targets. Geometry topology, I/O, plotting, and current FDR should remain on CPU. Dense decisively beats CSR for nearly complete inverse-distance matrices. Thresholded/KNN matrices should normally use CSR, though this MVP does not claim a speedup without additional scaling runs.

The measured break-even is workload-dependent: overhead dominates around 500 observations, while resident dense work benefits at 1,793. A fresh 9.62 ms transfer is much larger than either sub-millisecond lag, so keeping weights and panels resident matters. Observed-only Moran statistics are too small to justify transfer by themselves.

Float64 passed strict equivalence and remains the default. Float32 passes the documented tolerance tests but lacks enough classification benchmarking for a default recommendation. Observed statistics are numerically equivalent; permutation p-values are statistically rather than sequence-exactly equivalent across NumPy/Numba and CuPy RNGs.

Remaining bottlenecks include random ordering generation, Python-level panel result assembly, host/device summary conversions, and sparse setup when the graph is unsuitable for sparse storage. The next measurements should cover T4/L4/A100, feasible 5k–25k threshold/KNN cases, 10/100 columns resident on device, and isolated Local Moran timings.

## Raster extension: national Black Marble run

The raster measurements use the same RTX 3060, CuPy stack, and synchronized CUDA timing. They cover the complete private Black Marble Peru dataset for 2024-03-21: 6,111,958 occupied cells reconstructed into a 4,395 × 3,042 rectangle. All spatial weights are implicit stencils; no national weights matrix exists.

| National operation | Measured result |
|---|---:|
| Parquet read | 0.207 s |
| Read plus raster reconstruction | 7.143 s |
| Host-to-device transfer | 0.021 s |
| Complete observed script: 2 transforms × 3 stencils, figures and export | 17.854 s |
| Peak process RSS | 1.585 GB |
| Peak CuPy used pool | 548.2 MB |
| Peak CuPy reserved pool | 1.136 GB |
| Queen `log1p(NTL)` observed standardization | 0.0081 s |
| Queen `log1p(NTL)` observed lag | 0.0169 s |
| Queen `log1p(NTL)` Global reduction | 0.0024 s |
| Queen `log1p(NTL)` Local Moran | 0.0099 s |
| Queen `log1p(NTL)` 99 global permutations | 4.394 s |

Observed Global Moran I was:

| Transform | Rook | Queen | `d^-2`, radius 2 |
|---|---:|---:|---:|
| NTL | 0.3195941370 | 0.2122286480 | 0.2139488041 |
| `log1p(NTL)` | 0.8838645194 | 0.8645893256 | 0.8618631192 |

The national 99-permutation Queen global smoke run on `log1p(NTL)` produced `p_sim=0.01`, simulated mean `4.08e-06`, and simulated standard deviation `1.88e-04`. This validates the streaming path and is not presented as production inference. National conditional-local 99/999 inference is assigned to notebook 06 on A100 hardware, after its mandatory small-window CPU/GPU check.

The apparently small stencil kernel times are plausible because each pass is linear in the rectangular storage and uses only 4–12 shifted array operations. The 17.854 s total is dominated by reconstruction, compilation/warm-up, plotting, host transfers, and Parquet export. These measurements should not be generalized to A100 until the notebook benchmark table has been produced there.
