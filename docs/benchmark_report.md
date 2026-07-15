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

