# Benchmark Methodology

Benchmarks record hardware/package metadata, seed, `n`, dtype, format, density, `nnz`, permutations, and process/GPU memory. CPU uses `time.perf_counter`. GPU measurements distinguish cold construction, host-to-device transfer, synchronized warmed computation, and device-to-host transfer. Each warmed measurement synchronizes before and after work and reports repetitions, median, min, max, mean, and population standard deviation.

Planned sizes are 500, 1,000, 1,793, 5,000, 10,000, and 25,000; permutation counts are 0, 99, 999, and 9,999; workloads cover 1, 10, and 100 columns plus panels. Full dense combinations that exceed conservative memory estimates are recorded as skipped. Speedup is CPU runtime divided by equivalent GPU runtime, never between non-equivalent outputs.

Weights compared are Queen, KNN, full inverse-distance-squared, thresholded inverse-distance-squared, and KNN-limited inverse-distance-squared. Dense and CSR are compared at `n=1,793` when a GPU is available.
