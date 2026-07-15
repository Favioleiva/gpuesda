# GPU Powered ESDA

`gpu_esda` is an independent CPU/GPU backend for exploratory spatial data analysis. It implements inverse-distance weights, dense/CSR normalization, spatial lag, Global Moran I, Local Moran/LISA, conditional permutations, Benjamini–Hochberg FDR, and observation-first variable/panel inputs. PySAL remains the correctness reference and is never modified.

## Install

```powershell
python -m pip install -e .
python scripts/diagnose_environment.py
```

After diagnostics confirm a CUDA 13-compatible driver, install the tested GPU stack:

```powershell
python -m pip install -e ".[gpu,dev]"
```

For CUDA 12, install matching `cupy-cuda12x[ctk]` rather than the CUDA 13 extra. `backend="auto"` probes a real matrix kernel and falls back to CPU; explicit `backend="gpu"` raises a clear error if the complete runtime is unusable.

## Quick start

```python
import numpy as np
from gpu_esda import inverse_distance_weights, Moran, MoranLocal, spatial_lag

coordinates = np.random.default_rng(12345).random((500, 2))
y = np.random.default_rng(54321).normal(size=500)

weights = inverse_distance_weights(
    coordinates, power=2, row_standardize=True,
    output_format="auto", backend="auto",
)
lag = spatial_lag(weights, y, backend="auto")
global_result = Moran(y, weights, permutations=999, seed=12345, backend="auto")
local_result = MoranLocal(y, weights, permutations=999, seed=12345, fdr=True, backend="auto")

print(global_result.summary())
print(local_result.to_frame().head())
```

Shapes are observation-first: `(n,)`, `(n, columns)`, or `(n, variables, periods)`. Use `moran_panel` and `local_moran_panel` for multi-column results. Result objects expose summaries, dictionaries, DataFrames, metadata, backend/dtype, and timings.

## Reproduce

```powershell
python -m pytest -q
python -m ruff check src tests scripts benchmarks
python scripts/run_validation.py
python scripts/run_cpu_baseline.py
python scripts/run_benchmarks.py
```

Columbus crime data are loaded from `libpysal.examples`; no dataset is copied here. Synthetic benchmarks use fixed seeds. JSON/CSV outputs are generated under `results/` and `benchmarks/results/`.

## Current evidence

On the recorded RTX 3060 at `n=1,793`, a nearly complete d^-2 matrix is better dense than CSR: 25.72 MB versus estimated 38.56 MB, with warmed lag medians of 0.474 ms GPU and 0.888 ms CPU. The complete 999-permutation Global Moran workflow was 1.23× faster on GPU. A fresh dense transfer took 9.62 ms, so retaining data on-device matters more than accelerating one isolated lag.

Observed Columbus results agree with PySAL to floating-point tolerance with zero quadrant mismatches. CPU and GPU permutation streams are reproducible but not identical to ESDA's NumPy/Numba streams; pseudo-p values are statistically rather than bitwise equivalent.

See the [methodology](docs/methodology.md), [PySAL audit](docs/pysal_esda_audit.md), [benchmark methodology](docs/benchmark_methodology.md), [benchmark report](docs/benchmark_report.md), and [status](STATUS.md).

## Validation policy

NaN and infinity default to explicit `raise` and are never converted to zero. Dropping observations requires callers to subset values and weights together. Constant variables and duplicate coordinates are rejected; islands remain zero rows. Float64 is the default. Float32 is supported at `atol=1e-5, rtol=1e-4` but is not recommended without workload-specific validation.

## License

Project code is BSD-3-Clause. Columbus is accessed from the user's PySAL installation and is not redistributed.
