# GPU Powered ESDA Status

## Current state

- Phase: 5 — Performance
- Overall: in progress
- Next action: create notebooks, benchmark report, and final user documentation

## Updates

### 2026-07-15 23:23 JST — Repository and package scaffold initialized

- Completed task: initialized Git and created the specified directory skeleton and project-management files.
- Files created or modified: `.gitignore`, `pyproject.toml`, `environment.yml`, `PLAN.md`, `STATUS.md`; package, test, benchmark, notebook, documentation, script, and results directories.
- Commands executed: directory inspection, `git init`, Python/platform checks, `nvidia-smi`, `nvcc --version`, installed-package probe.
- Tests executed: none (implementation not yet present).
- Test results: not applicable.
- Benchmark results: none.
- Problems encountered: CUDA toolkit compiler is not on `PATH`; `libpysal`, `esda`, `numba`, and CuPy were not installed. NVIDIA driver 591.86 reports CUDA 13.1 compatibility and an RTX 3060 with 12 GiB VRAM.
- Unresolved questions: actual CuPy runtime compatibility will be verified after installing the CUDA 13 wheel.
- Next action: implement and run graceful environment diagnostics, then install missing dependencies.

### 2026-07-15 23:27 JST — Environment detected and dependencies configured

- Completed task: created graceful diagnostics, selected the CUDA 13 CuPy wheel based on driver compatibility, and installed the editable project with CPU, GPU, and development dependencies.
- Files created or modified: `scripts/diagnose_environment.py`, `pyproject.toml`, `environment.yml`, `STATUS.md`.
- Commands executed: `python scripts/diagnose_environment.py`; `python -m pip install -e ".[dev,gpu]"`; installed-source location probe.
- Tests executed: diagnostic execution and package import/version probes.
- Test results: CPU diagnostics pass; installed `libpysal 4.15.0`, `esda 2.10.0`, `numba 0.66.0`, and `cupy-cuda13x 14.1.1`. CuPy device initialization still requires a writable cache location and a follow-up runtime check.
- Benchmark results: none.
- Problems encountered: managed execution could not write CuPy's default user cache; diagnostic now redirects its cache to the temporary directory. A plain CuPy 14 wheel detects the device but does not supply CUDA math DLLs; the GPU extra now uses the official `[ctk]` component-wheel option. CuPy 14 supports SciPy through 1.16, so the package upper bound is explicit.
- Unresolved questions: CUDA matrix and sparse kernels require revalidation after installing component wheels.
- Next action: rerun diagnostics, inspect reference source, document the audit and CPU–GPU mapping.

### 2026-07-15 23:48 JST — Audit, numerical core, and validation completed

- Completed task: audited PySAL/ESDA, selected Columbus, implemented backend/weights/lag/Global Moran/Local Moran/conditional permutations/FDR/panels, added CPU reference helpers, and validated CPU/GPU behavior.
- Files created or modified: all modules under `src/gpu_esda`; eight requested test modules plus `conftest.py`; `docs/pysal_esda_audit.md`, `docs/cpu_gpu_mapping.md`, `docs/dataset_selection.md`, `docs/methodology.md`, `docs/benchmark_methodology.md`; validation/baseline scripts.
- Commands executed: source inspection with `rg`; CuPy `[ctk]` installation; editable install; smoke checks; `python scripts/run_validation.py`; `python scripts/run_cpu_baseline.py`; direct 9,999-permutation CPU/GPU run.
- Tests executed: `python -m pytest -q`; `python -m ruff check src tests scripts`.
- Test results: 28 passed; Ruff passed. Columbus deterministic comparison: CPU lag/global/local max differences 0; GPU lag `7.11e-15`, Local Moran `6.66e-16`; zero quadrant mismatches. At 999 permutations, mean absolute Local pseudo-p difference versus ESDA is 0.00737 CPU and 0.00765 GPU because the reproducible RNG streams differ.
- Benchmark results: both project backends completed Global and Local Moran with 9,999 permutations on Columbus; CPU 0.446 s total, GPU 0.321 s total, and both produced global pseudo-p 0.0001 and 20 uncorrected significant local observations.
- Problems encountered: plain CuPy wheel lacked CUDA math DLLs; resolved with official `cupy-cuda13x[ctk]`. Sparse all-island density initially divided by SciPy sparse `.size`; corrected to shape product.
- Unresolved questions: broader cross-hardware timing remains future work; current benchmark conclusions apply only to the recorded RTX 3060 system.
- Next action: run representative `n=500` and `n=1,793` benchmarks, then create notebooks/report/README.

### 2026-07-15 23:53 JST — Representative MVP benchmarks exported

- Completed task: benchmarked full inverse-distance-squared construction, dense/CSR lag, 999-permutation Global Moran, and host-to-device transfer at `n=500` and `n=1,793`.
- Files created or modified: benchmark entrypoints, `scripts/run_benchmarks.py`; generated ignored JSON/CSV under `benchmarks/results/`.
- Commands executed: `python scripts/run_benchmarks.py`.
- Tests executed: benchmark output and completion checks.
- Test results: all benchmark cases completed without OOM.
- Benchmark results (`n=1,793`, warmed medians): CPU dense lag 0.888 ms, GPU dense lag 0.474 ms (1.87× compute-only); CPU CSR lag 3.863 ms, GPU CSR lag 4.541 ms. Dense host-to-device transfer was 9.62 ms. 999-permutation Global Moran: CPU 267.7 ms, GPU 218.1 ms (1.23×). Full matrix density was 0.99944; dense storage 25.72 MB versus estimated CSR 38.56 MB. End-to-end dense construction totals: CPU 131.9 ms, GPU 97.7 ms. GPU CSR construction/normalization was a negative result and much slower for the nearly complete matrix.
- Problems encountered: first per-stage GPU timings allowed asynchronous spillover; synchronization was added around construction and normalization. End-to-end timings were already synchronized and remain valid.
- Unresolved questions: break-even varies by hardware/driver and workload; sizes above 1,793 were not run in this MVP pass.
- Next action: complete notebooks, benchmark report, README, and final acceptance verification.
