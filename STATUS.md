# GPU Powered ESDA Status

## Current state

- Phase: Raster extension — local delivery complete
- Overall: vector MVP protected; raster CPU/GPU 0.2.0 tested and packaged
- Branch: `feature/raster-stencil`
- Vector baseline tag: `v0.1.0-vector-mvp`
- Package version: `0.2.0`
- Latest implementation commit: `d7f15b1`
- Next action: execute notebook 06 on external A100 Colab hardware

## Updates

### 2026-07-16 01:12 JST — National observed Black Marble run completed

- Completed task: reconstructed the actual 6,111,958-cell raster and ran observed Global Moran, Local Moran, spatial lag, islands, and HH/LL/HL/LH classes for raw NTL and `log1p(NTL)` with Rook, Queen, and radius-2 `d^-2`.
- Files created or modified: national runner, raster Parquet/JSON I/O, reusable figure saver, six ignored PNG outputs, global JSON, and the 6,111,958-row Queen local Parquet.
- Commands executed: `python scripts/run_blackmarble_observed.py` against the authenticated private data already downloaded by the audit.
- Tests executed: output row/schema checks, retained-zero check, finite observed statistics, and synchronized CUDA timing/memory capture.
- Test results: all six configurations completed; 419,203 valid NTL zeros were preserved. The Queen observed local export retains `cell_id`, `local_i`, `spatial_lag`, `quadrant`, `p_sim`, `p_fdr`, and `significant`.
- Benchmark results: complete script wall time 17.854 s; read 0.207 s; reconstruction including indexed scatter 7.143 s; host-to-device 0.021 s; peak host RSS 1.585 GB; peak CuPy pool use 548.2 MB. `log1p(NTL)` Global Moran I: Rook 0.8838645194, Queen 0.8645893256, and `d^-2` radius 2 0.8618631192.
- Problems encountered: scalar standardization initially promoted raster arrays to float64. Means, standard deviations, and Local Moran scale are now cast back to the configured main dtype; a regression test enforces float32 residency.
- Decisions: national local permutation inference was not run on the RTX 3060 because it adds stability risk without validating new code beyond the small-grid and smoke tests.
- Next action: implement and validate streaming permutation inference.

### 2026-07-16 01:34 JST — Streaming raster inference validated

- Completed task: added global unconditional and focal-fixed conditional-local permutation inference with scalar/per-cell accumulators, seed control, automatic VRAM batching, cell chunking, and OOM batch reduction.
- Files created or modified: `src/gpu_esda/raster/permutations.py`, public raster statistics, permutation tests, smoke runner, and methodology documentation.
- Commands executed: focused CPU/GPU permutation tests, complete pytest/Ruff runs, and `python scripts/run_blackmarble_permutation_smoke.py`.
- Tests executed: CPU reproducibility, GPU smoke, accumulator outputs, FDR, and statistical comparison with ESDA on a small explicit graph.
- Test results: 52 passed; Ruff clean. No array shaped `n_cells × n_permutations` is allocated.
- Benchmark results: national `log1p(NTL)` Queen Global Moran with 99 permutations returned I=0.8645893254 and `p_sim=0.01`; inference took 4.394 s on the RTX 3060.
- Decisions: global and conditional-local randomization are documented as different methods, not equivalent procedures. RNG streams are reproducible within a backend but are not claimed bitwise identical to PySAL/Numba.
- Next action: package the tested implementation and generate the A100 notebook.

### 2026-07-16 01:52 JST — Package 0.2.0 and A100 notebook delivered

- Completed task: updated the package version, built and independently imported the wheel, and created a package-driven Colab notebook for the complete Peru workflow.
- Files created or modified: `pyproject.toml`, package version/export files, `environment.yml`, `dist/gpu_esda-0.2.0-py3-none-any.whl`, and `notebooks/06_blackmarble_peru_a100.ipynb`.
- Commands executed: wheel build, wheel metadata/hash inspection, isolated target installation/import, notebook schema and source compilation validation.
- Tests executed: wheel import plus raster Moran smoke outside the source tree; all notebook Python cells compile (the `%pip` cell is intentionally IPython syntax).
- Test results: wheel version 0.2.0 imports independently. The final documentation-bearing rebuild is 32,148 bytes with SHA-256 `40d3b7d46c5b0eeeae3d9f76302bec42ab620a74b8e0c68d313065cb5b068fd9`. Notebook has nine valid cells and installs the wheel instead of embedding algorithms.
- Notebook coverage: A100 assertion, hardware/software metadata, secure Hugging Face token, file hash/row verification, minimal-column PyArrow read, raster reconstruction, small CPU/GPU validation, raw/log analyses, 99-before-999 inference, FDR, all required exports, saved figures, and benchmark table.
- Limitation: the notebook was not executed on an A100 because this workstation has an RTX 3060. Its package path and algorithms were exercised locally on the complete national data; the external A100 run is the remaining operational action.
- Next action: final combined regression and documentation verification.

### 2026-07-16 02:06 JST — Final regression gate passed

- Completed task: reran the entire combined vector/raster quality gate after the notebook and documentation updates.
- Files created or modified: `README.md`, `PLAN.md`, `STATUS.md`, raster API and benchmark documentation, and notebook 06.
- Commands executed: full pytest; Ruff; compileall; pip dependency check; notebook nbformat/AST validation; Columbus validation; live environment diagnostics; final wheel rebuild without build isolation.
- Test results: 52 passed in 27.76 s with one upstream ESDA deprecation warning; Ruff clean; Python compilation successful; no broken requirements; notebook valid with nine cells. Columbus remains exact on CPU, within `7.11e-15` lag and `6.66e-16` Local Moran on GPU, with zero quadrant mismatches.
- Environment result: live RTX 3060 detected with 12 GiB, CuPy 14.1.1, CUDA runtime 13.2, and driver 591.86.
- Problems encountered: isolated wheel rebuild attempted to download Hatchling and was blocked by managed network access; the already-installed build backend was used safely with `--no-build-isolation`, producing the final wheel.
- Next action: execute notebook 06 on external A100 Colab hardware and retain its exported benchmark metadata; no local implementation task remains.

### 2026-07-16 00:18 JST — Vector MVP protected and baseline revalidated

- Completed task: verified clean Git state, tagged the validated vector MVP, created the raster feature branch, and reran every existing quality gate before code changes.
- Files created or modified: `STATUS.md` only; Git tag `v0.1.0-vector-mvp`; branch `feature/raster-stencil`.
- Commands executed: Git status/tag/branch operations; `python -m pytest -q`; Ruff; `compileall`; environment diagnostics; `pip check`.
- Tests executed: complete existing suite, live CPU/GPU tests included.
- Test results: 28 passed in 3.33 s; Ruff clean; compilation successful; no broken requirements.
- Benchmark results: none in this stage.
- Environment: RTX 3060 12 GiB, compute capability 8.6, CuPy 14.1.1/CUDA runtime 13.2, driver 591.86; 64 GiB host RAM. Numba CUDA is unavailable, so the raster GPU path will use CuPy.
- Problems encountered: none.
- Unresolved questions: private Hugging Face data/token availability and whether `grid_row/grid_col` form a continuous national raster.
- Next action: create the reproducible grid-audit script and inspect the actual Parquet files without exposing credentials.

### 2026-07-16 00:31 JST — Black Marble grid audited from private Parquet

- Completed task: implemented and ran authenticated, reproducible Parquet audit using minimum required columns.
- Files created or modified: `scripts/audit_blackmarble_grid.py`, generated `results/blackmarble_grid_audit.json`, `docs/blackmarble_grid_audit.md`, result directories.
- Commands executed: authenticated Hugging Face repository metadata query; `python scripts/audit_blackmarble_grid.py`; schema/hash inspection.
- Tests executed: row/schema/uniqueness/topology checks embedded in the audit.
- Test results: both files contain 6,111,958 unique one-to-one `cell_id` values in identical order; `(grid_row, grid_col)` is unique; tile row/column offsets are constant.
- Benchmark results: national rectangle is 4,395 × 3,042 (13,369,590 positions), with 6,111,958 occupied positions and 7,257,632 holes (45.7154% coverage). One float32 rectangle is 51.0 MiB; three are 153.0 MiB.
- Problems encountered: none; the national integer topology is continuous and required no correction.
- Unresolved questions: missing daily-value count and full observed GPU timings are pending reconstruction.
- Next action: implement and validate implicit spatial operators.

### 2026-07-16 00:42 JST — Implicit raster CPU/GPU operators validated

- Completed task: added operator abstraction, matrix adapter, Black Marble raster reconstruction, Rook/Queen/circular d^-2 stencils, NumPy/CuPy lag, observed Global/Local Moran, quadrant and island handling.
- Files created or modified: `src/gpu_esda/operators/*`, `src/gpu_esda/raster/*`, public API, three raster test modules, raster API/methodology docs.
- Commands executed: Ruff format/check; CPU/GPU smoke checks; complete pytest suite.
- Tests executed: 3×3, 5×5, 10×10, full grid, interior hole, irregular coast, island, simulated tile border, valid zero, NoData, batch shape, CPU/GPU equivalence, explicit PySAL equivalence.
- Test results: 47 passed in 4.11 s; Ruff clean. No vector regressions.
- Benchmark results: none at national scale yet.
- Problems encountered: CuPy's `where=` ufunc form is not portable; row normalization uses a safe denominator followed by masking.
- Unresolved questions: production permutation cost at 6.1 million valid cells remains to be profiled.
- Next action: run observed NTL/log1p workflows on the local RTX 3060 with staged timing and memory tracking.

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

### 2026-07-16 00:06 JST — Documented and verified MVP completed

- Completed task: created all five package-driven notebooks, benchmark report, complete README, real-kernel GPU fallback probe, and final acceptance verification.
- Files created or modified: `README.md`, `docs/benchmark_report.md`, five notebooks, `src/gpu_esda/backend.py`, `src/gpu_esda/weights.py`, `PLAN.md`, `STATUS.md`.
- Commands executed: notebook validation with `nbformat`; `python -m compileall`; `python -m pip check`; environment diagnostic; final Columbus validation; Git status/log review.
- Tests executed: `python -m pytest -q`; `python -m ruff check src tests scripts benchmarks`; validation script on CPU and live RTX 3060 GPU.
- Test results: 28 passed, Ruff clean, five notebooks valid, no broken Python requirements, compilation successful. Final Columbus CPU differences are zero; GPU lag max difference `7.11e-15`, Local Moran max difference `6.66e-16`, zero quadrant mismatches.
- Benchmark results: retained from the synchronized representative run above; no claims were changed by final verification.
- Problems encountered: notebook validator warns that compact hand-authored cells omit optional cell IDs; notebooks are valid under nbformat 4.5 and the warning does not affect execution.
- Unresolved questions: performance on T4/L4/A100 and 5k–25k workloads is not measured and is explicitly excluded from current claims.
- Next action: none required for the MVP; optional extended benchmark campaign.

## MVP acceptance checklist

- [x] Installable `src` package; CPU-only mode and automatic fallback.
- [x] Complete GPU detection and explicit failure behavior.
- [x] Reproducible official Columbus dataset, Queen/KNN CPU baselines.
- [x] Independent dense/CSR inverse-distance-squared weights and normalization.
- [x] CPU/GPU spatial lag, Global Moran, Local Moran, permutations, and FDR.
- [x] 999 and 9,999 permutations executed on the reference-size dataset.
- [x] Deterministic equivalence tests and documented RNG differences.
- [x] Synchronized warmed benchmarks with transfer time and dense/CSR comparison.
- [x] Multi-variable/panel APIs and result tables.
- [x] Machine-readable benchmark/validation exports.
- [x] Five runnable package-driven notebooks and complete documentation.
- [x] `STATUS.md` reflects the tested repository state.
