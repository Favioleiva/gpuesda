# GPU Powered ESDA Plan

## Phases and acceptance criteria

| Phase | Tasks | Depends on | Acceptance criteria | Expected outputs |
|---|---|---|---|---|
| 1. Foundation | Initialize Git, scaffold, diagnostics, packaging | None | Installable src-layout package; CPU-only diagnostics succeed | Project tree, `pyproject.toml`, diagnostic JSON |
| 2. Reference study | Audit PySAL/ESDA, map operations, choose open data | Dependencies | Algorithms and compatibility targets documented from installed source | Audit, mapping, dataset selection |
| 3. Numerical core | CPU baseline, backend, weights, lag, Moran, Local Moran, permutations, FDR | Phase 2 | APIs work on CPU; GPU or explicit fallback behavior is tested | Package modules and scripts |
| 4. Validation | PySAL equivalence and edge-case tests | Phase 3 | Required tests pass at documented float tolerances | Test suite and validation output |
| 5. Performance | Synthetic data and representative benchmarks | Phase 4 | Transfer and compute timing exported; infeasible cases recorded | JSON/CSV benchmark results |
| 6. Delivery | Notebooks, report, README, final status | Phase 5 | Reproducible documented MVP satisfying acceptance checklist | Five notebooks, reports, final commit |
| 7. Raster topology | Audit Black Marble cell/grid/tile indices and memory | Vector MVP | One-to-one verified national integer topology or documented correction | JSON audit and readable report |
| 8. Raster observed core | Spatial operators, CPU stencils, CuPy stencils, observed Moran/LISA | Phase 7 | Small-grid PySAL equivalence and CPU/GPU equivalence | Raster package modules and tests |
| 9. Raster inference | Streaming global and conditional-local permutations | Phase 8 | Reproducible 99/999 modes without `n × permutations` storage | Batched inference and OOM fallback |
| 10. Black Marble/A100 delivery | Local observed run, wheel, Colab notebook, figures, docs | Phase 9 | Reproducible outputs and complete regression suite | v0.2 wheel and notebook 06 |

## Risks

- CuPy wheels and CUDA runtime must match the NVIDIA driver and Python version.
- PySAL conditional Local Moran permutations may not share exact random streams with GPU RNGs.
- Dense all-pairs weights are quadratic in memory; blockwise and sparse restrictions are mandatory at scale.
- GPU availability in CI is optional, so GPU tests must skip cleanly while explicit GPU requests fail clearly.
- Full 9,999-permutation local inference can be expensive; streaming summaries avoid simulation storage.

## Current phase

Phase 10 — Black Marble/A100 delivery completed locally. The A100 execution artifact is ready but has not been run on A100 hardware from this workstation.

## Completed raster acceptance criteria

- The private Black Marble grid and daily Parquet were audited at 6,111,958 rows with a verified one-to-one integer topology.
- Rook, Queen, and circular `d^-2` run as implicit NumPy/CuPy stencils without an `n × n` matrix.
- Observed Global/Local Moran and quadrants were validated against explicit PySAL weights on small grids and run nationally on an RTX 3060.
- Global and focal-fixed conditional-local inference stream permutation batches and recover from GPU OOM by reducing batch size.
- The vector regression suite remains intact; the combined suite passes on CPU and live CUDA.
- Package version 0.2.0 builds as an independently importable wheel.
- Notebook 06 installs that wheel and contains the reproducible A100 workflow rather than a duplicate implementation.

## Next concrete action

Upload the tested 0.2.0 wheel to an A100 Colab session and execute notebook 06 end to end. This is an external-hardware run, not an outstanding local implementation task.
