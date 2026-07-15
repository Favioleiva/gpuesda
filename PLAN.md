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

## Risks

- CuPy wheels and CUDA runtime must match the NVIDIA driver and Python version.
- PySAL conditional Local Moran permutations may not share exact random streams with GPU RNGs.
- Dense all-pairs weights are quadratic in memory; blockwise and sparse restrictions are mandatory at scale.
- GPU availability in CI is optional, so GPU tests must skip cleanly while explicit GPU requests fail clearly.
- Full 9,999-permutation local inference can be expensive; streaming summaries avoid simulation storage.

## Current phase

Phase 6 — Delivery complete.

## Next concrete action

MVP complete. Next optional action: run the extended 5k–25k and cross-hardware benchmark matrix.
