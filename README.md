# GPU Powered ESDA

`gpu_esda` 0.2.0 is an independent CPU/GPU package for exploratory spatial data analysis. It now contains two complementary engines:

- the validated vector engine for dense/CSR weights, Columbus, Global Moran, Local Moran, conditional permutations, FDR, panels, tests, benchmarks, and notebooks;
- an implicit raster-stencil engine for regular grids, designed for millions of cells without constructing an `n × n` weights matrix.

PySAL is the correctness reference and is never modified. Julia is not part of either ESDA engine.

## Install

```powershell
python -m pip install -e ".[dev,gpu,blackmarble]"
python scripts/diagnose_environment.py
```

The local GPU setup uses `cupy-cuda13x[ctk]`. For a CUDA 12 environment, install the matching `cupy-cuda12x` package. Explicit `backend="gpu"` raises if CUDA is unavailable; `backend="auto"` probes a real kernel and can fall back to NumPy.

## Vector quick start

```python
import numpy as np
from gpu_esda import inverse_distance_weights, Moran, MoranLocal

coordinates = np.random.default_rng(12345).random((500, 2))
y = np.random.default_rng(54321).normal(size=500)
weights = inverse_distance_weights(coordinates, power=2, row_standardize=True)
global_result = Moran(y, weights, permutations=999, seed=12345)
local_result = MoranLocal(y, weights, permutations=999, seed=12345, fdr=True)
```

Vector inputs are observation-first: `(n,)`, `(n, columns)`, or `(n, variables, periods)`. Existing `moran_panel` and `local_moran_panel` APIs remain available.

## Raster quick start

```python
from gpu_esda import BlackMarbleRaster, RasterWeights, moran_global, moran_local

raster = BlackMarbleRaster.from_parquet(
    "grid_peru_vnp46a2.parquet",
    "blackmarble_peru_2024-03-21.parquet",
    value_column="ntl",
)
weights = RasterWeights.queen(raster.mask, normalization="row", backend="gpu")

global_result = moran_global(raster.values, weights, permutations=99, seed=12345)
local_result = moran_local(
    raster.values,
    weights,
    permutations=99,
    seed=12345,
    fdr=True,
)
```

Raster weights are implicit Rook, Queen, or circular `d^-2` stencils. Row normalization excludes NoData, coast, and holes from both numerator and denominator. NTL equal to zero remains valid. Main GPU arrays default to float32 and global reductions use float64; float64 validation mode is supported.

Global permutations are unconditional shuffles. Local inference uses focal-fixed conditional randomization with neighbor samples drawn without replacement from all other valid cells. Both stream batches and summary accumulators instead of allocating `n_cells × permutations`; GPU OOM retries with a smaller batch. See the [methodology](docs/methodology.md) for the distinction from exact PySAL random streams.

## Black Marble Peru workflow

The reproducible grid audit is:

```powershell
python scripts/audit_blackmarble_grid.py
```

The audited 2024-03-21 files each have 6,111,958 unique one-to-one `cell_id` values. Their integer topology reconstructs directly into a 4,395 × 3,042 national rectangle with 6,111,958 occupied positions and 7,257,632 structural holes. No spatial join is required.

Run observed local analysis on CUDA with:

```powershell
python scripts/run_blackmarble_observed.py
python scripts/run_blackmarble_permutation_smoke.py
```

Generated JSON, Parquet, CSV, and PNG artifacts are written below `results/` and `benchmarks/results/`. Figures are always saved at 200 dpi before display. These host outputs are intentionally ignored by Git; `.gitkeep` files retain the directories.

The canonical Google Colab run is `notebooks/06_blackmarble_peru_a100.ipynb`. It installed the unchanged 0.2.0 wheel and completed successfully on an NVIDIA A100-SXM4-80GB. It verified hashes and row counts, validated a small CPU/GPU window, used `log(1000 × NTL + 1)`, ran 999 production permutations with seed 12345, applied BH-FDR at `q ≤ 0.05`, saved every figure, and exported cell-level results retaining `cell_id`.

## Reproduce and verify

```powershell
python -m pytest -q
python -m ruff check src tests scripts benchmarks
python -m compileall -q src tests scripts benchmarks
python scripts/run_validation.py
python scripts/diagnose_environment.py
python -m build --wheel
```

The raster suite covers 3×3, 5×5, and 10×10 grids; complete, hole, irregular coast, island, simulated tile boundary, valid zero, and NoData cases; CPU/GPU parity; and explicit PySAL equivalents on small grids. The full suite also retains all vector tests.

## Current evidence

The production A100 results use `log(1000 × NTL + 1)`. Global Moran I is 0.6121386781 for Rook, 0.5834607333 for Queen, and 0.5826043915 for radius-2 `d^-2`; every global `p_sim` is 0.001. After BH-FDR and exclusion of NoData and islands, significant local clusters total 403,205, 1,174,888, and 1,360,043 respectively. Rook has the largest global association but far fewer significant local clusters. Queen and `d^-2` have similar Global Moran I, while `d^-2` retains 185,155 more local clusters.

The package's raw `significant` boolean is not the final cluster mask. Reporting uses `valid & ~island & significant & quadrant_between_1_and_4`. This distinction removes Queen's 10 islands from its raw 1,174,898 boolean count. Times in the A100 files describe this concrete execution only and are not universal benchmarks.

Run `python scripts/blackmarble_postprocessing.py` to regenerate the corrected comparison, interpretation JSON, and SHA-256 manifest without CUDA. Original A100 CSV/JSON artifacts remain immutable. Only Queen has an archived cell-level Parquet; the two Queen paths are identical copies, and no Rook or `d^-2` local Parquet was retained.

Vector evidence remains unchanged: Columbus matches PySAL to floating-point tolerance with zero quadrant mismatches, and all original notebooks and benchmarks remain present.

See the [A100 results](docs/blackmarble_a100_results.md), [artifact audit](docs/blackmarble_a100_artifact_audit.md), [raster API](docs/raster_api.md), [grid audit](docs/blackmarble_grid_audit.md), [methodology](docs/methodology.md), [benchmark report](docs/benchmark_report.md), and [status](STATUS.md).

## Validation policy

NaN and infinity are never converted to zero. Constant variables and duplicate vector coordinates are rejected. Vector islands remain zero rows; raster islands are explicit. Float64 remains the vector default. Raster float32 has dedicated CPU/GPU tolerance tests, while full float64 mode is available for validation.

## License

Project code is BSD-3-Clause. Columbus is accessed from the user's PySAL installation and is not redistributed. Black Marble inputs remain in the private Hugging Face repository and are not committed.
