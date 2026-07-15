# GPU Powered ESDA — Project Specification

We are starting the **GPU Powered ESDA** project from scratch. The current project directory is empty.

Build a complete Python project whose purpose is to develop, validate, and benchmark a GPU-oriented backend for Exploratory Spatial Data Analysis, while remaining statistically compatible with the main behavior of `libpysal` and `esda`.

The project should focus on accelerating:

* inverse-distance spatial weights, especially (d^{-2});
* spatial-weight normalization;
* spatial lag calculations (Wy);
* Global Moran’s I;
* Local Moran’s I / LISA;
* permutation-based inference;
* multiple variables and temporal panels;
* repeated ESDA workflows over many spatial units, dates, variables, and weight specifications.

The main goal is not merely to use a GPU. The goal is to reduce the total time required to move from spatial data to validated ESDA results.

---

# 1. Core Principles

Follow these principles throughout the project:

1. Use PySAL on CPU as the reference baseline.

2. Do not modify the installed source code of `libpysal` or `esda`.

3. Build an independent package that reproduces the relevant mathematical and statistical behavior.

4. Prioritize correctness, reproducibility, and statistical equivalence before speed.

5. Do not claim performance improvements without reproducible benchmarks.

6. Use GPU acceleration only where it is technically justified.

7. Keep geometry processing and plotting on CPU unless a measurable benefit from GPU processing is demonstrated.

8. Use CuPy and `cupyx.scipy.sparse` as the primary GPU tools.

9. Use Numba CUDA only when CuPy cannot efficiently implement the required operation.

10. Provide automatic CPU fallback whenever CUDA or CuPy is unavailable.

11. Do not assume that sparse matrices are always faster than dense matrices.

12. Test dense and sparse implementations according to matrix size, density, and memory requirements.

13. Separate data-transfer time from computation time in every benchmark.

14. Use fixed random seeds and document all random-number behavior.

15. Maintain clear project documentation and update progress continuously.

16. Work autonomously. Do not request confirmation unless there is a genuine technical blocker or a decision that cannot be resolved from this specification.

---

# 2. Initial Project Structure

Create the following project structure:

```text
GPU Powered ESDA/
├── README.md
├── PROJECT_SPEC.md
├── PLAN.md
├── STATUS.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── src/
│   └── gpu_esda/
│       ├── __init__.py
│       ├── backend.py
│       ├── diagnostics.py
│       ├── weights.py
│       ├── lag.py
│       ├── moran.py
│       ├── local_moran.py
│       ├── permutations.py
│       ├── multiple_testing.py
│       ├── validation.py
│       └── benchmarking.py
├── tests/
│   ├── test_backend.py
│   ├── test_weights.py
│   ├── test_lag.py
│   ├── test_moran.py
│   ├── test_local_moran.py
│   ├── test_permutations.py
│   ├── test_multiple_testing.py
│   └── test_cpu_gpu_equivalence.py
├── benchmarks/
│   ├── benchmark_open_dataset.py
│   ├── benchmark_inverse_distance.py
│   ├── benchmark_spatial_lag.py
│   ├── benchmark_moran.py
│   ├── benchmark_local_moran.py
│   ├── benchmark_permutations.py
│   ├── benchmark_panel.py
│   └── results/
├── notebooks/
│   ├── 01_open_dataset_validation.ipynb
│   ├── 02_inverse_distance_benchmark.ipynb
│   ├── 03_moran_cpu_gpu_equivalence.ipynb
│   ├── 04_local_moran_permutations.ipynb
│   └── 05_panel_scaling_benchmark.ipynb
├── docs/
│   ├── pysal_esda_audit.md
│   ├── cpu_gpu_mapping.md
│   ├── methodology.md
│   ├── dataset_selection.md
│   ├── benchmark_methodology.md
│   └── benchmark_report.md
├── scripts/
│   ├── diagnose_environment.py
│   ├── run_validation.py
│   ├── run_cpu_baseline.py
│   └── run_benchmarks.py
└── results/
```

Initialize Git if the folder is not already a repository.

Create small, descriptive commits after meaningful stages.

Do not commit:

* downloaded datasets that can be reproduced automatically;
* virtual environments;
* temporary benchmark files;
* cache directories;
* GPU compilation artifacts;
* credentials;
* large generated outputs that are not necessary.

---

# 3. Project Management Files

Create `PLAN.md` with:

* project phases;
* tasks;
* dependencies;
* acceptance criteria;
* risks;
* expected outputs;
* current phase;
* next concrete action.

Create `STATUS.md` and update it after every stage.

Each status update should include:

* date and time;
* completed task;
* files created or modified;
* commands executed;
* tests executed;
* test results;
* benchmark results, when applicable;
* problems encountered;
* unresolved questions;
* next action.

Do not use `STATUS.md` as a narrative diary. Keep it structured and technically useful.

---

# 4. Python Environment and Dependencies

Configure the package using `pyproject.toml` and a `src` layout.

Required CPU dependencies should include:

* `numpy`;
* `scipy`;
* `pandas`;
* `geopandas`;
* `shapely`;
* `pyproj`;
* `libpysal`;
* `esda`;
* `scikit-learn`;
* `statsmodels`;
* `psutil`;
* `pytest`.

Optional GPU dependencies should include:

* `cupy`;
* `numba`.

Do not install a CuPy build blindly.

First detect:

* operating system;
* Python version;
* NVIDIA driver;
* available GPU;
* CUDA driver compatibility;
* installed CUDA toolkit, if present;
* existing CuPy installation;
* GPU compute capability.

Then choose a compatible CuPy package.

Keep GPU dependencies optional so that the package can still install and run on CPU-only systems.

Use optional dependency groups where appropriate, for example:

```toml
[project.optional-dependencies]
gpu = [
    "cupy-cuda12x"
]

dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy"
]
```

The precise CuPy package must be selected according to the detected CUDA environment.

---

# 5. Environment Diagnostics

Create:

```text
scripts/diagnose_environment.py
```

It should report:

* operating system;
* platform architecture;
* Python version;
* CPU model;
* logical and physical CPU cores;
* total RAM;
* available RAM;
* NumPy version;
* SciPy version;
* pandas version;
* GeoPandas version;
* `libpysal` version;
* `esda` version;
* Numba version;
* whether CUDA is available through Numba;
* whether CuPy imports successfully;
* CuPy version;
* detected NVIDIA GPU;
* GPU compute capability;
* total GPU memory;
* available GPU memory;
* CUDA runtime version;
* NVIDIA driver version.

The script must fail gracefully on CPU-only systems.

Do not treat the absence of a GPU as an error. Report it clearly and continue with CPU mode.

---

# 6. Technical Audit of PySAL and ESDA

Before implementing the GPU backend, inspect the installed source code of:

* `libpysal.weights.Queen`;
* `libpysal.weights.Rook`, if relevant;
* `libpysal.weights.KNN`;
* `libpysal.weights.DistanceBand`;
* `libpysal.weights.W`;
* `libpysal.weights.WSP`;
* `libpysal.weights.lag_spatial`;
* `esda.Moran`;
* `esda.Moran_Local`.

Document the audit in:

```text
docs/pysal_esda_audit.md
docs/cpu_gpu_mapping.md
```

For each component, identify:

1. Source file and relevant class or function.

2. Public API.

3. Input and output formats.

4. Internal data structures.

5. Use of Python dictionaries, lists, NumPy arrays, SciPy sparse matrices, or Numba.

6. Formula implemented.

7. Weight transformations and normalization behavior.

8. Treatment of islands and disconnected observations.

9. Treatment of self-neighbors and diagonals.

10. Random seed behavior.

11. Permutation algorithm.

12. Pseudo-p-value definition.

13. Quadrant classification.

14. Analytical moments, expected values, and variances.

15. Computational complexity.

16. Memory behavior.

17. Python-level loops.

18. Operations suitable for CuPy.

19. Operations suitable for `cupyx.scipy.sparse`.

20. Operations that should remain on CPU.

21. Areas where exact replication may be difficult.

22. Potential numerical differences between float32 and float64.

23. Potential differences between NumPy and CuPy random-number generators.

The audit must distinguish between:

* geometry construction;
* graph or weight construction;
* matrix representation;
* numerical statistics;
* permutation inference;
* visualization.

Do not start by trying to port the entire PySAL library.

Identify the smallest numerical core that can provide the greatest performance gain.

---

# 7. CPU–GPU Functional Mapping

In `docs/cpu_gpu_mapping.md`, create a table mapping CPU operations to possible GPU implementations.

At minimum, evaluate:

| CPU operation           | Current tool                | Candidate GPU tool                            |
| ----------------------- | --------------------------- | --------------------------------------------- |
| Array operations        | NumPy                       | CuPy                                          |
| Sparse CSR operations   | SciPy sparse                | `cupyx.scipy.sparse`                          |
| Spatial lag             | SciPy matrix multiplication | CuPy or CuPy sparse                           |
| Pairwise distances      | NumPy/SciPy                 | CuPy broadcasting or kernels                  |
| KNN search              | SciPy / PySAL               | GPU distance blocks or external GPU tools     |
| Permutation generation  | NumPy / Numba               | CuPy random or Numba CUDA                     |
| Moran reductions        | NumPy                       | CuPy reductions                               |
| Local Moran computation | NumPy / Numba               | CuPy elementwise operations                   |
| FDR                     | statsmodels                 | CPU baseline or GPU vectorized implementation |
| Geometry topology       | GeoPandas/Shapely           | Remain on CPU initially                       |
| Mapping                 | GeoPandas/Matplotlib        | Remain on CPU                                 |

For every proposed GPU operation, state:

* expected benefit;
* expected implementation difficulty;
* numerical risks;
* memory risks;
* fallback strategy.

---

# 8. Open Reference Dataset

Use a fully open dataset officially distributed through `libpysal.examples` or another official PySAL example source.

The reference dataset must include:

* geometries;
* at least one meaningful numerical variable;
* a clear source;
* a documented license;
* enough observations to test spatial-weight construction;
* stable and reproducible access.

Inspect available datasets programmatically.

Select one dataset for correctness validation.

Document the selection in:

```text
docs/dataset_selection.md
```

Include:

* dataset name;
* original source;
* license;
* number of observations;
* geometry type;
* coordinate reference system;
* variables used;
* reason for selection;
* limitations;
* code required to load it.

The official open dataset must be used for equivalence testing.

If it is too small to measure GPU performance, also create synthetic benchmark datasets.

Do not modify or artificially enlarge the official dataset for correctness testing.

---

# 9. Synthetic Benchmark Data

Create reproducible synthetic datasets for performance testing.

Test sizes should include, when feasible:

```text
n = 500
n = 1,000
n = 1,793
n = 5,000
n = 10,000
n = 25,000
```

The value `n = 1,793` is important because it approximates the number of Peruvian districts used in the target research workflow.

Synthetic datasets should support:

* random coordinates;
* clustered coordinates;
* grid coordinates;
* multiple variables;
* multiple periods;
* spatially autocorrelated variables;
* independent random variables;
* missing-value scenarios;
* islands;
* duplicated coordinates as an explicit error case.

Use fixed seeds.

Document how each synthetic dataset is produced.

---

# 10. CPU Baseline

Implement a clean CPU baseline using `libpysal` and `esda`.

The baseline must calculate:

1. Queen weights.

2. KNN weights.

3. Distance-band weights.

4. Inverse-distance weights.

5. Inverse-distance-squared weights.

6. Row-standardized weights.

7. Binary weights.

8. Spatial lag (Wy).

9. Global Moran’s I.

10. Local Moran’s I.

11. LISA quadrants.

12. 99 permutations.

13. 999 permutations.

14. 9,999 permutations.

15. Pseudo-p-values.

16. Significance at 0.05.

17. Benjamini–Hochberg FDR correction.

18. Multiple variables.

19. Multiple time periods.

The CPU baseline should not be embedded only in notebooks.

Create reusable scripts and package functions.

Save benchmark and validation outputs in machine-readable formats such as:

* JSON;
* CSV;
* Parquet.

Record:

* random seed;
* package versions;
* number of observations;
* matrix density;
* number of nonzero values;
* data type;
* number of permutations;
* elapsed time;
* peak process memory.

Use `time.perf_counter()` for CPU timing.

---

# 11. Backend Abstraction

Create a backend abstraction that supports:

```python
backend="cpu"
backend="gpu"
backend="auto"
```

The public API should not force users to work directly with CuPy.

Suggested interface:

```python
from gpu_esda import (
    inverse_distance_weights,
    spatial_lag,
    Moran,
    MoranLocal,
)
```

Examples:

```python
W = inverse_distance_weights(
    coordinates,
    power=2,
    row_standardize=True,
    backend="auto",
)
```

```python
lag = spatial_lag(
    W,
    y,
    backend="auto",
)
```

```python
result = Moran(
    y,
    W,
    permutations=999,
    seed=12345,
    backend="auto",
)
```

```python
result = MoranLocal(
    y,
    W,
    permutations=999,
    seed=12345,
    backend="auto",
)
```

The backend module should:

* detect CuPy;
* detect CUDA;
* validate GPU availability;
* select CPU or GPU;
* expose the active backend;
* provide safe array conversion;
* provide safe synchronization;
* provide memory diagnostics;
* avoid unnecessary host-device transfers.

When `backend="gpu"` is explicitly requested and no GPU is available, raise a clear exception.

When `backend="auto"` is requested and no GPU is available, use CPU without failing.

---

# 12. Inverse-Distance Weights

Implement inverse-distance weights:

[
w_{ij} = \frac{1}{d_{ij}^{p}}
]

Use `power=2` by default.

The implementation must support:

* full distance matrices;
* thresholded distance matrices;
* K-nearest-neighbor restrictions;
* diagonal exclusion;
* zero-distance checks;
* duplicate-coordinate detection;
* row standardization;
* binary output;
* dense output;
* CSR output;
* automatic dense-versus-sparse selection;
* float32;
* float64;
* block processing;
* CPU backend;
* GPU backend.

Suggested API:

```python
W = inverse_distance_weights(
    coordinates,
    power=2,
    threshold=None,
    k=None,
    row_standardize=True,
    binary=False,
    output_format="auto",
    dtype="float64",
    block_size=None,
    backend="auto",
)
```

For every constructed matrix, collect diagnostics:

* number of observations;
* matrix shape;
* number of nonzero entries;
* matrix density;
* minimum nonzero weight;
* maximum weight;
* row-sum statistics;
* number of islands;
* estimated dense memory;
* estimated CSR memory;
* actual construction time;
* normalization time.

Do not automatically choose CSR for nearly complete matrices.

For a (1,793 \times 1,793) matrix, explicitly compare dense GPU and CSR GPU performance.

For larger matrices, avoid creating full dense pairwise arrays when they would exceed memory.

Implement blockwise distance calculation when necessary.

---

# 13. Spatial Lag

Implement:

[
Wy
]

for:

* one vector;
* multiple variables;
* multiple periods;
* dense matrices;
* CSR matrices;
* CPU;
* GPU;
* float32;
* float64.

Support inputs shaped as:

```text
n
n × variables
n × periods
n × variables × periods
```

Define and document the internal shape conventions.

Avoid repeated transfers when many lags are calculated using the same weights.

Validate against:

```python
libpysal.weights.lag_spatial
```

The validation should compare:

* elementwise results;
* maximum absolute difference;
* relative error;
* row-standardized and binary weights;
* dense and sparse formats.

---

# 14. Global Moran’s I

Implement Global Moran’s I consistently with `esda.Moran`.

The observed statistic is based on:

[
I =
\frac{n}{S_0}
\frac{z^\top Wz}{z^\top z}
]

where (z) is the centered or standardized variable according to the PySAL convention.

Return, when applicable:

* observed Moran’s I;
* expected value;
* analytical variance;
* analytical z-score;
* analytical p-value;
* permutation values;
* permutation mean;
* permutation standard deviation;
* permutation z-score;
* pseudo-p-value;
* number of permutations;
* random seed;
* backend;
* data type;
* timing diagnostics.

Support:

* no permutations;
* 99 permutations;
* 999 permutations;
* 9,999 permutations;
* one variable;
* multiple variables;
* multiple periods.

Validate numerical equivalence with `esda.Moran`.

Define explicit tolerances for float64 and float32.

---

# 15. Local Moran’s I / LISA

Implement Local Moran’s I consistently with `esda.Moran_Local`.

The local statistic is based on:

[
I_i = z_i \sum_j w_{ij}z_j
]

Return:

* local statistic `Is`;
* standardized variable;
* spatial lag;
* quadrant code;
* readable quadrant label;
* pseudo-p-value;
* permutation mean;
* permutation variance or standard deviation;
* permutation z-score, when available;
* uncorrected significance;
* FDR-adjusted p-value;
* FDR significance;
* backend;
* data type;
* timing diagnostics.

Quadrant labels must include:

* HH;
* LH;
* LL;
* HL;
* not significant.

Document the exact mapping between PySAL quadrant codes and readable labels.

Validate:

* local statistics;
* spatial lag;
* quadrants;
* p-values;
* significance;
* FDR classification.

Avoid Python loops over spatial observations in the GPU implementation.

---

# 16. Permutation Engine

The permutation engine is the highest-priority optimization target.

Implement permutation inference using batches.

Test batch sizes:

```text
64
128
256
512
1024
```

Select batch size automatically based on:

* number of observations;
* matrix density;
* matrix format;
* data type;
* free GPU memory;
* number of permutations;
* number of variables or periods.

The engine should support:

* Global Moran permutations;
* Local Moran conditional permutations;
* one variable;
* multiple variables;
* panel data;
* reproducible seeds;
* CPU fallback.

Measure separately:

* random-index generation;
* data rearrangement;
* spatial-lag computation;
* statistic computation;
* reductions;
* pseudo-p-value calculation;
* host-device transfers.

Avoid storing every simulated statistic when only summary information is needed.

Use streaming accumulators when possible for:

* number of simulated values as or more extreme than observed;
* mean;
* variance;
* minimum;
* maximum.

If complete simulation output is requested, make it optional because it may require substantial memory.

Investigate whether PySAL’s Local Moran permutations are conditional or unconditional and reproduce the same procedure.

Do not assume that globally permuting the full variable vector reproduces PySAL’s Local Moran inference.

Document the exact algorithm.

---

# 17. Random-Number Reproducibility

Investigate how PySAL and ESDA generate random permutations.

Determine:

* whether NumPy or Numba random generation is used;
* how seeds are passed;
* whether results are reproducible across threads;
* whether permutations are generated observation by observation;
* whether GPU output can reproduce the identical sequence.

If identical permutation sequences can be reproduced, test exact equality.

If identical sequences cannot be reproduced because of different random-number generators:

1. reproduce the same statistical procedure;

2. use fixed GPU seeds;

3. compare results over multiple seeds;

4. evaluate Monte Carlo stability;

5. document the expected simulation uncertainty;

6. do not claim exact p-value equality when only statistical equivalence is possible.

---

# 18. Multiple Testing

Implement Benjamini–Hochberg FDR correction.

The CPU reference may use:

```python
statsmodels.stats.multitest.multipletests
```

The package should expose a reusable function such as:

```python
adjust_pvalues(
    pvalues,
    method="fdr_bh",
    alpha=0.05,
    backend="auto",
)
```

Validate adjusted p-values and rejection decisions against `statsmodels`.

GPU acceleration is optional if FDR is not a meaningful runtime bottleneck.

Correctness has priority.

---

# 19. Missing Values and Data Validation

Do not silently convert missing values to zero.

Define an explicit missing-value policy.

Possible supported policies:

```text
raise
drop
mask
```

Document which policies are compatible with each statistic.

Validate input data for:

* NaN;
* infinity;
* constant variables;
* zero variance;
* duplicated coordinates;
* negative distances;
* invalid matrix dimensions;
* zero row sums;
* islands;
* mismatched IDs;
* mismatched observation ordering.

All errors should produce clear messages.

---

# 20. Dense Versus Sparse Strategy

The package must include logic to compare dense and sparse representations.

For each weight matrix, estimate:

* dense memory;
* CSR memory;
* density;
* expected multiplication cost;
* GPU memory availability.

Create an `output_format="auto"` strategy.

Do not base the decision only on matrix dimensions.

For nearly complete (d^{-2}) matrices, dense matrix multiplication may be faster.

For Queen, Rook, and KNN matrices, CSR will usually be more appropriate.

Benchmark both approaches rather than relying on assumptions.

---

# 21. Precision

Support:

```text
float32
float64
```

Use float64 as the main validation precision.

Measure float32 separately.

Compare:

* spatial weights;
* row sums;
* spatial lag;
* Global Moran’s I;
* Local Moran’s I;
* permutation p-values;
* classification differences.

Do not recommend float32 by default unless the benchmark demonstrates acceptable numerical behavior.

---

# 22. Automated Tests

Use `pytest`.

Create tests for:

## Backend

* CPU detection;
* GPU detection;
* `backend="auto"`;
* explicit GPU failure without CUDA;
* array conversion;
* synchronization.

## Weights

* inverse-distance weights;
* (d^{-2});
* diagonal zero;
* threshold;
* KNN restriction;
* row standardization;
* binary transformation;
* dense output;
* CSR output;
* duplicate coordinates;
* islands.

## Spatial lag

* vector input;
* matrix input;
* dense weights;
* CSR weights;
* CPU equivalence;
* GPU equivalence;
* float32;
* float64.

## Global Moran

* observed statistic;
* expected value;
* analytical inference;
* permutations;
* fixed seeds;
* multiple variables.

## Local Moran

* local statistics;
* spatial lag;
* quadrants;
* conditional permutations;
* pseudo-p-values;
* FDR;
* multiple variables.

## Error handling

* invalid shapes;
* NaN;
* infinity;
* constant vectors;
* incompatible IDs;
* insufficient memory where detectable.

Initial numerical tolerances:

```text
float64:
atol = 1e-10
rtol = 1e-8

float32:
atol = 1e-5
rtol = 1e-4
```

Adjust tolerances only with a documented justification.

---

# 23. Benchmark Methodology

Create:

```text
docs/benchmark_methodology.md
```

Benchmarks must compare:

* PySAL CPU baseline;
* project CPU backend;
* GPU backend including transfer time;
* GPU backend with data already resident on GPU.

Test weight types:

* Queen;
* KNN;
* full (d^{-2});
* thresholded (d^{-2});
* KNN-limited (d^{-2}).

Test numbers of permutations:

```text
0
99
999
9,999
```

Test workloads:

* one variable;
* ten variables;
* one hundred variables;
* temporal panel.

Test observation counts:

```text
500
1,000
1,793
5,000
10,000
25,000
```

Skip combinations that are not feasible in available memory, but record them as skipped and explain why.

Measure:

* data loading;
* weight construction;
* normalization;
* CPU-to-GPU transfer;
* spatial lag;
* Moran observed statistic;
* permutation generation;
* permutation evaluation;
* p-value reduction;
* GPU-to-CPU transfer;
* total runtime;
* CPU memory;
* GPU memory;
* matrix density;
* number of nonzero values;
* speedup.

Calculate:

[
\text{speedup}
==============

\frac{\text{CPU runtime}}
{\text{GPU runtime}}
]

Report speedup only for equivalent operations and outputs.

---

# 24. Correct GPU Timing

GPU operations are asynchronous.

Before timing:

1. run a warm-up operation;

2. synchronize the device;

3. start the timer;

4. run the operation;

5. synchronize again;

6. stop the timer.

Example:

```python
cp.cuda.Stream.null.synchronize()
start = time.perf_counter()

result = gpu_function(...)

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - start
```

Run multiple repetitions.

Report:

* number of repetitions;
* median;
* minimum;
* maximum;
* mean;
* standard deviation.

Do not report only the fastest execution.

Report both cold-start and warmed execution where useful.

---

# 25. GPU Memory Management

The implementation must avoid unnecessary GPU-memory copies.

Use:

* reusable buffers;
* block processing;
* explicit deletion where necessary;
* CuPy memory-pool diagnostics;
* batch-size adaptation.

Before large operations, estimate required memory.

When an operation cannot fit:

* reduce batch size;
* use blockwise processing;
* switch from dense to sparse if appropriate;
* raise a clear memory error if no safe strategy exists.

Do not allow uncontrolled out-of-memory failures when memory requirements can be estimated.

---

# 26. Panel and Multi-Variable Processing

The main productivity goal is to process many variables or periods without repeated Python calls.

Support matrices such as:

```text
n observations × m variables
```

and:

```text
n observations × t periods
```

Potential panel API:

```python
results = moran_panel(
    Y,
    weights=W,
    permutations=999,
    seed=12345,
    backend="gpu",
)
```

The result should identify:

* variable;
* date or period;
* Global Moran’s I;
* expected value;
* pseudo-p-value;
* runtime.

For Local Moran panels, return a tidy result containing:

* observation ID;
* variable;
* period;
* local statistic;
* quadrant;
* pseudo-p-value;
* adjusted p-value;
* significance.

Avoid recalculating or transferring the same spatial-weight matrix for every period.

---

# 27. Public API Design

Keep the user-facing API simple.

Possible usage:

```python
from gpu_esda import inverse_distance_weights
from gpu_esda import spatial_lag
from gpu_esda import Moran
from gpu_esda import MoranLocal
```

Example:

```python
W = inverse_distance_weights(
    coordinates=coordinates,
    power=2,
    row_standardize=True,
    output_format="auto",
    backend="auto",
)
```

```python
global_result = Moran(
    y=values,
    weights=W,
    permutations=9999,
    seed=12345,
    backend="auto",
)
```

```python
local_result = MoranLocal(
    y=values,
    weights=W,
    permutations=9999,
    seed=12345,
    fdr=True,
    backend="auto",
)
```

The package should expose useful result objects rather than loose tuples.

Result objects should support:

* readable summaries;
* conversion to dictionaries;
* conversion to pandas DataFrames;
* metadata;
* timing information;
* backend information;
* data type information.

---

# 28. Notebooks

Create the following notebooks:

```text
notebooks/
├── 01_open_dataset_validation.ipynb
├── 02_inverse_distance_benchmark.ipynb
├── 03_moran_cpu_gpu_equivalence.ipynb
├── 04_local_moran_permutations.ipynb
└── 05_panel_scaling_benchmark.ipynb
```

The notebooks must use functions from the package.

Do not duplicate the implementation inside notebook cells.

Notebook 1 should:

* load the official open dataset;
* create the CPU baseline;
* run GPU results when available;
* compare numerical results.

Notebook 2 should:

* compare CPU and GPU (d^{-2});
* compare dense and CSR;
* test normalization;
* report memory and timing.

Notebook 3 should:

* compare spatial lag;
* compare Global Moran;
* compare analytical and permutation results.

Notebook 4 should:

* compare Local Moran;
* compare quadrants;
* compare permutations;
* compare FDR;
* analyze random-seed differences.

Notebook 5 should:

* test multiple variables;
* test multiple periods;
* identify the GPU break-even point.

---

# 29. Benchmark Report

Create:

```text
docs/benchmark_report.md
```

The report should answer:

1. Which PySAL operations can be accelerated effectively?

2. Which operations should remain on CPU?

3. How fast is GPU construction of full (d^{-2}) weights?

4. How fast is thresholded or KNN-limited (d^{-2})?

5. Is dense or CSR better for nearly complete matrices?

6. How much faster is spatial lag on GPU?

7. How much faster is Global Moran’s I?

8. How much faster is Local Moran’s I?

9. How much faster are 999 permutations?

10. How much faster are 9,999 permutations?

11. At what observation count does GPU execution become worthwhile?

12. How much does batching multiple variables improve performance?

13. How much do CPU-to-GPU transfers reduce the speedup?

14. What changes when data remain resident on the GPU?

15. What is the difference between float32 and float64?

16. Are GPU results numerically equivalent to PySAL?

17. Are permutation p-values exactly reproducible or only statistically equivalent?

18. What are the main remaining bottlenecks?

Do not exaggerate results.

Include failed or negative results when relevant.

---

# 30. Target Hardware

The project should be designed to run on:

* a local NVIDIA RTX 3060 with 12 GB VRAM;
* NVIDIA T4;
* NVIDIA L4;
* NVIDIA A100.

Do not assume all hardware is available during the first development stage.

The benchmark system must record the hardware used for every result.

Avoid hardcoding GPU-specific assumptions.

---

# 31. Acceptance Criteria for the MVP

The MVP is complete when all of the following are true:

1. The project installs successfully.

2. The package uses a proper `src` structure.

3. CPU-only execution works.

4. GPU detection works.

5. Automatic CPU fallback works.

6. An official open spatial dataset is loaded reproducibly.

7. Queen weights are available through the CPU baseline.

8. KNN weights are available through the CPU baseline.

9. Inverse-distance-squared weights are implemented independently.

10. Dense and CSR (d^{-2}) representations are supported.

11. Spatial lag is implemented on CPU and GPU.

12. Global Moran’s I is implemented on CPU and GPU.

13. Local Moran’s I is implemented on CPU and GPU.

14. Permutation inference is implemented.

15. 999 permutations work.

16. 9,999 permutations work for the reference-size dataset.

17. FDR correction is implemented.

18. CPU and GPU results pass equivalence tests.

19. Benchmarks include transfer time.

20. Benchmarks include warmed GPU execution.

21. Dense and sparse performance are compared.

22. Multiple variables or periods can be processed in batches.

23. Benchmark results are exported.

24. Documentation explains all known differences from PySAL.

25. `STATUS.md` reflects the actual project state.

---

# 32. Work Order

Proceed in the following order:

```text
1. Initialize the repository and package structure.
2. Create PLAN.md and STATUS.md.
3. Create environment diagnostics.
4. Detect CPU, GPU, driver, CUDA, and CuPy compatibility.
5. Configure dependencies.
6. Audit libpysal and esda source code.
7. Create the CPU–GPU mapping document.
8. Select and document the official open dataset.
9. Implement the reproducible CPU baseline.
10. Implement backend abstraction.
11. Implement inverse-distance weights on CPU.
12. Implement inverse-distance weights on GPU.
13. Implement dense and CSR strategies.
14. Implement spatial lag.
15. Validate spatial lag against PySAL.
16. Implement Global Moran’s I.
17. Validate Global Moran’s I.
18. Implement Local Moran’s I observed statistics.
19. Validate local statistics and quadrants.
20. Implement permutation inference.
21. Validate permutation behavior.
22. Implement FDR.
23. Add multi-variable and panel support.
24. Complete automated tests.
25. Run benchmarks.
26. Create notebooks.
27. Create the benchmark report.
28. Update README.md.
29. Update STATUS.md.
30. Commit the final MVP.
```

Do not jump directly to a large GPU implementation before the CPU baseline and validation tests exist.

---

# 33. Development Conduct

While working:

* inspect existing files before modifying them;
* create modular code;
* use type hints;
* write docstrings;
* add input validation;
* avoid unnecessary dependencies;
* avoid hidden global state;
* keep random behavior explicit;
* keep benchmark code separate from library code;
* run tests frequently;
* update `STATUS.md`;
* create small Git commits;
* document design decisions;
* report limitations honestly.

Do not stop after creating placeholders.

Continue until a functional, tested, and documented first version exists, unless there is a genuine blocker.

When a blocker occurs:

1. describe the blocker;

2. record it in `STATUS.md`;

3. explain what was attempted;

4. implement any safe partial solution;

5. identify the exact next step required.

---

# 34. Expected Final Outcome

The final project should make it possible to run code similar to:

```python
from gpu_esda import inverse_distance_weights
from gpu_esda import Moran
from gpu_esda import MoranLocal

W = inverse_distance_weights(
    coordinates,
    power=2,
    row_standardize=True,
    backend="gpu",
)

global_result = Moran(
    y,
    W,
    permutations=9999,
    seed=12345,
    backend="gpu",
)

local_result = MoranLocal(
    y,
    W,
    permutations=9999,
    seed=12345,
    fdr=True,
    backend="gpu",
)
```

The project must then demonstrate, using an official open dataset and reproducible synthetic scaling tests:

* whether the results agree with PySAL;
* how long the CPU implementation takes;
* how long the GPU implementation takes;
* how much time is spent transferring data;
* where GPU processing begins to provide a meaningful advantage;
* how much time can be saved in realistic ESDA workflows.

The final purpose is to create a practical research tool that reduces the time required to test spatial-weight specifications, variables, periods, permutation counts, and LISA classifications.
