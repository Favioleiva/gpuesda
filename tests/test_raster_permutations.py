import esda
import numpy as np
import pytest

from gpu_esda import RasterWeights, gpu_available, moran_global, moran_local
from gpu_esda.backend import to_numpy
from gpu_esda.raster.permutations import choose_raster_batch_size
from gpu_esda.raster.stencils import queen_stencil
from tests.test_raster_operator import explicit_weights


def test_global_raster_permutations_are_streaming_and_reproducible():
    values = np.arange(25, dtype=float).reshape(5, 5)
    operator = RasterWeights.queen(np.ones((5, 5), bool), backend="cpu", dtype="float64")
    one = moran_global(values, operator, permutations=99, seed=42, batch_size=3)
    two = moran_global(values, operator, permutations=99, seed=42, batch_size=3)
    assert one.p_sim == two.p_sim
    assert one.EI_sim == two.EI_sim
    assert 0 < one.p_sim <= 0.5
    assert one.permutations == 99


def test_local_conditional_method_and_fdr():
    rng = np.random.default_rng(12)
    mask = np.ones((5, 5), dtype=bool)
    values = rng.normal(size=(5, 5))
    operator = RasterWeights.queen(mask, backend="cpu", dtype="float64")
    one = moran_local(values, operator, permutations=999, seed=9, batch_size=4, cell_chunk_size=7)
    two = moran_local(values, operator, permutations=999, seed=9, batch_size=4, cell_chunk_size=7)
    np.testing.assert_array_equal(one.p_sim, two.p_sim)
    assert one.p_fdr.shape == values.shape
    assert one.significant.dtype == bool
    weights, positions = explicit_weights(mask, queen_stencil())
    vector = np.array([values[position] for position in positions])
    reference = esda.Moran_Local(vector, weights, permutations=999, seed=9)
    ours = np.array([one.p_sim[position] for position in positions])
    # RNG sequences differ; both implement the same conditional null procedure.
    assert np.mean(np.abs(ours - reference.p_sim)) < 0.06


def test_auto_batch_is_positive():
    operator = RasterWeights.rook(np.ones((10, 10), bool), backend="cpu")
    assert 1 <= choose_raster_batch_size(operator, 99) <= 99


@pytest.mark.skipif(not gpu_available(), reason="CUDA/CuPy unavailable")
def test_gpu_permutation_smoke_and_residency():
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    operator = RasterWeights.queen(np.ones((5, 5), bool), backend="gpu")
    global_result = moran_global(values, operator, permutations=19, seed=5, batch_size=2)
    local_result = moran_local(
        values, operator, permutations=19, seed=5, batch_size=2, cell_chunk_size=8
    )
    assert global_result.p_sim >= 1 / 20
    assert local_result.backend == "gpu"
    assert to_numpy(local_result.p_sim).shape == values.shape
    assert np.isfinite(to_numpy(local_result.p_sim)).all()
