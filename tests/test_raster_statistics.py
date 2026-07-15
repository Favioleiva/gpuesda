import esda
import numpy as np
import pytest

from gpu_esda import MatrixWeightsOperator, RasterWeights, moran_global, moran_local
from gpu_esda.backend import to_numpy
from tests.test_raster_operator import explicit_weights
from gpu_esda.raster.stencils import queen_stencil, rook_stencil


@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (10, 10)])
@pytest.mark.parametrize("kind,stencil", [("rook", rook_stencil()), ("queen", queen_stencil())])
def test_observed_statistics_match_pysal(shape, kind, stencil):
    rng = np.random.default_rng(99)
    mask = np.ones(shape, dtype=bool)
    if shape != (3, 3):
        mask[shape[0] // 2, shape[1] // 2] = False
    values = rng.normal(size=shape)
    values[~mask] = np.nan
    operator = getattr(RasterWeights, kind)(mask, backend="cpu", dtype="float64")
    global_result = moran_global(values, operator)
    local_result = moran_local(values, operator)
    weights, positions = explicit_weights(mask, stencil)
    vector = np.array([values[position] for position in positions])
    expected_global = esda.Moran(vector, weights, permutations=0)
    expected_local = esda.Moran_Local(vector, weights, permutations=0)
    np.testing.assert_allclose(global_result.I, expected_global.I, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(
        [local_result.local_i[position] for position in positions],
        expected_local.Is,
        atol=1e-10,
        rtol=1e-8,
    )
    np.testing.assert_array_equal(
        [local_result.quadrant_code[position] for position in positions], expected_local.q
    )


def test_matrix_operator_uses_same_observed_core(values):
    import libpysal

    weights = libpysal.weights.lat2W(1, 5)
    weights.transform = "r"
    operator = MatrixWeightsOperator(weights.sparse, backend="cpu")
    expected = esda.Moran(values, weights, permutations=0)
    assert moran_global(values, operator).I == pytest.approx(expected.I)
    expected_local = esda.Moran_Local(values, weights, permutations=0)
    np.testing.assert_allclose(to_numpy(moran_local(values, operator).local_i), expected_local.Is)


def test_local_marks_island_explicitly():
    mask = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=bool)
    values = np.full((3, 3), np.nan)
    values[mask] = [0, 2, 9]
    result = moran_local(values, RasterWeights.rook(mask, backend="cpu", dtype="float64"))
    assert result.island[2, 2]
    assert result.quadrant_code[2, 2] == 0
