import libpysal
import numpy as np
import pytest

from gpu_esda import RasterWeights, gpu_available
from gpu_esda.backend import to_numpy
from gpu_esda.raster.stencils import inverse_distance_stencil, queen_stencil, rook_stencil


def explicit_weights(mask, stencil):
    positions = list(zip(*np.nonzero(mask), strict=True))
    index = {position: i for i, position in enumerate(positions)}
    neighbors, weights = {}, {}
    for position, i in index.items():
        row, col = position
        items = []
        for offset, weight in zip(stencil.offsets, stencil.weights, strict=True):
            neighbor = (row + offset[0], col + offset[1])
            if neighbor in index:
                items.append((index[neighbor], weight))
        neighbors[i] = [item[0] for item in items]
        weights[i] = [item[1] for item in items]
    result = libpysal.weights.W(neighbors, weights, silence_warnings=True)
    result.transform = "r"
    return result, positions


@pytest.mark.parametrize(
    "factory,stencil",
    [
        (lambda mask: RasterWeights.rook(mask, backend="cpu", dtype="float64"), rook_stencil()),
        (lambda mask: RasterWeights.queen(mask, backend="cpu", dtype="float64"), queen_stencil()),
        (
            lambda mask: RasterWeights.inverse_distance(
                mask, radius=2, backend="cpu", dtype="float64"
            ),
            inverse_distance_stencil(2),
        ),
    ],
)
def test_lag_matches_explicit_pysal(factory, stencil):
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    values = np.arange(25, dtype=float).reshape(5, 5)
    operator = factory(mask)
    actual = operator.apply(values)
    weights, positions = explicit_weights(mask, stencil)
    vector = np.array([values[position] for position in positions])
    expected = libpysal.weights.lag_spatial(weights, vector)
    np.testing.assert_allclose([actual[position] for position in positions], expected)
    assert actual[2, 2] == 0


def test_coast_hole_island_and_valid_zero():
    mask = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
    values = np.full(mask.shape, np.nan, dtype=np.float64)
    values[mask] = [0.0, 2.0, 4.0, 7.0]
    operator = RasterWeights.rook(mask, backend="cpu", dtype="float64")
    lag = operator.apply(values)
    assert lag[0, 0] == 3.0  # zero remains a valid neighbor value
    assert lag[1, 3] == 0.0
    assert operator.islands()[1, 3]
    assert operator.s0() == 3.0


def test_batch_and_simulated_tile_border():
    mask = np.ones((3, 6), dtype=bool)
    values = np.arange(18, dtype=float).reshape(3, 6)
    operator = RasterWeights.rook(mask, backend="cpu", dtype="float64")
    batch = np.stack([values, values + 1])
    result = operator.apply(batch)
    assert result.shape == batch.shape
    # Columns 2/3 represent a tile boundary; adjacency must remain continuous.
    assert result[0, 1, 2] == np.mean([values[0, 2], values[1, 1], values[1, 3], values[2, 2]])


def test_nodata_inside_mask_rejected():
    mask = np.ones((3, 3), dtype=bool)
    values = np.ones((3, 3))
    values[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RasterWeights.queen(mask, backend="cpu").apply(values)


@pytest.mark.skipif(not gpu_available(), reason="CUDA/CuPy unavailable")
@pytest.mark.parametrize("kind", ["rook", "queen", "inverse_distance"])
def test_cpu_gpu_equivalence(kind):
    rng = np.random.default_rng(4)
    mask = rng.random((10, 10)) > 0.25
    values = rng.normal(size=(10, 10)).astype(np.float32)
    values[~mask] = np.nan
    kwargs = {"radius": 2} if kind == "inverse_distance" else {}
    cpu = getattr(RasterWeights, kind)(mask, backend="cpu", **kwargs)
    gpu = getattr(RasterWeights, kind)(mask, backend="gpu", **kwargs)
    np.testing.assert_allclose(to_numpy(gpu.apply(values)), cpu.apply(values), atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(to_numpy(gpu.islands()), cpu.islands())
