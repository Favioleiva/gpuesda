import numpy as np
import pytest

from gpu_esda import Moran, MoranLocal, gpu_available, inverse_distance_weights, spatial_lag


pytestmark = pytest.mark.skipif(not gpu_available(), reason="CUDA/CuPy unavailable")


@pytest.mark.parametrize("dtype,atol,rtol", [("float64", 1e-10, 1e-8), ("float32", 1e-5, 1e-4)])
def test_weights_lag_and_observed_equivalence(coordinates, values, dtype, atol, rtol):
    cpu = inverse_distance_weights(coordinates, dtype=dtype, backend="cpu", output_format="dense")
    gpu = inverse_distance_weights(coordinates, dtype=dtype, backend="gpu", output_format="dense")
    from gpu_esda.backend import to_numpy

    np.testing.assert_allclose(to_numpy(gpu), cpu, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        spatial_lag(gpu, values, "gpu"), spatial_lag(cpu, values, "cpu"), atol=atol, rtol=rtol
    )
    mg = Moran(values, gpu, 0, backend="gpu", dtype=dtype)
    mc = Moran(values, cpu, 0, backend="cpu", dtype=dtype)
    np.testing.assert_allclose(mg.I, mc.I, atol=atol, rtol=rtol)
    lg = MoranLocal(values, gpu, 0, backend="gpu", dtype=dtype)
    lc = MoranLocal(values, cpu, 0, backend="cpu", dtype=dtype)
    np.testing.assert_allclose(lg.Is, lc.Is, atol=atol, rtol=rtol)


def test_dense_and_sparse_gpu(coordinates, values):
    dense = inverse_distance_weights(coordinates, backend="gpu", output_format="dense")
    sparse = inverse_distance_weights(coordinates, backend="gpu", output_format="csr")
    np.testing.assert_allclose(
        spatial_lag(dense, values, "gpu"), spatial_lag(sparse, values, "gpu")
    )


def test_gpu_permutations_reproducible(coordinates, values):
    w = inverse_distance_weights(coordinates, backend="gpu", output_format="dense")
    one = Moran(values, w, 99, 42, "gpu", keep_simulations=True)
    two = Moran(values, w, 99, 42, "gpu", keep_simulations=True)
    np.testing.assert_array_equal(one.sim, two.sim)
