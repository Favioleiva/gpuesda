import esda
import libpysal
import numpy as np

from gpu_esda import Moran, moran_panel


def test_observed_and_analytical_equivalence(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    expected = esda.Moran(values, w, permutations=0)
    result = Moran(values, w.sparse, permutations=0, backend="cpu")
    np.testing.assert_allclose(result.I, expected.I, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(result.EI, expected.EI)
    np.testing.assert_allclose(result.VI_norm, expected.VI_norm, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(result.VI_rand, expected.VI_rand, atol=1e-10, rtol=1e-8)


def test_permutations_reproducible(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    one = Moran(values, w.sparse, 99, 42, "cpu", keep_simulations=True)
    two = Moran(values, w.sparse, 99, 42, "cpu", keep_simulations=True)
    np.testing.assert_array_equal(one.sim, two.sim)
    assert one.p_sim == two.p_sim and one.to_frame().shape[0] == 1


def test_panel(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    result = moran_panel(
        np.column_stack([values, values**2]), w.sparse, permutations=0, backend="cpu"
    )
    assert len(result) == 2
