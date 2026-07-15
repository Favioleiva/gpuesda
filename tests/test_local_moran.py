import esda
import libpysal
import numpy as np

from gpu_esda import MoranLocal, local_moran_panel


def test_observed_local_and_quadrants(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    expected = esda.Moran_Local(values, w, permutations=0)
    result = MoranLocal(values, w.sparse, permutations=0, backend="cpu")
    np.testing.assert_allclose(result.Is, expected.Is, atol=1e-10, rtol=1e-8)
    np.testing.assert_array_equal(result.q, expected.q)
    assert set(result.quadrant) <= {"HH", "LH", "LL", "HL"}


def test_conditional_permutations_fdr_reproducible(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    one = MoranLocal(values, w.sparse, 99, 42, "cpu", keep_simulations=True)
    two = MoranLocal(values, w.sparse, 99, 42, "cpu", keep_simulations=True)
    np.testing.assert_array_equal(one.sim, two.sim)
    np.testing.assert_array_equal(one.p_sim, two.p_sim)
    assert one.to_frame().shape == (5, 13)


def test_local_panel(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    frame = local_moran_panel(
        np.column_stack([values, values**2]), w.sparse, permutations=0, backend="cpu"
    )
    assert len(frame) == 10
