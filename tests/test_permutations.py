import libpysal
import numpy as np

from gpu_esda.permutations import choose_batch_size, global_permutations


def test_batch_size_documented_values():
    assert choose_batch_size(100, 999, np.float64, 10**9) in {64, 128, 256, 512, 1024}


def test_streaming_without_simulations(values):
    w = libpysal.weights.lat2W(1, 5)
    w.transform = "r"
    z = values - values.mean()
    observed = 5 / w.s0 * (z @ (w.sparse @ z)) / (z @ z)
    result = global_permutations(z, w.sparse, observed, 99, 7, "cpu")
    assert result.simulations is None
    assert 0 < result.pvalue <= 0.5
