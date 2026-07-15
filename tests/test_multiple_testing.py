import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from gpu_esda import adjust_pvalues


def test_bh_matches_statsmodels():
    p = np.array([0.001, 0.02, 0.2, 0.8, 0.04])
    reject, adjusted, _, _ = multipletests(p, method="fdr_bh")
    actual, actual_reject = adjust_pvalues(p)
    np.testing.assert_allclose(actual, adjusted)
    np.testing.assert_array_equal(actual_reject, reject)


def test_invalid_pvalues():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, np.nan])
