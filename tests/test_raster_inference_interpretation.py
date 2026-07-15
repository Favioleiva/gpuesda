import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gpu_esda import RasterWeights, moran_global, moran_local
from scripts.blackmarble_postprocessing import (
    CANONICAL_MANIFEST,
    OUTPUT_RELATIVE,
    build_manifest,
    inferential_cluster_mask,
    lisa_class_counts,
    moran_scatter_line,
    queen_parquet_counts,
    sha256_file,
)


def synthetic_classification():
    valid = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=bool,
    )
    island = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        dtype=bool,
    )
    significant = np.array(
        [[1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1]],
        dtype=bool,
    )
    quadrant = np.array(
        [[1, 2, 1, 3], [3, 4, 4, 2], [1, 2, 3, 4]],
        dtype=np.int8,
    )
    return valid, island, significant, quadrant


def test_inferential_mask_is_exact_and_excludes_nodata_and_islands():
    valid, island, significant, quadrant = synthetic_classification()
    actual = inferential_cluster_mask(valid, island, significant, quadrant)
    expected = valid & ~island & significant & ((quadrant >= 1) & (quadrant <= 4))
    np.testing.assert_array_equal(actual, expected)
    assert not actual[~valid].any()
    assert not actual[island].any()


def test_lisa_class_identities_for_significant_and_all_nonislands():
    valid, island, significant, quadrant = synthetic_classification()
    inferential = inferential_cluster_mask(valid, island, significant, quadrant)
    significant_counts = lisa_class_counts(quadrant, inferential)
    assert sum(significant_counts.values()) == int(inferential.sum())

    nonisland = valid & ~island
    all_counts = lisa_class_counts(quadrant, nonisland)
    assert sum(all_counts.values()) == int(valid.sum() - (valid & island).sum())


def test_queen_corrected_count_excludes_exactly_ten_synthetic_islands():
    shape = (5, 6)
    valid = np.ones(shape, dtype=bool)
    island = np.zeros(shape, dtype=bool)
    island.flat[:10] = True
    significant = np.zeros(shape, dtype=bool)
    significant.flat[:20] = True
    quadrant = np.ones(shape, dtype=np.int8)

    corrected = inferential_cluster_mask(valid, island, significant, quadrant)
    assert int(significant.sum()) == 20
    assert int(corrected.sum()) == 10
    assert int(significant.sum() - corrected.sum()) == 10


def test_permutation_pvalues_respect_monte_carlo_floor_and_keep_raster_shape():
    values = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    operator = RasterWeights.queen(np.ones_like(values, dtype=bool), backend="cpu")
    permutations = 9
    global_result = moran_global(values, operator, permutations=permutations, seed=12345)
    local_result = moran_local(values, operator, permutations=permutations, seed=12345)
    floor = 1.0 / (permutations + 1)

    assert global_result.p_sim >= floor
    assert local_result.local_i.shape == values.shape
    assert local_result.spatial_lag.shape == values.shape
    assert local_result.quadrant_code.shape == values.shape
    assert local_result.p_sim.shape == values.shape
    assert np.nanmin(local_result.p_sim) >= floor


def test_canonical_moran_scatter_line_passes_through_origin():
    moran_i = 0.625
    x, wy = moran_scatter_line(moran_i, -3.0, 5.0, points=10)
    np.testing.assert_allclose(wy, moran_i * x)
    origin = np.flatnonzero(x == 0.0)
    assert origin.size == 1
    assert wy[origin[0]] == 0.0


def test_queen_parquet_counts_separate_raw_islands(tmp_path):
    path = tmp_path / "queen.parquet"
    pq.write_table(
        pa.table(
            {
                "quadrant": ["HH", "LL", "island", "island", "LH"],
                "significant": [True, False, True, False, True],
            }
        ),
        path,
    )
    counts = queen_parquet_counts(path)
    assert counts == {
        "rows": 5,
        "raw_significant_total": 3,
        "raw_significant_nodata": 0,
        "raw_significant_islands": 1,
        "inferential_significant": 2,
    }


def test_manifest_hashes_artifacts_and_excludes_itself(tmp_path):
    output_root = tmp_path / OUTPUT_RELATIVE
    artifact = output_root / "metadata" / "example.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = build_manifest(tmp_path, output_root)
    assert not manifest["relative_path"].str.endswith(CANONICAL_MANIFEST.name).any()
    assert manifest.loc[0, "size_bytes"] == artifact.stat().st_size
    assert manifest.loc[0, "sha256"] == sha256_file(artifact)
