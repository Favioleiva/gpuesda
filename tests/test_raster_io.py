import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gpu_esda import BlackMarbleRaster


def test_blackmarble_reconstruction_preserves_zero_and_nodata(tmp_path):
    grid_path = tmp_path / "grid.parquet"
    daily_path = tmp_path / "daily.parquet"
    grid = pa.table(
        {"cell_id": [10, 11, 12, 13], "grid_row": [5, 5, 6, 7], "grid_col": [8, 9, 8, 10]}
    )
    daily = pa.table({"cell_id": [10, 11, 12, 13], "ntl": [0.0, 2.0, None, 5.0]})
    pq.write_table(grid, grid_path)
    pq.write_table(daily, daily_path)
    raster = BlackMarbleRaster.from_parquet(grid_path, daily_path)
    assert raster.values.shape == (3, 3)
    assert raster.values[0, 0] == 0
    assert raster.mask[0, 0]
    assert raster.structural_mask[1, 0] and not raster.mask[1, 0]
    assert raster.metadata["missing_values"] == 1
    assert raster.origin == (5, 8)


def test_blackmarble_aligns_different_cell_order(tmp_path):
    grid_path = tmp_path / "grid.parquet"
    daily_path = tmp_path / "daily.parquet"
    pq.write_table(pa.table({"cell_id": [1, 2], "grid_row": [0, 0], "grid_col": [0, 1]}), grid_path)
    pq.write_table(pa.table({"cell_id": [2, 1], "ntl": [20.0, 10.0]}), daily_path)
    raster = BlackMarbleRaster.from_parquet(grid_path, daily_path)
    np.testing.assert_array_equal(raster.values, [[10, 20]])
