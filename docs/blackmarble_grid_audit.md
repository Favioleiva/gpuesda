# Black Marble Peru Grid Audit

## Verdict

- Continuous national integer index: **True**
- Reconstructable without a spatial join: **True**
- Grid/daily one-to-one by `cell_id`: **True**

## Dimensions and coverage

- Grid rows: 6,111,958; daily rows: 6,111,958
- Rectangular shape: 4,395 × 3,042
- Valid positions: 6,111,958; holes: 7,257,632
- Coverage: 45.7154%
- Orientation: rows north_to_south; columns west_to_east

## Topology checks

- Unique `cell_id` in grid/daily: True / True
- Unique `(grid_row, grid_col)`: True
- Constant row/column tile offsets: True / True
- Cross-tile Rook edges: horizontal 2,316, vertical 1,624

## Memory

- One float32 rectangular array: 51.0 MiB
- Three float32 result arrays: 153.0 MiB
- One float64 rectangular array: 102.0 MiB
- Boolean mask: 12.8 MiB

The machine-readable audit, hashes, bounds, and per-tile summaries are in `results/blackmarble_grid_audit.json`. No topology is inferred from coordinates: stencil adjacency is permitted only because integer national positions are unique and tile offsets are consistent.
