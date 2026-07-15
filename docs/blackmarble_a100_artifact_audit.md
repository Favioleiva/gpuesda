# A100 artifact audit

Audit performed read-only after the completed A100 run. No inference or CUDA operation was repeated.

## Inventory

The original output tree contains 28 files: 15 PNG, 6 CSV, 4 JSON, 2 Parquet, and 1 ZIP. Three derived canonical files were added afterward: corrected comparison CSV, interpretation JSON, and reproducible manifest.

| Category | Files | Size notes |
|---|---:|---|
| Individual LISA cluster maps | 3 | 266–408 KiB each |
| Individual Moran scatter plots | 3 | 542–550 KiB each |
| Comparative compositions | 2 | 501 KiB LISA; 1.02 MiB scatter |
| Other figures | 7 | 46 KiB–2.70 MiB |
| Tables | 5 plus manifest | 250–1,607 bytes |
| Metadata/results JSON | 2 unique, each stored twice | 3,026 and 3,588 bytes |
| Queen local Parquet | 1 unique, stored twice | 74,762,977 bytes (71.30 MiB) |
| ZIP archive | 1 | 78,727,036 bytes (75.08 MiB) |

Required figures are present:

- `lisa_clusters_rook_log1000p1_fdr999.png`
- `lisa_clusters_queen_log1000p1_fdr999.png`
- `lisa_clusters_d2_r2_log1000p1_fdr999.png`
- `lisa_clusters_comparison_log1000p1_fdr999.png`
- `moran_scatter_rook_log1000p1_bhfdr999.png`
- `moran_scatter_queen_log1000p1_bhfdr999.png`
- `moran_scatter_d2_r2_log1000p1_bhfdr999.png`
- `moran_scatter_comparison_log1000p1_bhfdr999.png`

All 15 PNGs open successfully and have valid PNG metadata. The ZIP passes its CRC integrity test.

## Canonical hashes

| Artifact | SHA-256 |
|---|---|
| Comparison CSV | `3b620639681058df23ad517ddeb999ac0cda14ba050966bdef5dd45f0f8acf6c` |
| Global JSON | `3ec5394a6dbb396e39f147599071172b3cbc35bda3b40faa09656bffeac1974e` |
| Run metadata JSON | `935f46e946131e1e40a96660c5d6f99e00544cfc0d91b1aa2f33a703644e1b4d` |
| Queen local Parquet | `3d3ededdf43c40850d53c180bfe9de15b75993c70bc43cfa675aebbdc368778d` |
| ZIP archive | `4b5ca18eb1646a791c695af088b4cbbeb52389cead131d0c2d3803a0ddbf4f91` |

The two Parquet paths are byte-identical. The JSON copies under `blackmarble/` and `metadata/` are also byte-identical; these are aliases, not conflicting results.

## Consistency findings

- The original comparison CSV currently contains exactly `rook`, `queen`, and `d2_r2`, and its stored Queen row reports `raw_significant_islands=10`. A separately generated canonical table makes the raw total and inferential total explicit and does not rewrite this execution artifact.
- Significant HH, LH, LL, and HL counts sum exactly to `local_FDR_significant` for every operator.
- `valid_nonisland + islands` equals 6,111,958 for every operator.
- The Queen Parquet has 1,174,898 raw `significant=True` rows, including its 10 island rows. The comparison CSV correctly reports the sanitized 1,174,888 inferential clusters.
- Both run-metadata aliases still label the raw preliminary value 1,174,898 as `local_fdr_significant`. The files are preserved as immutable run evidence. `canonical/blackmarble_2024-03-21_log1000p1_interpretation.json` records `raw_local_significant=1,174,898`, `significant_on_islands=10`, and `inferential_local_fdr_significant=1,174,888`.
- The manifest lists 24 canonical archive members and all listed external paths exist. Twenty-three recorded sizes are exact. Its self-reported size is stale because the manifest grew after calculating that row.
- The manifest does not provide hashes and intentionally does not enumerate the ZIP or the duplicate `metadata/` and `local_results/` aliases.

The original `tables/output_manifest.csv` remains unchanged. Its reproducible replacement is `canonical/artifact_manifest_sha256.csv`, generated after the canonical table and interpretation JSON. It records relative path, byte size, SHA-256, artifact type, condition, and canonical reference. It deliberately excludes itself.

The only local cell-level Parquet is Queen, duplicated byte for byte under `blackmarble/` and `local_results/`. No Rook or `d2_r2` Parquet was found. Their global statistics, FDR counts, and maps remain available, so this archival limitation does not block closure.

## Git and publication policy

Recommended normal Git content:

- canonical notebook, despite its 6.74 MiB executed outputs;
- comparison CSV and the two small metadata JSON files;
- eight required individual/comparative figures;
- documentation, interpretation helpers, and deterministic tests.

Recommended local-only content:

- pre-comparison notebook backup;
- duplicate aliases;
- smoke, memory, and intermediate tables/figures unless specifically needed for a release archive.

Recommended Hugging Face publication:

- the canonical 71.30 MiB Queen Parquet;
- the 75.08 MiB ZIP archive;
- optionally a checksum manifest generated outside the immutable run directory.

Git LFS is an acceptable alternative if the repository must directly version those two binaries. They should not be added to ordinary Git history.
