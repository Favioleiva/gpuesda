"""Pure CPU post-processing and provenance for Black Marble A100 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LISA_QUADRANTS = (1, 2, 3, 4)
LISA_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
MASK_DEFINITION = "valid & ~island & significant & quadrant.isin([1, 2, 3, 4])"
OUTPUT_RELATIVE = Path("outputs/blackmarble_peru_2024-03-21_a100")
ORIGINAL_COMPARISON = Path(
    "tables/blackmarble_2024-03-21_log1000p1_a100_comparison.csv"
)
QUEEN_PARQUET = Path(
    "local_results/blackmarble_2024-03-21_log1000p1_local_queen.parquet"
)
CANONICAL_DIR = Path("canonical")
CANONICAL_COMPARISON = Path(
    "canonical/blackmarble_2024-03-21_log1000p1_a100_comparison_canonical.csv"
)
CANONICAL_INTERPRETATION = Path(
    "canonical/blackmarble_2024-03-21_log1000p1_interpretation.json"
)
CANONICAL_MANIFEST = Path("canonical/artifact_manifest_sha256.csv")


def _same_shape(**arrays: Any) -> dict[str, np.ndarray]:
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    shapes = {value.shape for value in converted.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={value.shape}" for name, value in converted.items())
        raise ValueError(f"all raster interpretation arrays must share one shape; {details}")
    return converted


def inferential_cluster_mask(
    valid: Any,
    island: Any,
    significant: Any,
    quadrant: Any,
) -> np.ndarray:
    """Return ``valid & ~island & significant & quadrant_between_1_and_4``."""

    arrays = _same_shape(
        valid=valid,
        island=island,
        significant=significant,
        quadrant=quadrant,
    )
    return (
        arrays["valid"].astype(bool, copy=False)
        & ~arrays["island"].astype(bool, copy=False)
        & arrays["significant"].astype(bool, copy=False)
        & np.isin(arrays["quadrant"], LISA_QUADRANTS)
    )


def lisa_class_counts(quadrant: Any, selection: Any) -> dict[str, int]:
    """Count HH, LH, LL, and HL within an audited selection mask."""

    arrays = _same_shape(quadrant=quadrant, selection=selection)
    selected = arrays["quadrant"][arrays["selection"].astype(bool, copy=False)]
    return {label: int(np.count_nonzero(selected == code)) for code, label in LISA_LABELS.items()}


def moran_scatter_line(
    moran_i: float,
    x_min: float,
    x_max: float,
    *,
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the canonical no-intercept Moran line ``WY = I * Y``."""

    if points < 2:
        raise ValueError("points must be at least 2")
    if not np.isfinite([moran_i, x_min, x_max]).all():
        raise ValueError("Moran line inputs must be finite")
    if x_min > x_max:
        raise ValueError("x_min must not exceed x_max")
    x = np.linspace(float(x_min), float(x_max), points, dtype=np.float64)
    if x_min < 0 < x_max and not np.any(x == 0.0):
        x = np.sort(np.append(x, 0.0))
    return x, float(moran_i) * x


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def queen_parquet_counts(path: Path) -> dict[str, int]:
    """Read only Queen classification columns and distinguish raw/inferential counts."""

    table = pq.read_table(path, columns=["quadrant", "significant"])
    significant = table["significant"].combine_chunks()
    quadrant = table["quadrant"].combine_chunks()
    raw_total = int(pc.sum(pc.cast(significant, pa.int64())).as_py())
    island = pc.equal(quadrant, pa.scalar("island"))
    significant_on_islands = int(
        pc.sum(pc.cast(pc.and_(significant, island), pa.int64())).as_py()
    )
    recognized = pc.is_in(quadrant, value_set=pa.array(["HH", "LH", "LL", "HL"]))
    inferential = int(
        pc.sum(pc.cast(pc.and_(significant, recognized), pa.int64())).as_py()
    )
    return {
        "rows": table.num_rows,
        "raw_significant_total": raw_total,
        "raw_significant_nodata": 0,
        "raw_significant_islands": significant_on_islands,
        "inferential_significant": inferential,
    }


def build_canonical_comparison(output_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Derive an unambiguous comparison without rewriting the execution CSV."""

    original_path = output_root / ORIGINAL_COMPARISON
    original = pd.read_csv(original_path)
    if original["stencil"].tolist() != ["rook", "queen", "d2_r2"]:
        raise ValueError("execution comparison must contain rook, queen, and d2_r2 in order")

    canonical = original.rename(
        columns={
            "local_FDR_significant": "inferential_significant",
            "local_FDR_share": "inferential_significant_share",
        }
    ).copy()
    canonical.insert(
        canonical.columns.get_loc("inferential_significant"),
        "raw_significant_total",
        canonical["inferential_significant"]
        + canonical["raw_significant_nodata"]
        + canonical["raw_significant_islands"],
    )

    queen_counts = queen_parquet_counts(output_root / QUEEN_PARQUET)
    queen_index = canonical.index[canonical["stencil"] == "queen"].item()
    for column in (
        "raw_significant_total",
        "raw_significant_nodata",
        "raw_significant_islands",
        "inferential_significant",
    ):
        canonical.loc[queen_index, column] = queen_counts[column]
    canonical.loc[queen_index, "inferential_significant_share"] = (
        queen_counts["inferential_significant"]
        / canonical.loc[queen_index, "valid_nonisland"]
    )

    assert queen_counts["raw_significant_total"] == 1_174_898
    assert queen_counts["raw_significant_nodata"] == 0
    assert queen_counts["raw_significant_islands"] == 10
    assert queen_counts["inferential_significant"] == 1_174_888
    assert (
        queen_counts["raw_significant_total"] - queen_counts["raw_significant_islands"]
        == queen_counts["inferential_significant"]
    )
    class_sum = (
        canonical["HH_significant"]
        + canonical["LH_significant"]
        + canonical["LL_significant"]
        + canonical["HL_significant"]
    )
    if not np.array_equal(class_sum, canonical["inferential_significant"]):
        raise ValueError("significant LISA classes do not sum to the inferential total")

    target = output_root / CANONICAL_COMPARISON
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(target, index=False)
    return canonical, queen_counts


def build_interpretation_json(
    output_root: Path,
    canonical: pd.DataFrame,
    queen_counts: dict[str, int],
) -> dict[str, Any]:
    """Write semantic corrections separately from immutable execution metadata."""

    original_metadata = Path(
        "metadata/blackmarble_2024-03-21_log1000p1_run_metadata.json"
    )
    operators: dict[str, Any] = {}
    for row in canonical.to_dict(orient="records"):
        operators[row["stencil"]] = {
            "moran_i": row["Moran_I"],
            "p_sim": row["p_sim"],
            "permutations": int(row["permutations"]),
            "islands": int(row["islands"]),
            "valid_nonisland": int(row["valid_nonisland"]),
            "raw_local_significant": int(row["raw_significant_total"]),
            "significant_on_nodata": int(row["raw_significant_nodata"]),
            "significant_on_islands": int(row["raw_significant_islands"]),
            "inferential_local_fdr_significant": int(row["inferential_significant"]),
            "inferential_local_fdr_share": row["inferential_significant_share"],
        }

    payload = {
        "artifact_role": "canonical post-run interpretation; original A100 files are immutable",
        "original_execution_metadata": original_metadata.as_posix(),
        "original_comparison_csv": ORIGINAL_COMPARISON.as_posix(),
        "canonical_comparison_csv": CANONICAL_COMPARISON.as_posix(),
        "dataset": {"product": "VNP46A2", "date": "2024-03-21", "rows": 6_111_958},
        "transformation": "log(1000 * NTL + 1)",
        "seed": 12345,
        "permutations": 999,
        "fdr": "Benjamini-Hochberg q <= 0.05",
        "inferential_mask": MASK_DEFINITION,
        "operators": operators,
        "queen_reconciliation": {
            "equation": "1,174,898 - 10 = 1,174,888",
            **queen_counts,
        },
        "parquet_availability": {
            "queen": QUEEN_PARQUET.as_posix(),
            "rook": None,
            "d2_r2": None,
            "limitation": (
                "No local Rook or d2_r2 Parquet was archived. Their global results, "
                "counts, and figures are preserved; inference was not rerun."
            ),
        },
    }
    target = output_root / CANONICAL_INTERPRETATION
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def artifact_type(path: Path) -> str:
    return {
        ".csv": "table/csv",
        ".json": "metadata/json",
        ".parquet": "local-results/parquet",
        ".png": "figure/png",
        ".zip": "archive/zip",
        ".ipynb": "notebook/ipynb",
    }.get(path.suffix.lower(), "other")


def artifact_condition(relative: str) -> tuple[str, str]:
    duplicate_map = {
        "outputs/blackmarble_peru_2024-03-21_a100/blackmarble/blackmarble_2024-03-21_log1000p1_local_queen.parquet": (
            "outputs/blackmarble_peru_2024-03-21_a100/local_results/blackmarble_2024-03-21_log1000p1_local_queen.parquet"
        ),
        "outputs/blackmarble_peru_2024-03-21_a100/blackmarble/blackmarble_2024-03-21_log1000p1_global.json": (
            "outputs/blackmarble_peru_2024-03-21_a100/metadata/blackmarble_2024-03-21_log1000p1_global.json"
        ),
        "outputs/blackmarble_peru_2024-03-21_a100/blackmarble/blackmarble_2024-03-21_log1000p1_run_metadata.json": (
            "outputs/blackmarble_peru_2024-03-21_a100/metadata/blackmarble_2024-03-21_log1000p1_run_metadata.json"
        ),
    }
    if relative in duplicate_map:
        return "duplicate", duplicate_map[relative]
    if relative.endswith(".zip"):
        return "archived", ""
    canonical_paths = {
        "notebooks/06_blackmarble_peru_a100.ipynb",
        "outputs/blackmarble_peru_2024-03-21_a100/local_results/blackmarble_2024-03-21_log1000p1_local_queen.parquet",
        "outputs/blackmarble_peru_2024-03-21_a100/metadata/blackmarble_2024-03-21_log1000p1_global.json",
        "outputs/blackmarble_peru_2024-03-21_a100/metadata/blackmarble_2024-03-21_log1000p1_run_metadata.json",
        "outputs/blackmarble_peru_2024-03-21_a100/canonical/blackmarble_2024-03-21_log1000p1_a100_comparison_canonical.csv",
        "outputs/blackmarble_peru_2024-03-21_a100/canonical/blackmarble_2024-03-21_log1000p1_interpretation.json",
    }
    if relative in canonical_paths or (
        relative.startswith("outputs/blackmarble_peru_2024-03-21_a100/figures/")
        and ("lisa_clusters_" in relative or "moran_scatter_" in relative)
    ):
        return "canonical", ""
    return "local-only", ""


def build_manifest(repo_root: Path, output_root: Path) -> pd.DataFrame:
    """Hash every run artifact except this manifest, marking aliases explicitly."""

    manifest_path = output_root / CANONICAL_MANIFEST
    paths = [path for path in output_root.rglob("*") if path.is_file() and path != manifest_path]
    paths.extend(
        path
        for path in (
            repo_root / "notebooks/06_blackmarble_peru_a100.ipynb",
            repo_root / "notebooks/06_blackmarble_peru_a100_PRE_COMPARISON_BACKUP.ipynb",
        )
        if path.is_file()
    )
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(repo_root).as_posix()
        condition, canonical_reference = artifact_condition(relative)
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_type": artifact_type(path),
                "condition": condition,
                "canonical_reference": canonical_reference,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    if CANONICAL_MANIFEST.as_posix() in set(manifest["relative_path"]):
        raise AssertionError("canonical manifest must exclude itself")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = repo_root / OUTPUT_RELATIVE
    canonical, queen_counts = build_canonical_comparison(output_root)
    build_interpretation_json(output_root, canonical, queen_counts)
    manifest = build_manifest(repo_root, output_root)
    print(f"canonical comparison: {output_root / CANONICAL_COMPARISON}")
    print(f"canonical interpretation: {output_root / CANONICAL_INTERPRETATION}")
    print(f"canonical manifest: {output_root / CANONICAL_MANIFEST} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
