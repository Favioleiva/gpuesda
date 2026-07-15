"""Validate the MVP against the bundled Columbus PySAL example."""

from __future__ import annotations

import json
from pathlib import Path

import esda
import geopandas as gpd
import libpysal
import numpy as np

from gpu_esda import Moran, MoranLocal, gpu_available, spatial_lag


def main() -> int:
    frame = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    values = frame["CRIME"].to_numpy()
    weights = libpysal.weights.Queen.from_dataframe(frame, use_index=True)
    weights.transform = "r"
    np.random.seed(12345)
    reference_global = esda.Moran(values, weights, permutations=999)
    reference_local = esda.Moran_Local(values, weights, permutations=999, seed=12345)
    record: dict[str, object] = {"dataset": "columbus", "n": len(frame), "seed": 12345}
    for backend in ["cpu", "gpu"] if gpu_available() else ["cpu"]:
        ours_global = Moran(values, weights.sparse, 999, 12345, backend)
        ours_local = MoranLocal(values, weights.sparse, 999, 12345, backend)
        lag = spatial_lag(weights.sparse, values, backend)
        record[backend] = {
            "lag_max_abs_difference": float(
                np.max(np.abs(lag - libpysal.weights.lag_spatial(weights, values)))
            ),
            "global_I": ours_global.I,
            "global_I_abs_difference": abs(ours_global.I - reference_global.I),
            "global_p_sim": ours_global.p_sim,
            "local_I_max_abs_difference": float(np.max(np.abs(ours_local.Is - reference_local.Is))),
            "quadrant_mismatches": int(np.count_nonzero(ours_local.q != reference_local.q)),
            "local_p_mean_abs_difference": float(
                np.mean(np.abs(ours_local.p_sim - reference_local.p_sim))
            ),
            "timings": {"global": ours_global.timings, "local": ours_local.timings},
        }
    target = Path("results/validation_columbus.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    deterministic = all(
        record[name][key] < 1e-10  # type: ignore[index]
        for name in record
        if name in {"cpu", "gpu"}
        for key in (
            "lag_max_abs_difference",
            "global_I_abs_difference",
            "local_I_max_abs_difference",
        )
    )
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
