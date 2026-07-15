"""Minimal batched panel benchmark."""

from gpu_esda import inverse_distance_weights, moran_panel
from gpu_esda.benchmarking import benchmark_call, synthetic_coordinates, synthetic_panel


def main() -> int:
    coordinates = synthetic_coordinates(500)
    values = synthetic_panel(500, 10, 4)
    weights = inverse_distance_weights(coordinates, backend="auto")
    print(
        benchmark_call(
            lambda: moran_panel(values, weights, 99, backend="auto"), "auto", 3
        ).to_dict()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
