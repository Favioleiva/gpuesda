"""Deterministic local stencil definitions on regular grids."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stencil:
    name: str
    offsets: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    radius: int

    def __post_init__(self) -> None:
        if not self.offsets or len(self.offsets) != len(self.weights):
            raise ValueError("a stencil needs equally sized non-empty offsets and weights")
        if (0, 0) in self.offsets:
            raise ValueError("the stencil center must be excluded")


def rook_stencil() -> Stencil:
    return Stencil("rook", ((-1, 0), (0, -1), (0, 1), (1, 0)), (1.0,) * 4, 1)


def queen_stencil() -> Stencil:
    offsets = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    return Stencil("queen", offsets, (1.0,) * len(offsets), 1)


def inverse_distance_stencil(radius: int = 2) -> Stencil:
    """Circular d^-2 stencil: include offsets with dr²+dc² <= radius²."""
    if radius < 1:
        raise ValueError("radius must be at least one cell")
    offsets = tuple(
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if (dr, dc) != (0, 0) and dr * dr + dc * dc <= radius * radius
    )
    weights = tuple(1.0 / (dr * dr + dc * dc) for dr, dc in offsets)
    return Stencil(f"inverse_distance_r{radius}", offsets, weights, radius)
