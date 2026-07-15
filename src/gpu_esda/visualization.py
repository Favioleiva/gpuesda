"""Reusable figure saving with deterministic project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(
    fig: Any, filename: str, directory: str | Path = "results/figures", *, show: bool = False
) -> Path:
    target = Path(directory) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return target
