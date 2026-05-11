"""Regenerate the performance chart from saved metrics."""

from __future__ import annotations

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import plot_performance_by_tier


def main() -> None:
    ensure_project_dirs()
    plot_performance_by_tier(
        [
            METRICS_DIR / "small.json",
            METRICS_DIR / "medium.json",
            METRICS_DIR / "large.json",
        ],
        FIGURES_DIR / "performance_by_tier.png",
    )
    print("Saved results/figures/performance_by_tier.png")


if __name__ == "__main__":
    main()
