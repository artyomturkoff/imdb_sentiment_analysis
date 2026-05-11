"""Create final comparison figures from saved experiment metrics."""

from __future__ import annotations

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from pathlib import Path

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import plot_all_model_results, plot_performance_by_tier


METRICS_PATHS = [
    METRICS_DIR / "small.json",
    METRICS_DIR / "medium.json",
    METRICS_DIR / "large.json",
]


def run() -> list[Path]:
    ensure_project_dirs()
    missing = [path for path in METRICS_PATHS if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing metric files. Run the tier scripts first: "
            "python scripts/run_small.py, python scripts/run_medium.py, "
            f"python scripts/run_large.py. Missing: {missing_names}"
        )

    output_paths = [
        FIGURES_DIR / "all_model_results.png",
        FIGURES_DIR / "performance_by_tier.png",
    ]
    plot_all_model_results(METRICS_PATHS, output_paths[0])
    plot_performance_by_tier(METRICS_PATHS, output_paths[1])
    return output_paths


def main() -> None:
    output_paths = run()
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
