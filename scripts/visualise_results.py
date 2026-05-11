"""Create separate result figures for each saved model run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import (
    load_json,
    plot_run_metric_summary,
    plot_saved_confusion_matrix,
)


TIERS = ("small", "medium", "large")


def run(*, tier: str = "all", split: str = "test") -> list[Path]:
    """Create per-model metric and confusion-matrix figures from saved JSON."""

    ensure_project_dirs()
    tiers = TIERS if tier == "all" else (tier,)

    output_paths: list[Path] = []
    for tier_name in tiers:
        metrics_path = METRICS_DIR / f"{tier_name}.json"
        if not metrics_path.exists():
            raise SystemExit(
                f"Missing {metrics_path}. Run python scripts/run_{tier_name}.py first."
            )

        payload = load_json(metrics_path)
        for run_data in payload.get("runs", []):
            stem = figure_stem(tier_name, run_data, split)
            metrics_output = FIGURES_DIR / f"{stem}_metrics.png"
            confusion_output = FIGURES_DIR / f"{stem}_confusion_matrix.png"

            plot_run_metric_summary(
                run_data,
                metrics_output,
                split=split,
                title=figure_title(tier_name, run_data, split, "metrics"),
            )
            plot_saved_confusion_matrix(
                run_data,
                confusion_output,
                split=split,
                title=figure_title(tier_name, run_data, split, "confusion matrix"),
            )

            output_paths.extend([metrics_output, confusion_output])

    return output_paths


def figure_stem(tier: str, run_data: dict, split: str) -> str:
    model_name = str(run_data.get("model", "model"))
    variant = str(run_data.get("variant", "variant")).lower()
    return f"{tier}_{model_name}_{variant}_{split}"


def figure_title(tier: str, run_data: dict, split: str, suffix: str) -> str:
    model_name = str(run_data.get("model", "model"))
    variant = str(run_data.get("variant", "")).upper()
    return f"{tier.title()} tier {model_name} variant {variant} {split} {suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default="all",
        choices=("all", *TIERS),
        help="Which tier to visualise. Use after the matching run script has finished.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("validation", "test"),
        help="Which saved split metrics to visualise.",
    )
    args = parser.parse_args()

    output_paths = run(tier=args.tier, split=args.split)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
