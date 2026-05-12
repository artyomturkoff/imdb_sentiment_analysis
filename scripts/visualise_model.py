"""Visualise validation and/or test results for one trained model."""

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


def selected_splits(split: str) -> list[str]:
    if split == "all":
        return ["validation", "test"]
    return [split]


def run(*, model_name: str, split: str = "all") -> list[Path]:
    ensure_project_dirs()
    metrics_path = METRICS_DIR / f"{model_name}.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"Missing {metrics_path}. Train it first with scripts/train_model.py."
        )

    payload = load_json(metrics_path)
    output_paths: list[Path] = []
    for split_name in selected_splits(split):
        metrics_output = FIGURES_DIR / f"{model_name}_{split_name}_metrics.png"
        confusion_output = FIGURES_DIR / f"{model_name}_{split_name}_confusion_matrix.png"
        plot_run_metric_summary(
            payload,
            metrics_output,
            split=split_name,
            title=f"{model_name} {split_name} metrics",
        )
        plot_saved_confusion_matrix(
            payload,
            confusion_output,
            split=split_name,
            title=f"{model_name} {split_name} confusion matrix",
        )
        output_paths.extend([metrics_output, confusion_output])
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model run name, for example small_main_lr_b",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "validation", "test"),
    )
    args = parser.parse_args()

    for path in run(model_name=args.model_name, split=args.split):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
