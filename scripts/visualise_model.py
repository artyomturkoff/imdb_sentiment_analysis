"""Visualise saved validation and test results."""

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import (
    load_json,
    plot_run_metric_summary,
    plot_saved_confusion_matrix,
)


def run(*, model_name: str, split: str) -> list:
    ensure_project_dirs()
    payload = load_json(METRICS_DIR / f"{model_name}.json")
    output_paths = []
    splits = ["validation", "test"] if split == "all" else [split]
    for split_name in splits:
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
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("all", "validation", "test"),
    )
    args = parser.parse_args()

    for path in run(model_name=args.model_name, split=args.split):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
