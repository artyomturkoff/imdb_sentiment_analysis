"""Compare all saved model-result files by one selected metric."""

from __future__ import annotations

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import load_json, plot_metric_across_runs


METRIC_CHOICES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def result_files() -> list:
    return sorted(
        path
        for path in METRICS_DIR.glob("*.json")
        if path.is_file() and load_json(path).get("run_id")
    )


def run(*, metric: str, split: str) -> list:
    ensure_project_dirs()
    paths = result_files()
    if not paths:
        raise SystemExit(
            "No model result files found. Train a model first with scripts/train_model.py."
        )

    runs = [load_json(path) for path in paths]
    output_paths = [FIGURES_DIR / f"compare_{split}_{metric}.png"]
    plot_metric_across_runs(
        runs,
        output_paths[0],
        split=split,
        metric=metric,
    )
    return output_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="f1", choices=METRIC_CHOICES)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    args = parser.parse_args()

    output_paths = run(metric=args.metric, split=args.split)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
