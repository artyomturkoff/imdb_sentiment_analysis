"""Compare saved model results by one metric."""

from _bootstrap import bootstrap

bootstrap()

from src.config import FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from src.evaluate import load_json, plot_metric_across_runs


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def run(*, metric: str, split: str) -> list:
    ensure_project_dirs()
    runs = [load_json(path) for path in sorted(METRICS_DIR.glob("*.json"))]
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
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    args = parser.parse_args()

    output_paths = run(metric=args.metric, split=args.split)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
