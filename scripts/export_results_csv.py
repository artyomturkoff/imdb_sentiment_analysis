"""Export saved validation and test metrics to CSV."""

import argparse
import csv
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from src.config import METRICS_DIR
from src.evaluate import load_json


PRIMARY_COLUMNS = ("n_examples", "accuracy", "precision", "recall", "f1", "roc_auc")


def clean_column_name(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def flatten_metric_value(prefix: str, value) -> dict:
    if isinstance(value, int | float | str | bool):
        return {clean_column_name(prefix): value}

    if isinstance(value, dict):
        row = {}
        for key, nested_value in value.items():
            row.update(flatten_metric_value(f"{prefix}_{key}", nested_value))
        return row

    if isinstance(value, list):
        row = {}
        for index, nested_value in enumerate(value):
            row.update(flatten_metric_value(f"{prefix}_{index}", nested_value))
        return row

    return {}


def flatten_metrics(metrics: dict) -> dict:
    row = {}
    for metric_name, metric_value in metrics.items():
        row.update(flatten_metric_value(metric_name, metric_value))
    return row


def load_rows() -> list[dict]:
    rows = []
    for metrics_path in sorted(METRICS_DIR.glob("*.json")):
        payload = load_json(metrics_path)
        for result_type in ("validation", "test"):
            row = {
                "model name": payload["run_id"],
                "result type": result_type,
            }
            row.update(flatten_metrics(payload[result_type]))
            rows.append(row)
    return rows


def csv_columns(rows: list[dict]) -> list[str]:
    columns = {"model name", "result type"}
    for row in rows:
        columns.update(row)

    primary_columns = [column for column in PRIMARY_COLUMNS if column in columns]
    other_columns = sorted(columns - {"model name", "result type"} - set(primary_columns))
    return ["model name", "result type", *primary_columns, *other_columns]


def run(*, output_path: Path) -> Path:
    rows = load_rows()
    columns = csv_columns(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output_path = run(output_path=args.output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
