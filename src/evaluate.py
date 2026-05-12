"""Metrics and plots used by the scripts."""

import json
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import LABEL_NAMES


def evaluate_model(model, texts: list[str], labels: list[int]) -> dict:
    """Return the main classification scores."""

    predictions = [int(value) for value in model.predict(texts)]
    positive_scores = predict_positive_scores(model, texts)

    metrics = {
        "n_examples": len(labels),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(
            float(precision_score(labels, predictions, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            output_dict=True,
            zero_division=0,
        ),
    }
    if positive_scores is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(labels, positive_scores)), 4)
    return metrics


def predict_positive_scores(model, texts: list[str]):
    """Positive-class scores for ROC-AUC."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1)
        return probabilities[:, positive_index]
    return None


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metric_across_runs(
    runs: list[dict],
    output_path: Path,
    *,
    split: str,
    metric: str,
) -> None:
    """Plot one metric for saved runs."""

    rows = []
    for run in runs:
        score = run.get(split, {}).get(metric)
        if score is None:
            continue
        rows.append(
            (
                str(run.get("run_id", "model")),
                float(score),
                str(run.get("model", "")),
            )
        )

    if not rows:
        return

    labels, scores, model_names = zip(*rows)
    colours = [model_colour(model_name) for model_name in model_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(9, len(rows) * 0.9)
    plt.figure(figsize=(width, 5))
    bars = plt.bar(labels, scores, color=colours)
    plt.ylim(0.0, 1.0)
    plt.ylabel(pretty_metric_name(metric))
    plt.title(f"{split.title()} {pretty_metric_name(metric)} across saved model runs")
    plt.xticks(rotation=35, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 0.02, 0.98),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_run_metric_summary(
    run: dict,
    output_path: Path,
    *,
    split: str = "test",
    title: str | None = None,
) -> None:
    """Save metric bars for one model."""

    split_metrics = run.get(split, {})
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = [
        (metric_name, split_metrics[metric_name])
        for metric_name in metric_names
        if metric_name in split_metrics
    ]
    if not rows:
        return

    labels = [pretty_metric_name(metric_name) for metric_name, _ in rows]
    scores = [float(score) for _, score in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, scores, color=model_colour(str(run.get("model", ""))))
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title or model_run_title(run, split, "metric summary"))
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 0.02, 0.98),
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_saved_confusion_matrix(
    run: dict,
    output_path: Path,
    *,
    split: str = "test",
    title: str | None = None,
) -> None:
    """Save a confusion matrix image."""

    matrix = run.get(split, {}).get("confusion_matrix")
    if matrix is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix),
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    display.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(title or model_run_title(run, split, "confusion matrix"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def model_run_title(run: dict, split: str, suffix: str) -> str:
    model_name = str(run.get("model", "model"))
    variant = str(run.get("variant", "")).upper()
    return f"{model_name} variant {variant} {split} {suffix}"


def pretty_metric_name(metric_name: str) -> str:
    if metric_name == "f1":
        return "F1"
    if metric_name == "roc_auc":
        return "ROC-AUC"
    return metric_name.title()


def model_colour(model_name: str) -> str:
    if model_name == "naive_bayes_baseline":
        return "#5b8def"
    if model_name == "tfidf_logreg_main":
        return "#47a878"
    return "#8f8f8f"
