"""Metrics and plots for the experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import shorten

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    """Calculate the scores used in the report."""

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
    """Get positive-class probabilities for ROC-AUC."""

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


def plot_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    *,
    title: str,
    output_path: Path,
) -> None:
    """Save the confusion matrix used in the report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=[0, 1],
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        values_format="d",
        cmap="Blues",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_performance_by_tier(metrics_paths: list[Path], output_path: Path) -> None:
    """Draw a simple chart showing F1 over the three tiers."""

    rows = []
    for path in metrics_paths:
        if not path.exists():
            continue
        metrics = load_json(path)
        selected_variant = metrics.get("selected_variant")
        runs = metrics.get("runs", [])
        candidates = [
            run
            for run in runs
            if run.get("model") == "main_lr"
            and (selected_variant is None or run.get("variant") == selected_variant)
        ]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda run: run.get("validation", {}).get("f1", 0.0),
        )
        test_metrics = selected.get("test", {})
        rows.append((metrics["tier"], test_metrics.get("f1", 0.0)))

    if not rows:
        return

    tiers, f1_scores = zip(*rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(tiers, f1_scores, color=["#5b8def", "#47a878", "#db6b5f"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 score")
    plt.title("Main model performance by tier")
    for bar, score in zip(bars, f1_scores):
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


def plot_tier_model_performance(
    payload: dict,
    output_path: Path,
    *,
    split: str = "test",
    metric: str = "f1",
) -> None:
    """Save a bar chart comparing all model runs inside one tier."""

    rows = []
    for run in payload.get("runs", []):
        score = run.get(split, {}).get(metric)
        if score is None:
            continue
        label = f"{run.get('model', 'model')}\nvariant {str(run.get('variant', '')).upper()}"
        rows.append((label, float(score), str(run.get("model", ""))))

    if not rows:
        return

    labels, scores, model_names = zip(*rows)
    colours = [model_colour(model_name) for model_name in model_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(7, len(rows) * 1.15)
    plt.figure(figsize=(width, 4.5))
    bars = plt.bar(labels, scores, color=colours)
    plt.ylim(0.0, 1.0)
    plt.ylabel(metric.upper() if metric == "f1" else metric.title())
    plt.title(f"{payload.get('tier', 'Tier').title()} tier {split} {metric.upper()} comparison")
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


def plot_all_model_results(
    metrics_paths: list[Path],
    output_path: Path,
    *,
    split: str = "test",
    metric: str = "f1",
) -> None:
    """Compare all saved model runs across all tiers."""

    rows = []
    for path in metrics_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        tier = str(payload.get("tier", path.stem))
        for run in payload.get("runs", []):
            score = run.get(split, {}).get(metric)
            if score is None:
                continue
            model_name = str(run.get("model", "model"))
            variant = str(run.get("variant", "")).upper()
            label = f"{tier}\n{model_name}\n{variant}"
            rows.append((label, float(score), model_name))

    if not rows:
        return

    labels, scores, model_names = zip(*rows)
    colours = [model_colour(model_name) for model_name in model_names]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(9, len(rows) * 0.95)
    plt.figure(figsize=(width, 5))
    bars = plt.bar(labels, scores, color=colours)
    plt.ylim(0.0, 1.0)
    plt.ylabel(metric.upper() if metric == "f1" else metric.title())
    plt.title(f"All model runs by {split} {metric.upper()}")
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


def model_colour(model_name: str) -> str:
    if model_name == "baseline_nb":
        return "#5b8def"
    if model_name == "main_lr":
        return "#47a878"
    return "#8f8f8f"


def write_error_analysis(
    model,
    texts: list[str],
    labels: list[int],
    output_path: Path,
    *,
    limit: int = 25,
) -> None:
    """Save wrong predictions so they can be discussed in the report."""

    predictions = [int(value) for value in model.predict(texts)]
    scores = predict_positive_scores(model, texts)

    wrong_rows = []
    for index, (text, gold, predicted) in enumerate(zip(texts, labels, predictions)):
        if gold == predicted:
            continue
        confidence = None
        if scores is not None:
            confidence = float(scores[index]) if predicted == 1 else 1.0 - float(scores[index])
        wrong_rows.append(
            {
                "index": index,
                "gold": LABEL_NAMES[int(gold)],
                "predicted": LABEL_NAMES[int(predicted)],
                "confidence": confidence,
                "cause": infer_error_cause(text),
                "snippet": clean_snippet(text),
            }
        )

    wrong_rows = wrong_rows[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Error Analysis\n\n")
        handle.write(
            "Representative misclassified reviews from the final large-tier main model.\n\n"
        )
        handle.write("| # | Gold | Predicted | Confidence | Likely cause | Snippet |\n")
        handle.write("|---|---|---|---:|---|---|\n")
        for row_number, row in enumerate(wrong_rows, start=1):
            confidence = (
                "" if row["confidence"] is None else f"{row['confidence']:.3f}"
            )
            handle.write(
                f"| {row_number} | {row['gold']} | {row['predicted']} | "
                f"{confidence} | {row['cause']} | {row['snippet']} |\n"
            )


def infer_error_cause(text: str) -> str:
    lowered = text.lower()
    word_count = len(lowered.split())
    if any(marker in lowered for marker in ("not", "n't", "never", "no ")):
        return "negation or contrast"
    if any(marker in lowered for marker in ("but", "however", "although", "though")):
        return "mixed sentiment"
    if any(marker in lowered for marker in ("sarcasm", "irony", "ironic")):
        return "sarcasm or irony"
    if word_count > 350:
        return "very long review"
    if any(marker in lowered for marker in ("plot", "character", "scene", "story")):
        return "plot-heavy wording"
    return "ambiguous wording"


def clean_snippet(text: str, *, width: int = 220) -> str:
    text = text.replace("|", "/").replace("\n", " ")
    text = " ".join(text.split())
    return shorten(text, width=width, placeholder="...")
