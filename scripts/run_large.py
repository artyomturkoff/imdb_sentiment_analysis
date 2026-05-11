"""Run the final large tier and save the local outputs."""

from __future__ import annotations

import argparse

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import (
    DEFAULT_VARIANT,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_project_dirs,
)
from src.data_loader import load_imdb_dataset, load_tier_data
from src.evaluate import (
    plot_confusion_matrix,
    plot_tier_model_performance,
    save_json,
    write_error_analysis,
)
from src.experiment import load_selected_variant
from src.preprocess import normalize_variant
from src.train_baseline import train_and_evaluate_baseline
from src.train_main_model import train_and_evaluate_main


def run(*, variant: str | None = None, force_splits: bool = False) -> dict:
    ensure_project_dirs()
    selected_variant = (
        variant
        or load_selected_variant(METRICS_DIR / "medium.json", default=None)
        or load_selected_variant(METRICS_DIR / "small.json", default=DEFAULT_VARIANT)
    )
    selected_variant = normalize_variant(selected_variant)

    raw = load_imdb_dataset()
    tier_data = load_tier_data(
        "large",
        raw=raw,
        force_rebuild_splits=force_splits,
    )

    baseline_model, baseline_run = train_and_evaluate_baseline(
        tier_data,
        variant=selected_variant,
        save_model_path=MODELS_DIR / "baseline_nb.joblib",
    )
    baseline_predictions = [
        int(value) for value in baseline_model.predict(tier_data.test_texts)
    ]
    plot_confusion_matrix(
        tier_data.test_labels,
        baseline_predictions,
        title=f"Large-tier baseline NB variant {selected_variant.upper()} confusion matrix",
        output_path=FIGURES_DIR
        / f"large_baseline_nb_{selected_variant}_confusion_matrix.png",
    )

    main_model, main_run = train_and_evaluate_main(
        tier_data,
        variant=selected_variant,
        save_model_path=MODELS_DIR / "main_lr.joblib",
    )

    runs = [baseline_run, main_run]
    payload = {
        "tier": "large",
        "test_source": tier_data.test_source,
        "selection_rule": "selected validation variant evaluated once on full official test",
        "selected_variant": selected_variant,
        "runs": runs,
    }
    save_json(METRICS_DIR / "large.json", payload)

    predictions = [int(value) for value in main_model.predict(tier_data.test_texts)]
    plot_confusion_matrix(
        tier_data.test_labels,
        predictions,
        title=f"Large-tier main LR variant {selected_variant.upper()} confusion matrix",
        output_path=FIGURES_DIR
        / f"large_main_lr_{selected_variant}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        tier_data.test_labels,
        predictions,
        title="Large-tier main model confusion matrix",
        output_path=FIGURES_DIR / "confusion_matrix_large.png",
    )
    write_error_analysis(
        main_model,
        tier_data.test_texts,
        tier_data.test_labels,
        RESULTS_DIR / "error_analysis.md",
        limit=25,
    )
    plot_tier_model_performance(
        payload,
        FIGURES_DIR / "large_model_performance.png",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--force-splits", action="store_true")
    args = parser.parse_args()
    payload = run(variant=args.variant, force_splits=args.force_splits)
    print(f"Saved large-tier outputs. Variant: {payload['selected_variant']}")


if __name__ == "__main__":
    main()
