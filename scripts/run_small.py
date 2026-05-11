"""Run the small tier and compare preprocessing variants."""

from __future__ import annotations

import argparse

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    VARIANT_ORDER,
    ensure_project_dirs,
)
from src.data_loader import load_imdb_dataset, load_tier_data
from src.evaluate import plot_confusion_matrix, plot_tier_model_performance, save_json
from src.experiment import choose_best_variant
from src.train_baseline import MODEL_NAME as BASELINE_NAME
from src.train_baseline import train_and_evaluate_baseline
from src.train_main_model import MODEL_NAME as MAIN_NAME
from src.train_main_model import train_and_evaluate_main


def run(*, force_splits: bool = False, save_models: bool = False) -> dict:
    ensure_project_dirs()
    raw = load_imdb_dataset()
    tier_data = load_tier_data(
        "small",
        raw=raw,
        force_rebuild_splits=force_splits,
    )

    runs = []
    for variant in VARIANT_ORDER:
        baseline_path = (
            MODELS_DIR / f"{BASELINE_NAME}_small_{variant}.joblib"
            if save_models
            else None
        )
        baseline_model, baseline_run = train_and_evaluate_baseline(
            tier_data,
            variant=variant,
            save_model_path=baseline_path,
        )
        runs.append(baseline_run)
        baseline_predictions = [
            int(value) for value in baseline_model.predict(tier_data.test_texts)
        ]
        plot_confusion_matrix(
            tier_data.test_labels,
            baseline_predictions,
            title=f"Small-tier baseline NB variant {variant.upper()} confusion matrix",
            output_path=FIGURES_DIR
            / f"small_{BASELINE_NAME}_{variant}_confusion_matrix.png",
        )

        main_path = (
            MODELS_DIR / f"{MAIN_NAME}_small_{variant}.joblib"
            if save_models
            else None
        )
        main_model, main_run = train_and_evaluate_main(
            tier_data,
            variant=variant,
            save_model_path=main_path,
        )
        runs.append(main_run)
        main_predictions = [int(value) for value in main_model.predict(tier_data.test_texts)]
        plot_confusion_matrix(
            tier_data.test_labels,
            main_predictions,
            title=f"Small-tier main LR variant {variant.upper()} confusion matrix",
            output_path=FIGURES_DIR
            / f"small_{MAIN_NAME}_{variant}_confusion_matrix.png",
        )

    selected_variant = choose_best_variant(runs)
    payload = {
        "tier": "small",
        "test_source": tier_data.test_source,
        "selection_rule": "best main_lr validation f1",
        "selected_variant": selected_variant,
        "runs": runs,
    }
    save_json(METRICS_DIR / "small.json", payload)
    plot_tier_model_performance(
        payload,
        FIGURES_DIR / "small_model_performance.png",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    payload = run(force_splits=args.force_splits, save_models=args.save_models)
    print(f"Saved small-tier metrics. Selected variant: {payload['selected_variant']}")


if __name__ == "__main__":
    main()
