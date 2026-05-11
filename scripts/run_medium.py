"""Run the medium tier with the selected preprocessing variant."""

from __future__ import annotations

import argparse

try:
    from _bootstrap import bootstrap
except ImportError:
    from scripts._bootstrap import bootstrap

bootstrap()

from src.config import DEFAULT_VARIANT, METRICS_DIR, MODELS_DIR, ensure_project_dirs
from src.data_loader import load_imdb_dataset, load_tier_data
from src.evaluate import save_json
from src.experiment import load_selected_variant
from src.preprocess import normalize_variant
from src.train_baseline import MODEL_NAME as BASELINE_NAME
from src.train_baseline import train_and_evaluate_baseline
from src.train_main_model import MODEL_NAME as MAIN_NAME
from src.train_main_model import train_and_evaluate_main


def run(
    *,
    variant: str | None = None,
    force_splits: bool = False,
    save_models: bool = False,
) -> dict:
    ensure_project_dirs()
    selected_variant = variant or load_selected_variant(
        METRICS_DIR / "small.json",
        default=DEFAULT_VARIANT,
    )
    selected_variant = normalize_variant(selected_variant)

    raw = load_imdb_dataset()
    tier_data = load_tier_data(
        "medium",
        raw=raw,
        force_rebuild_splits=force_splits,
    )

    baseline_path = (
        MODELS_DIR / f"{BASELINE_NAME}_medium_{selected_variant}.joblib"
        if save_models
        else None
    )
    _, baseline_run = train_and_evaluate_baseline(
        tier_data,
        variant=selected_variant,
        save_model_path=baseline_path,
    )

    main_path = (
        MODELS_DIR / f"{MAIN_NAME}_medium_{selected_variant}.joblib"
        if save_models
        else None
    )
    _, main_run = train_and_evaluate_main(
        tier_data,
        variant=selected_variant,
        save_model_path=main_path,
    )

    runs = [baseline_run, main_run]
    payload = {
        "tier": "medium",
        "test_source": tier_data.test_source,
        "selection_rule": "small-tier selected variant carried forward",
        "selected_variant": selected_variant,
        "runs": runs,
    }
    save_json(METRICS_DIR / "medium.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    payload = run(
        variant=args.variant,
        force_splits=args.force_splits,
        save_models=args.save_models,
    )
    print(f"Saved medium-tier metrics. Variant: {payload['selected_variant']}")


if __name__ == "__main__":
    main()
