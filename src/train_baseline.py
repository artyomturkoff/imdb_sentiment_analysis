"""Train the Naive Bayes baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.config import DEFAULT_VARIANT, MODELS_DIR
from src.data_loader import TierData, load_tier_data
from src.evaluate import evaluate_model
from src.experiment import run_summary
from src.features import make_baseline_pipeline
from src.preprocess import normalize_variant


MODEL_NAME = "baseline_nb"


def train_baseline_model(tier_data: TierData, *, variant: str = DEFAULT_VARIANT):
    variant = normalize_variant(variant)
    model = make_baseline_pipeline(variant)
    model.fit(tier_data.train_texts, tier_data.train_labels)
    return model


def train_and_evaluate_baseline(
    tier_data: TierData,
    *,
    variant: str = DEFAULT_VARIANT,
    save_model_path: Path | None = None,
) -> tuple[object, dict]:
    variant = normalize_variant(variant)
    model = train_baseline_model(tier_data, variant=variant)
    validation_metrics = evaluate_model(
        model, tier_data.validation_texts, tier_data.validation_labels
    )
    test_metrics = evaluate_model(model, tier_data.test_texts, tier_data.test_labels)

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_model_path)

    return model, run_summary(
        model_name=MODEL_NAME,
        variant=variant,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    tier_data = load_tier_data(args.tier)
    save_path = None
    if args.save_model:
        save_path = MODELS_DIR / f"{MODEL_NAME}_{args.tier}_{args.variant}.joblib"
    _, metrics = train_and_evaluate_baseline(
        tier_data,
        variant=args.variant,
        save_model_path=save_path,
    )
    print(metrics)


if __name__ == "__main__":
    main()
