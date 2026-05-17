"""Train one or both project models."""

import argparse

import joblib

from _bootstrap import bootstrap

bootstrap()

from src.config import METRICS_DIR, MODELS_DIR, ROOT_DIR, ensure_project_dirs, split_path
from src.data_loader import load_subset_data, read_json
from src.evaluate import evaluate_model, save_json
from src.features import make_baseline_pipeline, make_main_pipeline


MODEL_FACTORIES = {
    "naive_bayes_baseline": make_baseline_pipeline,
    "tfidf_logreg_main": make_main_pipeline,
}


def run_id(subset: str, model_name: str, variant: str) -> str:
    return f"{subset}_{model_name}_{variant}"


def train_one_model(
    subset: str,
    variant: str,
    model_name: str,
    subset_data,
    split_payload: dict,
) -> dict:
    run_name = run_id(subset, model_name, variant)
    run_model_path = MODELS_DIR / f"{run_name}.joblib"
    run_metrics_path = METRICS_DIR / f"{run_name}.json"

    model = MODEL_FACTORIES[model_name](variant)
    model.fit(subset_data.train_texts, subset_data.train_labels)

    run_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_model_path)

    payload = {
        "run_id": run_name,
        "subset": subset,
        "model": model_name,
        "variant": variant,
        "model_path": str(run_model_path.relative_to(ROOT_DIR)),
        "split_path": str(split_path(subset).relative_to(ROOT_DIR)),
        "random_state": split_payload.get("random_state"),
        "counts": split_payload.get("counts", {}),
        "test_source": subset_data.test_source,
        "validation": evaluate_model(
            model,
            subset_data.validation_texts,
            subset_data.validation_labels,
        ),
        "test": evaluate_model(
            model,
            subset_data.test_texts,
            subset_data.test_labels,
        ),
    }
    save_json(run_metrics_path, payload)
    return payload


def run(*, subset: str, variant: str, model: str) -> list[dict]:
    ensure_project_dirs()
    subset = subset.lower()
    variant = variant.lower()
    subset_data = load_subset_data(subset)
    split_payload = read_json(split_path(subset))
    model_names = MODEL_FACTORIES if model == "both" else [model]
    return [
        train_one_model(subset, variant, model_name, subset_data, split_payload)
        for model_name in model_names
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--variant", required=True, type=str.lower, choices=("a", "b", "c"))
    parser.add_argument(
        "--model",
        default="both",
        choices=("both", "naive_bayes_baseline", "tfidf_logreg_main"),
    )
    args = parser.parse_args()

    payloads = run(subset=args.subset, variant=args.variant, model=args.model)
    for payload in payloads:
        metric_file = METRICS_DIR / f"{payload['run_id']}.json"
        print(f"Saved {payload['model_path']} and {metric_file.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
