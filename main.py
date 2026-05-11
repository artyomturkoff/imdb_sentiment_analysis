from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMDb movie review sentiment analysis runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    small = subparsers.add_parser("run-small", help="Run small-tier experiments")
    small.add_argument("--force-splits", action="store_true")
    small.add_argument("--save-models", action="store_true")

    medium = subparsers.add_parser("run-medium", help="Run medium-tier experiments")
    medium.add_argument("--variant", default=None)
    medium.add_argument("--force-splits", action="store_true")
    medium.add_argument("--save-models", action="store_true")

    large = subparsers.add_parser("run-large", help="Run final large-tier evaluation")
    large.add_argument("--variant", default=None)
    large.add_argument("--force-splits", action="store_true")

    predict = subparsers.add_parser("predict", help="Classify one review")
    predict.add_argument("--text", required=True)
    predict.add_argument("--model", default="models/main_lr.joblib")

    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    if args.command == "run-small":
        from run_small import run

        payload = run(
            force_splits=args.force_splits,
            save_models=args.save_models,
        )
        print(f"Saved small-tier metrics. Selected variant: {payload['selected_variant']}")
        return

    if args.command == "run-medium":
        from run_medium import run

        payload = run(
            variant=args.variant,
            force_splits=args.force_splits,
            save_models=args.save_models,
        )
        print(f"Saved medium-tier metrics. Variant: {payload['selected_variant']}")
        return

    if args.command == "run-large":
        from run_large import run

        payload = run(variant=args.variant, force_splits=args.force_splits)
        print(f"Saved large-tier outputs. Variant: {payload['selected_variant']}")
        return

    if args.command == "predict":
        from src.predict import load_model, predict_sentiment

        model_path = Path(args.model)
        if not model_path.exists():
            raise SystemExit(
                f"Model not found at {model_path}. Run python scripts/run_large.py first."
            )
        result = predict_sentiment(args.text, load_model(model_path))
        confidence = result["confidence"]
        if confidence is None:
            print(result["label"])
        else:
            print(f"{result['label']} (confidence: {confidence:.4f})")
        return


if __name__ == "__main__":
    main()
