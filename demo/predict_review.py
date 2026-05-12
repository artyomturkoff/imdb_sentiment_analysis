"""Demo script for predicting the sentiment of one review."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    # The demo is run directly, so it adds the project root before importing src.
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR
from src.predict import load_model, predict_sentiment


def resolve_model_path(model_name: str) -> Path:
    """Return the saved model path for a model name such as small_main_lr_b."""

    path = Path(model_name)
    if path.name != model_name:
        raise SystemExit(
            "Use only the model name, for example: --model small_main_lr_b"
        )

    # Accept both small_main_lr_b and small_main_lr_b.joblib as names.
    if path.suffix == ".joblib":
        return MODELS_DIR / path.name
    return MODELS_DIR / f"{model_name}.joblib"


def main() -> None:
    """Load one trained model and classify one review from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Raw movie review text to classify")
    parser.add_argument(
        "--model",
        required=True,
        help="Saved model name from the models directory, for example small_main_lr_b",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at {model_path}. Train it with scripts/train_model.py first."
        )

    pipeline = load_model(model_path)
    result = predict_sentiment(args.text, pipeline)
    confidence = result["confidence"]
    # Some models may not provide probabilities, so confidence is optional.
    if confidence is None:
        print(result["label"])
    else:
        print(f"{result['label']} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()
