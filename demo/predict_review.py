"""Predict sentiment for one review."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR
from src.predict import load_model, predict_sentiment


def resolve_model_path(model_name: str) -> Path:
    """Resolve a saved model name."""

    if model_name.endswith(".joblib"):
        return MODELS_DIR / model_name
    return MODELS_DIR / f"{model_name}.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    pipeline = load_model(model_path)
    result = predict_sentiment(args.text, pipeline)
    print(f"{result['label']} (confidence: {result['confidence']:.4f})")


if __name__ == "__main__":
    main()
