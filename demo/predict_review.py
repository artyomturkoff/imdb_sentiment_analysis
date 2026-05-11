"""Demo script for predicting the sentiment of one review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import DEFAULT_MODEL_PATH, load_model, predict_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Raw movie review text to classify")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at {model_path}. Run scripts/run_large.py first."
        )

    pipeline = load_model(model_path)
    result = predict_sentiment(args.text, pipeline)
    confidence = result["confidence"]
    if confidence is None:
        print(result["label"])
    else:
        print(f"{result['label']} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()
