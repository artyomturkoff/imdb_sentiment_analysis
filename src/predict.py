"""Load a saved model and predict one review."""

from pathlib import Path

import joblib

from src.config import LABEL_NAMES


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def predict_sentiment(review: str, pipeline) -> dict:
    """Return label and confidence."""

    prediction = int(pipeline.predict([review])[0])
    probabilities = pipeline.predict_proba([review])[0]
    class_index = list(pipeline.classes_).index(prediction)

    return {
        "label": LABEL_NAMES[prediction],
        "prediction": prediction,
        "confidence": round(float(probabilities[class_index]), 4),
    }
