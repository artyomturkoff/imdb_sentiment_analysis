"""Use a saved model to predict one review."""

from pathlib import Path

import joblib

from src.config import LABEL_NAMES


def load_model(model_path: str | Path):
    """Load a saved sklearn pipeline from a .joblib file."""

    return joblib.load(model_path)


def predict_sentiment(review: str, pipeline) -> dict:
    """Return the label and confidence for raw review text."""

    prediction = int(pipeline.predict([review])[0])
    confidence = None

    # Most sklearn classifiers here expose probabilities, which gives a useful confidence.
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([review])[0]
        classes = list(getattr(pipeline, "classes_", [0, 1]))
        class_index = classes.index(prediction)
        confidence = round(float(probabilities[class_index]), 4)

    return {
        "label": LABEL_NAMES[prediction],
        "prediction": prediction,
        "confidence": confidence,
    }
