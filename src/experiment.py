"""Small helpers shared by the experiment scripts."""

from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_VARIANT
from src.evaluate import load_json


def choose_best_variant(
    runs: list[dict],
    *,
    model_name: str = "main_lr",
    split: str = "validation",
    metric: str = "f1",
    default: str = DEFAULT_VARIANT,
) -> str:
    """Choose the variant with the best validation F1."""

    candidates = [run for run in runs if run.get("model") == model_name]
    if not candidates:
        return default
    best = max(candidates, key=lambda run: run.get(split, {}).get(metric, -1.0))
    return str(best.get("variant", default))


def load_selected_variant(
    path: Path,
    *,
    default: str | None = DEFAULT_VARIANT,
) -> str | None:
    """Read the chosen variant from an earlier run."""

    if not path.exists():
        return default
    payload = load_json(path)
    selected = payload.get("selected_variant") or default
    return None if selected is None else str(selected)


def run_summary(
    *,
    model_name: str,
    variant: str,
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    """Keep model scores in the same JSON format."""

    return {
        "model": model_name,
        "variant": variant,
        "validation": validation_metrics,
        "test": test_metrics,
    }
