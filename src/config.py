"""Small set of paths and constants used by the project scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
DATASET_NAME = "stanfordnlp/imdb"
LABEL_NAMES = {0: "negative", 1: "positive"}
DEFAULT_VARIANT = "c"
VARIANT_ORDER = ("a", "b", "c")

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
NLTK_DATA_DIR = RAW_DATA_DIR / "nltk_data"
HF_CACHE_DIR = RAW_DATA_DIR / "huggingface"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class TierSpec:
    """How many reviews are used in one experiment tier."""

    train_size: int
    validation_size: int
    test_size: int | None


TIER_SPECS = {
    "small": TierSpec(train_size=2_000, validation_size=500, test_size=5_000),
    "medium": TierSpec(train_size=8_000, validation_size=2_000, test_size=5_000),
    "large": TierSpec(
        train_size=20_000,
        validation_size=5_000,
        test_size=None,
    ),
}


def ensure_project_dirs() -> None:
    """Create folders that scripts write into."""

    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        SPLITS_DIR,
        NLTK_DATA_DIR,
        HF_CACHE_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def split_path(tier: str) -> Path:
    return SPLITS_DIR / f"{tier}_split.json"
