"""Small set of paths and constants used by the project scripts."""

from pathlib import Path


RANDOM_STATE = 42
DATASET_NAME = "stanfordnlp/imdb"
LABEL_NAMES = {0: "negative", 1: "positive"}

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
