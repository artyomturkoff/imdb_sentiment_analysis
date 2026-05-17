"""Load IMDb and prepare train/validation/test splits."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_NAME,
    HF_CACHE_DIR,
    LABEL_NAMES,
    ensure_project_dirs,
    split_path,
)


@dataclass(frozen=True)
class RawImdbData:
    """Official IMDb train and test splits."""

    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


@dataclass(frozen=True)
class SubsetData:
    """Data for one saved subset."""

    subset: str
    train_texts: list[str]
    train_labels: list[int]
    validation_texts: list[str]
    validation_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]
    test_source: str


def load_imdb_dataset() -> RawImdbData:
    """Load IMDb reviews from Hugging Face."""

    ensure_project_dirs()
    use_local_cache = imdb_cache_exists()
    if use_local_cache:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    from datasets import DownloadConfig, load_dataset
    from datasets import logging as datasets_logging

    if use_local_cache:
        datasets_logging.set_verbosity_error()

    imdb = load_dataset(
        DATASET_NAME,
        cache_dir=str(HF_CACHE_DIR),
        download_config=DownloadConfig(local_files_only=use_local_cache),
    )
    return RawImdbData(
        train_texts=list(imdb["train"]["text"]),
        train_labels=[int(label) for label in imdb["train"]["label"]],
        test_texts=list(imdb["test"]["text"]),
        test_labels=[int(label) for label in imdb["test"]["label"]],
    )


def imdb_cache_exists() -> bool:
    cache_root = HF_CACHE_DIR / "stanfordnlp___imdb"
    has_train = any(cache_root.glob("**/imdb-train.arrow"))
    has_test = any(cache_root.glob("**/imdb-test.arrow"))
    return has_train and has_test


def load_subset_data(subset: str) -> SubsetData:
    """Load one generated subset."""

    subset = subset.lower()
    raw = load_imdb_dataset()
    payload = read_json(split_path(subset))

    train_indices = payload["train_indices"]
    validation_indices = payload["validation_indices"]
    test_indices = payload["test_indices"]

    return SubsetData(
        subset=subset,
        train_texts=select(raw.train_texts, train_indices),
        train_labels=select(raw.train_labels, train_indices),
        validation_texts=select(raw.train_texts, validation_indices),
        validation_labels=select(raw.train_labels, validation_indices),
        test_texts=select(raw.test_texts, test_indices),
        test_labels=select(raw.test_labels, test_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        test_source=str(payload["test_source"]),
    )


def build_custom_split_payload(
    raw: RawImdbData,
    *,
    subset: str,
    train_size: int,
    validation_size: int,
    test_size: int | None,
    random_state: int,
) -> dict:
    """Build one split payload."""

    train_pool_size = train_size + validation_size
    train_pool = stratified_subset(
        range(len(raw.train_labels)),
        raw.train_labels,
        train_pool_size,
        random_state=random_state,
    )
    train_indices, validation_indices = train_test_split(
        train_pool,
        train_size=train_size,
        test_size=validation_size,
        stratify=[raw.train_labels[index] for index in train_pool],
        random_state=random_state,
    )

    if test_size is None:
        test_indices = list(range(len(raw.test_labels)))
        test_source = "official_test_full"
    else:
        test_indices = stratified_subset(
            range(len(raw.test_labels)),
            raw.test_labels,
            test_size,
            random_state=random_state,
        )
        test_source = f"official_test_fixed_{test_size}"

    return {
        "dataset": DATASET_NAME,
        "subset": subset,
        "tier": subset,
        "random_state": random_state,
        "label_names": LABEL_NAMES,
        "counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "test_source": test_source,
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in validation_indices],
        "test_indices": [int(index) for index in test_indices],
    }


def stratified_subset(
    indices,
    labels: list[int],
    size: int,
    *,
    random_state: int,
) -> list[int]:
    """Take a balanced subset of indices."""

    indices = list(indices)
    if size == len(indices):
        return indices

    subset, _ = train_test_split(
        indices,
        train_size=size,
        stratify=[labels[index] for index in indices],
        random_state=random_state,
    )
    return [int(index) for index in subset]


def select(values: list, indices: list[int]) -> list:
    return [values[index] for index in indices]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
