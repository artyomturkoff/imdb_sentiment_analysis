"""Load IMDb and make the train/validation/test splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_NAME,
    DEFAULT_VARIANT,
    HF_CACHE_DIR,
    LABEL_NAMES,
    RANDOM_STATE,
    TIER_SPECS,
    ensure_project_dirs,
    split_path,
)


@dataclass(frozen=True)
class RawImdbData:
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


@dataclass(frozen=True)
class TierData:
    tier: str
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
    """Load the labelled IMDb reviews from Hugging Face."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install project dependencies before loading data: pip install -r requirements.txt"
        ) from exc

    ensure_project_dirs()
    imdb = load_dataset(DATASET_NAME, cache_dir=str(HF_CACHE_DIR))
    return RawImdbData(
        train_texts=list(imdb["train"]["text"]),
        train_labels=[int(label) for label in imdb["train"]["label"]],
        test_texts=list(imdb["test"]["text"]),
        test_labels=[int(label) for label in imdb["test"]["label"]],
    )


def ensure_splits(raw: RawImdbData, *, force: bool = False) -> None:
    """Create split files if they are not already present."""

    ensure_project_dirs()
    payloads = build_split_payloads(raw)
    for tier, payload in payloads.items():
        path = split_path(tier)
        if force or not path.exists():
            write_json(path, payload)


def load_tier_data(
    tier: str,
    *,
    raw: RawImdbData | None = None,
    force_rebuild_splits: bool = False,
) -> TierData:
    """Return the texts and labels for one experiment tier."""

    tier = tier.lower()
    if tier not in TIER_SPECS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(TIER_SPECS)}")

    raw = raw or load_imdb_dataset()
    ensure_splits(raw, force=force_rebuild_splits)

    payload = read_json(split_path(tier))
    train_indices = [int(index) for index in payload["train_indices"]]
    validation_indices = [int(index) for index in payload["validation_indices"]]
    test_indices = [int(index) for index in payload["test_indices"]]

    return TierData(
        tier=tier,
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


def build_split_payloads(raw: RawImdbData) -> dict[str, dict]:
    """Build the three project splits.

    The official IMDb test set is kept apart. Validation examples come only
    from the official train split, as described in the project plan.
    """

    train_indices = list(range(len(raw.train_labels)))
    large_train, large_validation = train_test_split(
        train_indices,
        test_size=TIER_SPECS["large"].validation_size,
        stratify=raw.train_labels,
        random_state=RANDOM_STATE,
    )

    medium_train = stratified_subset(
        large_train,
        raw.train_labels,
        TIER_SPECS["medium"].train_size,
    )
    small_train = stratified_subset(
        medium_train,
        raw.train_labels,
        TIER_SPECS["small"].train_size,
    )
    medium_validation = stratified_subset(
        large_validation,
        raw.train_labels,
        TIER_SPECS["medium"].validation_size,
    )
    small_validation = stratified_subset(
        medium_validation,
        raw.train_labels,
        TIER_SPECS["small"].validation_size,
    )
    fixed_test = stratified_subset(
        range(len(raw.test_labels)),
        raw.test_labels,
        TIER_SPECS["small"].test_size or 0,
    )

    split_data = {
        "small": (small_train, small_validation, fixed_test, "official_test_fixed_5000"),
        "medium": (
            medium_train,
            medium_validation,
            fixed_test,
            "official_test_fixed_5000",
        ),
        "large": (
            list(large_train),
            list(large_validation),
            list(range(len(raw.test_labels))),
            "official_test_full",
        ),
    }

    payloads: dict[str, dict] = {}
    for tier, (tier_train, tier_validation, tier_test, test_source) in split_data.items():
        payloads[tier] = {
            "dataset": DATASET_NAME,
            "tier": tier,
            "random_state": RANDOM_STATE,
            "default_preprocessing_variant": DEFAULT_VARIANT,
            "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
            "counts": {
                "train": len(tier_train),
                "validation": len(tier_validation),
                "test": len(tier_test),
            },
            "test_source": test_source,
            "train_indices": tier_train,
            "validation_indices": tier_validation,
            "test_indices": tier_test,
        }
    return payloads


def stratified_subset(indices, labels: list[int], size: int) -> list[int]:
    """Take a smaller balanced subset from a list of dataset indices."""

    indices = [int(index) for index in indices]
    if size > len(indices):
        raise ValueError(f"Requested {size} examples from only {len(indices)} indices")
    if size == len(indices):
        return indices

    subset, _ = train_test_split(
        indices,
        train_size=size,
        stratify=[labels[index] for index in indices],
        random_state=RANDOM_STATE,
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
