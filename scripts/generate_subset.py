"""Generate an IMDb subset split."""

import argparse

from _bootstrap import bootstrap

bootstrap()

from src.config import ensure_project_dirs, split_path
from src.data_loader import build_custom_split_payload, load_imdb_dataset, write_json


def parse_test_size(value: str) -> int | None:
    if value.lower() in {"all", "full"}:
        return None
    return int(value)


def run(
    *,
    subset: str,
    train_size: int,
    validation_size: int,
    test_size: int | None,
    random_seed: int,
) -> dict:
    ensure_project_dirs()
    raw = load_imdb_dataset()
    payload = build_custom_split_payload(
        raw,
        subset=subset,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_seed,
    )
    write_json(split_path(subset), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--test-size", type=parse_test_size, required=True)
    args = parser.parse_args()

    payload = run(
        subset=args.subset.lower(),
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    counts = payload["counts"]
    print(
        f"Saved {split_path(args.subset.lower())} "
        f"(train={counts['train']}, validation={counts['validation']}, "
        f"test={counts['test']}, seed={payload['random_state']})"
    )


if __name__ == "__main__":
    main()
