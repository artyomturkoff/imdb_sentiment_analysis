"""Text cleaning steps used before vectorising movie reviews."""

import html
import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import NLTK_DATA_DIR


# These words are sentiment-important, so stop-word removal keeps them.
NEGATORS = {"no", "not", "nor", "never"}

# Common contractions are expanded before punctuation is removed.
_NEGATION_CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}
_NEGATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in _NEGATION_CONTRACTIONS) + r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Preprocessor:
    """Small callable object so sklearn can save the pipeline."""

    remove_stopwords: bool = True
    lemmatise: bool = True

    def __call__(self, text: str) -> str:
        """Let sklearn call this object as a vectoriser preprocessor."""

        return preprocess_text(
            text,
            remove_stopwords=self.remove_stopwords,
            lemmatise=self.lemmatise,
        )


def normalize_variant(variant: str) -> str:
    """Accept short variant names used in the project plan."""

    value = variant.strip().lower()
    aliases = {
        "minimal": "a",
        "required": "b",
        "required_cleaning": "b",
        "lemmatised": "c",
        "lemmatized": "c",
        "wordnet": "c",
    }
    value = aliases.get(value, value)
    if value not in {"a", "b", "c"}:
        raise ValueError(f"Unknown preprocessing variant: {variant!r}")
    return value


def get_preprocessor(variant: str) -> Preprocessor:
    """Return one of the three preprocessing variants."""

    variant = normalize_variant(variant)
    if variant == "a":
        return Preprocessor(remove_stopwords=False, lemmatise=False)
    if variant == "b":
        return Preprocessor(remove_stopwords=True, lemmatise=False)
    return Preprocessor(remove_stopwords=True, lemmatise=True)


def expand_negations(text: str) -> str:
    """Keep words like not before punctuation is removed."""

    text = text.replace("\u2019", "'")

    # Known contractions are safer to expand from the explicit dictionary.
    def replace_known(match: re.Match[str]) -> str:
        return _NEGATION_CONTRACTIONS[match.group(0).lower()]

    text = _NEGATION_PATTERN.sub(replace_known, text)
    return re.sub(r"\b(\w+)n't\b", r"\1 not", text)


def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatise: bool = True,
) -> str:
    """Clean one review with the chosen variant."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Basic text cleaning: HTML, case, negations, non-letters, and extra spaces.
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = expand_negations(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Very short tokens usually add noise in this bag-of-words setting.
    tokens = [token for token in text.split() if len(token) > 1]

    if remove_stopwords:
        # Variant B and C remove common words but keep important negators.
        stop_words = get_stop_words()
        tokens = [token for token in tokens if token not in stop_words]

    if lemmatise:
        # Variant C reduces related word forms, for example films -> film.
        lemmatizer = get_lemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def ensure_nltk_resources() -> None:
    """Download the small NLTK resources needed for this project."""

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = str(NLTK_DATA_DIR)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    for resource_paths, package_name in (
        (("corpora/stopwords", "corpora/stopwords.zip"), "stopwords"),
        (("corpora/wordnet", "corpora/wordnet.zip"), "wordnet"),
        (("corpora/omw-1.4", "corpora/omw-1.4.zip"), "omw-1.4"),
    ):
        # NLTK may store resources as folders or zip files, so both are checked.
        if has_nltk_resource(resource_paths):
            continue
        downloaded = nltk.download(package_name, download_dir=data_dir, quiet=True)
        if not downloaded or not has_nltk_resource(resource_paths):
            raise RuntimeError(
                f"NLTK resource {package_name!r} is missing. "
                f"Download it with: python -m nltk.downloader -d {data_dir} {package_name}"
            )


def has_nltk_resource(resource_paths: tuple[str, ...]) -> bool:
    """Check whether at least one possible NLTK resource path exists."""

    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            continue
    return False


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    """Load English stop words once and remove sentiment negators from the set."""

    ensure_nltk_resources()
    return set(stopwords.words("english")) - NEGATORS


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    """Create the WordNet lemmatiser once for repeated preprocessing calls."""

    ensure_nltk_resources()
    return WordNetLemmatizer()
