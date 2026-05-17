"""Text cleaning used before vectorising reviews."""

import html
import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config import NLTK_DATA_DIR


NEGATORS = {"no", "not", "nor", "never"}
VARIANT_SETTINGS = {
    "a": {"remove_stopwords": False, "lemmatise": False},
    "b": {"remove_stopwords": True, "lemmatise": False},
    "c": {"remove_stopwords": True, "lemmatise": True},
}

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
    """Callable preprocessor for sklearn pipelines."""

    remove_stopwords: bool = True
    lemmatise: bool = True

    def __call__(self, text: str) -> str:
        return preprocess_text(
            text,
            remove_stopwords=self.remove_stopwords,
            lemmatise=self.lemmatise,
        )


def get_preprocessor(variant: str) -> Preprocessor:
    """Return one preprocessing variant."""

    return Preprocessor(**VARIANT_SETTINGS[variant.lower()])


def expand_negations(text: str) -> str:
    """Expand negation contractions."""

    text = text.replace("\u2019", "'")

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
    """Clean one review."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = expand_negations(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [token for token in text.split() if len(token) > 1]

    if remove_stopwords:
        stop_words = get_stop_words()
        tokens = [token for token in tokens if token not in stop_words]

    if lemmatise:
        lemmatizer = get_lemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def ensure_nltk_resources() -> None:
    """Download missing NLTK resources."""

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = str(NLTK_DATA_DIR)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet.zip": "wordnet",
        "corpora/omw-1.4.zip": "omw-1.4",
    }
    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, download_dir=data_dir, quiet=True)


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    """English stop words without negators."""

    ensure_nltk_resources()
    return set(stopwords.words("english")) - NEGATORS


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_resources()
    return WordNetLemmatizer()
