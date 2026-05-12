"""Create the two sklearn models used in the project."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocess import get_preprocessor


def make_baseline_pipeline(variant: str = "c") -> Pipeline:
    """Baseline: word counts with Naive Bayes."""

    return Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    preprocessor=get_preprocessor(variant),
                    lowercase=False,
                    ngram_range=(1, 1),
                    min_df=2,
                    binary=True,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


def make_main_pipeline(variant: str = "c") -> Pipeline:
    """Main model: TF-IDF features with Logistic Regression."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=get_preprocessor(variant),
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1_000,
                    C=1.0,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
