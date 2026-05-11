# IMDb Sentiment Analysis

This is a small NLP course project for binary sentiment classification of movie reviews.
It uses the public Stanford IMDb dataset from Hugging Face: `stanfordnlp/imdb`.

The main deliverable is a reproducible classical NLP pipeline:

- baseline model: `CountVectorizer` + `MultinomialNB`
- main model: `TfidfVectorizer` with unigrams and bigrams + `LogisticRegression`
- three experiment tiers: small, medium, and large
- saved metrics, figures, trained models, and a command-line prediction demo

`main.py` is only an optional convenience wrapper. The real project workflow is in the
scripts below, so the project can be run completely without using `main.py`.

## Setup

```bash
uv sync --extra dev
```

Or, with plain `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use Python 3.11 or newer. The Python install must include normal compression modules
such as `lzma`, because Hugging Face `datasets` needs them.

## Recommended Workflow

Run the project in this order:

```bash
python scripts/run_small.py
python scripts/run_medium.py
python scripts/run_large.py
```

### 1. Small Tier

```bash
python scripts/run_small.py
```

This is the first experiment and works like a smoke test. It uses:

- 2,000 training reviews
- 500 validation reviews
- a fixed 5,000-review test subset

It trains and evaluates both models for all three preprocessing variants:

- A: minimal cleaning
- B: cleaning with stop-word removal, while keeping negators
- C: B plus WordNet lemmatisation

The selected variant is the one with the best validation F1 score for the main Logistic
Regression model. This selected variant is carried forward to the medium and large tiers.

### 2. Medium Tier

```bash
python scripts/run_medium.py
```

This run uses the selected preprocessing variant from the small tier. It uses:

- 8,000 training reviews
- 2,000 validation reviews
- the same fixed 5,000-review test subset as the small tier

It trains the Naive Bayes baseline and the main Logistic Regression model again on the
medium data. The baseline is not calculated once and reused. It is trained separately for
each tier, because each tier has a different amount of training data.

### 3. Large Tier

```bash
python scripts/run_large.py
```

This is the final evaluation run. It uses:

- 20,000 training reviews
- 5,000 validation reviews
- the full official 25,000-review IMDb test split

It trains the final baseline model and the final main model using the selected preprocessing
variant. It also saves the final trained models and creates the figures used for the report.

The large-tier test result is the main result to report, because it uses the full held-out
IMDb test set.

## Optional Commands

You can also run the same workflow through `main.py`:

```bash
python main.py run-small
python main.py run-medium
python main.py run-large
```

This is only a wrapper around the scripts. It is useful, but not required.

To regenerate only the performance-by-tier figure from existing metric files:

```bash
python scripts/make_figures.py
```

## Training, Evaluation, And Testing

Training happens inside each tier script. For each model, the pipeline is fitted only on
the training split for that tier. The vectoriser is inside the scikit-learn `Pipeline`, so it
is fitted only on training data and does not see validation or test reviews.

Evaluation is done on both validation and test data:

- validation scores are used for model or preprocessing decisions
- test scores are saved as reported performance
- the final large-tier test set should be treated as the main final result

The baseline is trained and evaluated separately for each tier:

- small tier: baseline and main model are both run for variants A, B, and C
- medium tier: baseline and main model are run with the selected variant
- large tier: final baseline and final main model are run with the selected variant

## Results And Artefacts

The scripts save outputs in these locations:

- split indices: `data/splits/small_split.json`, `medium_split.json`, `large_split.json`
- metrics: `results/metrics/small.json`, `medium.json`, `large.json`
- final models: `models/baseline_nb.joblib`, `models/main_lr.joblib`
- final confusion matrix: `results/figures/confusion_matrix_large.png`
- performance chart: `results/figures/performance_by_tier.png`
- error analysis examples: `results/error_analysis.md`

The comparison between models is mainly read from the metric JSON files. Each run contains
accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a classification report.

The final visual comparison is:

- `confusion_matrix_large.png`: shows where the final main model is correct or wrong
- `performance_by_tier.png`: shows how the main model's F1 score changes from small to
  medium to large

Generated datasets, saved models, metrics, and figures are ignored by git. Split index
files in `data/splits/` are kept because they only contain public integer indices and make
the runs easier to repeat.

## Demo

After running the large tier, classify one new review with:

```bash
python demo/predict_review.py --text "A slow start, but a powerful ending."
```

The demo loads `models/main_lr.joblib` and prints a sentiment label with a confidence
score.

Run `python scripts/run_large.py` first if `models/main_lr.joblib` does not exist.

## Usual Full Run

For a clean project run from start to finish, use:

```bash
python scripts/run_small.py
python scripts/run_medium.py
python scripts/run_large.py
python demo/predict_review.py --text "A slow start, but a powerful ending."
```

## Files

- `src/`: reusable dataset, preprocessing, model, evaluation, and prediction code
- `scripts/`: experiment scripts for the three tiers
- `demo/`: command-line prediction demo
- `data/splits/`: saved split indices
- `MODEL_CARD.md`: short notes about data, model, results, and limits
