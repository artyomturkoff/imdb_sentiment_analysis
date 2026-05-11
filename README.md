# IMDb Sentiment Analysis

This is a small NLP course project for binary sentiment classification of movie reviews.
It uses the public Stanford IMDb dataset from Hugging Face: `stanfordnlp/imdb`.

The main deliverable is a reproducible classical NLP pipeline:

- baseline model: `CountVectorizer` + `MultinomialNB`
- main model: `TfidfVectorizer` with unigrams and bigrams + `LogisticRegression`
- three experiment tiers: small, medium, and large
- saved metrics, figures, trained models, and a command-line prediction demo

The project is run through small scripts in the `scripts/` folder. There is no `main.py`
wrapper, because running the steps one by one makes the experiment order clearer and easier
to explain in the report.

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
python scripts/visualise_results.py --tier small
python scripts/run_medium.py
python scripts/visualise_results.py --tier medium
python scripts/run_large.py
python scripts/visualise_results.py --tier large
python scripts/compare_results.py
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

The script saves `results/metrics/small.json`, a small-tier model comparison plot, and
confusion matrices for every small-tier model run.

To create separate charts for each small-tier model after this run:

```bash
python scripts/visualise_results.py --tier small
```

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

The script saves `results/metrics/medium.json`, a medium-tier model comparison plot, and
confusion matrices for the medium baseline and medium main model.

To create separate charts for each medium-tier model after this run:

```bash
python scripts/visualise_results.py --tier medium
```

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

The script saves `results/metrics/large.json`, the final trained model files, a large-tier
model comparison plot, confusion matrices for the final baseline and final main model, and
the error-analysis table.

To create separate charts for each large-tier model after this run:

```bash
python scripts/visualise_results.py --tier large
```

## Model-by-Model Visualisation

The script `scripts/visualise_results.py` creates separate figures for each saved model
run. It does not train models again. It reads the metric JSON files and saves:

- one metrics chart per model run
- one confusion matrix per model run

For one tier:

```bash
python scripts/visualise_results.py --tier small
python scripts/visualise_results.py --tier medium
python scripts/visualise_results.py --tier large
```

For all tiers after all training scripts have finished:

```bash
python scripts/visualise_results.py --tier all
```

By default, it visualises the test results. To visualise validation results instead:

```bash
python scripts/visualise_results.py --tier all --split validation
```

### 4. Final Comparison

```bash
python scripts/compare_results.py
```

This script does not train any model. It reads the saved metric files from the three tiers
and creates the final comparison figures. Run it after the small, medium, and large scripts
have finished.

It saves:

- `results/figures/all_model_results.png`
- `results/figures/performance_by_tier.png`

## Optional Command

To regenerate only the older performance-by-tier figure from existing metric files, use:

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

The final comparison script is separate from training. This is useful because it lets the
experiment scripts produce their own local outputs, and then the final script collects the
finished results into report-ready figures.

## Results And Artefacts

The scripts save outputs in these locations:

- split indices: `data/splits/small_split.json`, `medium_split.json`, `large_split.json`
- metrics: `results/metrics/small.json`, `medium.json`, `large.json`
- final models: `models/baseline_nb.joblib`, `models/main_lr.joblib`
- per-tier performance plots: `results/figures/small_model_performance.png`,
  `medium_model_performance.png`, `large_model_performance.png`
- per-model confusion matrices: `results/figures/*_confusion_matrix.png`
- per-model metric charts: `results/figures/*_metrics.png`
- final all-model comparison: `results/figures/all_model_results.png`
- main-model tier chart: `results/figures/performance_by_tier.png`
- error analysis examples: `results/error_analysis.md`

The comparison between models is mainly read from the metric JSON files. Each run contains
accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a classification report.

The final visual comparison is:

- `small_model_performance.png`, `medium_model_performance.png`, and
  `large_model_performance.png`: compare the models inside each tier
- `*_metrics.png`: shows accuracy, precision, recall, F1, and ROC-AUC for one model run
- `*_confusion_matrix.png`: shows correct and wrong predictions for each saved model run
- `all_model_results.png`: compares all saved model runs across the project
- `performance_by_tier.png`: shows how the selected main model's F1 score changes from
  small to medium to large

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
python scripts/visualise_results.py --tier small
python scripts/run_medium.py
python scripts/visualise_results.py --tier medium
python scripts/run_large.py
python scripts/visualise_results.py --tier large
python scripts/compare_results.py
python demo/predict_review.py --text "A slow start, but a powerful ending."
```

## Files

- `src/`: reusable dataset, preprocessing, model, evaluation, and prediction code
- `scripts/`: experiment scripts for the three tiers
- `demo/`: command-line prediction demo
- `data/splits/`: saved split indices
- `MODEL_CARD.md`: short notes about data, model, results, and limits
