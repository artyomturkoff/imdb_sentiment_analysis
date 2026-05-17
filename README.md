# IMDb Sentiment Analysis

This is a IU – NLP Project - Sentiment Analysis on Movie Reviews, course ID: DLBAIPNLP01. The
task is binary sentiment classification: negative review or positive review.

The project uses the Stanford IMDb dataset from Hugging Face:

```text
stanfordnlp/imdb
```

I used two classical machine learning models:

- `naive_bayes_baseline`: `CountVectorizer` with `MultinomialNB`
- `tfidf_logreg_main`: `TfidfVectorizer` with unigrams/bigrams and `LogisticRegression`

There are also three preprocessing variants:

- `a`: basic text cleaning
- `b`: basic cleaning and stop-word removal, but keeping `not`, `no`, `nor`, and `never`
- `c`: variant `b` plus WordNet lemmatisation

The code was run on a MacBook with an Apple M2 chip and 16 GB memory. This is not a hard
requirement, but it gives some context for running time. The experiments should be
fine on most modern laptops.

## Project Files

```text
src/                         reusable project code
scripts/generate_subset.py   creates saved train/validation/test splits
scripts/train_model.py       trains one selected model
scripts/visualise_model.py   creates figures for one saved run
scripts/compare_results.py   compares saved runs by one metric
demo/predict_review.py       predicts one review with a saved model
data/splits/                 saved split indices
models/                      saved sklearn pipelines after training
results/metrics/             JSON metrics after training
results/figures/             plots after visualising
MODEL_CARD.md                short model notes and example results
```

## Setup

Python 3.11 or newer is recommended.

With `uv`:

```bash
uv sync --extra dev
```

If the virtual environment is not activated, use `uv run python` instead of `python` in
the script commands below.

With normal `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The main libraries are `datasets`, `scikit-learn`, `nltk`, `matplotlib`, `numpy`, and
`joblib`.

## Data

The raw IMDb dataset is not stored in this repository. When you run the first data command,
Hugging Face `datasets` downloads it into:

```text
data/raw/huggingface/
```

The repository keeps only the split index files in `data/splits/`. This is enough to
recreate the same small, medium, and large experiments after the dataset is downloaded.

## Reproduce the Small Run

The model card reports a small run with preprocessing variant `b`. To recreate it, run:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/train_model.py --subset small --variant b --model naive_bayes_baseline
python scripts/train_model.py --subset small --variant b --model tfidf_logreg_main
python scripts/visualise_model.py --model-name small_tfidf_logreg_main_b
python scripts/compare_results.py --metric f1 --split test
```

This creates files like:

```text
models/small_tfidf_logreg_main_b.joblib
results/metrics/small_tfidf_logreg_main_b.json
results/figures/small_tfidf_logreg_main_b_test_metrics.png
```

## Other Dataset Sizes

The project already has split files for these sizes:

| Subset | Train | Validation | Test |
|---|---:|---:|---:|
| `small` | 2,000 | 500 | 5,000 |
| `medium` | 8,000 | 2,000 | 5,000 |
| `large` | 20,000 | 5,000 | 25,000 |

You can regenerate them with:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/generate_subset.py --subset medium --random-seed 42 --train-size 8000 --validation-size 2000 --test-size 5000
python scripts/generate_subset.py --subset large --random-seed 42 --train-size 20000 --validation-size 5000 --test-size all
```

## Train Models

Training runs one model at a time:

```bash
python scripts/train_model.py --subset small --variant b --model naive_bayes_baseline
python scripts/train_model.py --subset small --variant b --model tfidf_logreg_main
```

Run names use this format:

```text
<subset>_<model>_<variant>
```

Examples:

```text
small_naive_bayes_baseline_b
small_tfidf_logreg_main_b
```

Each run saves a `.joblib` model and one JSON metrics file.

## Visualise and Compare

Create figures for one model:

```bash
python scripts/visualise_model.py --model-name small_tfidf_logreg_main_b
```

Compare all saved result files by one metric:

```bash
python scripts/compare_results.py --metric f1 --split test
```

The available comparison metrics are:

```text
accuracy
precision
recall
f1
roc_auc
```

## Demo Prediction

After a model is trained, predict one review:

```bash
python demo/predict_review.py \
  --model small_tfidf_logreg_main_b \
  --text "A slow start, but a powerful ending."
```

The script reads from `models/`, so only the model name is needed.

## Notes

Validation data is selected from the official IMDb training split. Test data comes from the
official IMDb test split. This keeps the final test score separate from model selection.

The larger runs can take longer, mainly because of TF-IDF feature extraction and Logistic
Regression training.

Generated models, metrics, figures, and raw dataset files are local outputs and are ignored
by git.
