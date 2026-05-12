# IMDb Sentiment Analysis

This is a college NLP project for binary sentiment classification of movie reviews. It uses
the public Stanford IMDb dataset from Hugging Face: `stanfordnlp/imdb`.

The project now uses a more manual and flexible workflow. Instead of one fixed script for
small, medium, and large experiments, you create the subset you want, train the variant you
want, inspect the saved results, and then decide what to run next.

The two core models are:

- baseline model: `CountVectorizer` + `MultinomialNB`
- main model: `TfidfVectorizer` with unigrams and bigrams + `LogisticRegression`

The preprocessing variants are:

- A: minimal cleaning
- B: A plus stop-word removal, while keeping negators such as `not`, `no`, `nor`, and
  `never`
- C: B plus WordNet lemmatisation

## Setup

```bash
uv sync --extra dev
```

Or with plain `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Use Python 3.11 or newer. The Python install must include normal compression modules such
as `lzma`, because Hugging Face `datasets` needs them.

## Dataset Download

The raw IMDb dataset is not stored in this repository. The folder `data/raw/` is ignored by
git, because the dataset is downloaded locally from Hugging Face and should not be committed
to a public repository.

The project uses:

```text
stanfordnlp/imdb
```

The easiest way to download it is to run the first subset-generation command. The script
will call Hugging Face `datasets`, download the IMDb data if it is not already cached, and
then save only the split indices in `data/splits/`:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
```

The local cache is stored under:

```text
data/raw/huggingface/
```

If you want to download the dataset before running any project script, use:

```bash
python -c "from datasets import load_dataset; load_dataset('stanfordnlp/imdb', cache_dir='data/raw/huggingface')"
```

After the dataset is cached once, the scripts can reuse the local copy. If the cache is not
present, the first run needs an internet connection.

## Workflow

The usual order is:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/train_model.py --subset small --variant b
python scripts/visualise_model.py --model-name small_main_lr_b
python scripts/compare_results.py --metric f1 --split test
```

This is intentionally hands-on. After visualising the small results, you can decide which
variant to use for the medium subset. After visualising medium results, you can decide what
to use for the large subset.

## 1. Generate Subsets

Create a named split with:

```bash
python scripts/generate_subset.py \
  --subset small \
  --random-seed 42 \
  --train-size 2000 \
  --validation-size 500 \
  --test-size 5000
```

The script saves the split indices to:

```text
data/splits/<subset>_split.json
```

Example project sizes:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/generate_subset.py --subset medium --random-seed 42 --train-size 8000 --validation-size 2000 --test-size 5000
python scripts/generate_subset.py --subset large --random-seed 42 --train-size 20000 --validation-size 5000 --test-size all
```

Use `--test-size all` for the full official IMDb test split.

## 2. Train Models

Train both the baseline and main model for one subset and one preprocessing variant:

```bash
python scripts/train_model.py --subset small --variant b
```

By default, this trains both:

- `baseline_nb`
- `main_lr`

To train only one model:

```bash
python scripts/train_model.py --subset small --variant b --model main_lr
python scripts/train_model.py --subset small --variant b --model baseline_nb
```

Model run names use this format:

```text
<subset>_<model>_<variant>
```

For example:

```text
small_baseline_nb_b
small_main_lr_b
```

Training saves:

```text
models/<subset>_<model>_<variant>.joblib
results/metrics/<subset>_<model>_<variant>.json
```

Each metric JSON file contains validation and test results: accuracy, precision, recall,
F1, ROC-AUC, a confusion matrix, and a classification report.

## 3. Visualise One Model

After training, visualise one model run:

```bash
python scripts/visualise_model.py --model-name small_main_lr_b
```

This creates validation and test figures for that model:

```text
results/figures/small_main_lr_b_validation_metrics.png
results/figures/small_main_lr_b_validation_confusion_matrix.png
results/figures/small_main_lr_b_test_metrics.png
results/figures/small_main_lr_b_test_confusion_matrix.png
```

To visualise only one split:

```bash
python scripts/visualise_model.py --model-name small_main_lr_b --split validation
python scripts/visualise_model.py --model-name small_main_lr_b --split test
```

## 4. Compare All Saved Results

Compare all saved model-result files by one metric:

```bash
python scripts/compare_results.py --metric f1 --split test
```

Other available metrics are:

```text
accuracy
precision
recall
f1
roc_auc
```

Example:

```bash
python scripts/compare_results.py --metric roc_auc --split validation
```

The comparison plot is saved as:

```text
results/figures/compare_<split>_<metric>.png
```

## Suggested Manual Experiment Path

Start with all three variants on the small subset:

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/train_model.py --subset small --variant a
python scripts/train_model.py --subset small --variant b
python scripts/train_model.py --subset small --variant c
python scripts/compare_results.py --metric f1 --split validation
```

Then inspect the models you care about:

```bash
python scripts/visualise_model.py --model-name small_main_lr_b
python scripts/visualise_model.py --model-name small_main_lr_c
```

If A looks weak, you can ignore it and continue with B and C on medium:

```bash
python scripts/generate_subset.py --subset medium --random-seed 42 --train-size 8000 --validation-size 2000 --test-size 5000
python scripts/train_model.py --subset medium --variant b
python scripts/train_model.py --subset medium --variant c
python scripts/compare_results.py --metric f1 --split validation
```

After choosing the final variant, train the large model:

```bash
python scripts/generate_subset.py --subset large --random-seed 42 --train-size 20000 --validation-size 5000 --test-size all
python scripts/train_model.py --subset large --variant b
python scripts/visualise_model.py --model-name large_main_lr_b
python scripts/compare_results.py --metric f1 --split test
```

The large test result is the main final result to report, because it uses the full official
IMDb test split.

## Demo Prediction

After training a model, use it for one review:

```bash
python demo/predict_review.py \
  --model large_main_lr_b \
  --text "A slow start, but a powerful ending."
```

Use the model name that matches the model you selected. The script looks inside
`models/`, so you do not need to type the full path.

## Files

- `src/`: reusable dataset, preprocessing, model, evaluation, and prediction code
- `scripts/generate_subset.py`: creates named train/validation/test splits
- `scripts/train_model.py`: trains the baseline and/or main model for one subset and
  variant
- `scripts/visualise_model.py`: creates metrics and confusion-matrix figures for one
  saved model run
- `scripts/compare_results.py`: compares all saved model-result files by one metric
- `demo/`: command-line prediction demo
- `data/splits/`: saved split indices
- `results/metrics/`: one JSON file per trained model run
- `results/figures/`: generated plots
- `models/`: saved model pipelines
- `MODEL_CARD.md`: short notes about data, model, results, and limits

Generated datasets, saved models, metrics, and figures are ignored by git. Split index
files are small and reproducible, so they can be kept if needed.
