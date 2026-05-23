# Model Card: IMDb Sentiment Classifier

## Model Details

- Task: binary sentiment classification for English movie reviews
- Dataset: Stanford IMDb dataset from Hugging Face, `stanfordnlp/imdb`
- Labels: `negative` and `positive`
- Baseline model: `naive_bayes_baseline`
- Main model: `tfidf_logreg_main`
- Selected run: large subset, preprocessing variant `b`
- Selected model file: `models/large_tfidf_logreg_main_b.joblib`

The selected model is a scikit-learn pipeline:

```text
variant b preprocessor -> TF-IDF vectorizer -> Logistic Regression classifier
```

## Intended Use

This repository is for the IU course DLBAIPNLP01, NLP Project "Sentiment Analysis on
Movie Reviews". The selected model classifies IMDb-style movie reviews as positive or
negative. It is intended for coursework, experimentation, and demonstration, not for
important decisions or general sentiment analysis across every domain.

## Data

The project uses the labelled IMDb train and test splits. Validation data is selected
from the official training split, while test data comes from the official test split.

The selected large run uses:

- train: 20,000 reviews
- validation: 5,000 reviews
- test: 25,000 reviews
- random seed: `42`

The raw dataset is downloaded locally by Hugging Face `datasets` and is not committed to
the repository. The files in `data/splits/` store only the selected indices, so the same
small, medium, and large experiments can be reproduced after the dataset is available.

## Preprocessing

The selected run uses preprocessing variant `b`:

- HTML tags and entities are cleaned
- text is lowercased
- negation contractions are expanded, for example `didn't` becomes `did not`
- punctuation and non-letter characters are removed
- stop words are removed, but `not`, `no`, `nor`, and `never` are kept

The other variants were:

- `a`: basic cleaning without stop-word removal
- `c`: variant `b` plus WordNet lemmatisation

Variant `b` gave the best balance in the saved experiments because it removed many high
frequency words that add little sentiment signal, while still keeping negation words that
often change the meaning of a review. Variant `a` kept more noisy common words. Variant
`c` added lemmatisation, but that did not improve the TF-IDF Logistic Regression results
in these runs, likely because useful surface forms and phrase-level cues were already
captured well by the unigram/bigram TF-IDF features.

## Results

The selected model is `large_tfidf_logreg_main_b`. It achieved the strongest overall
held-out test result among the saved runs, including the small and medium tiers and the
available preprocessing variants `a`, `b`, and `c`.

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| `large_naive_bayes_baseline_b` | Validation | 0.8570 | 0.8745 | 0.8336 | 0.8536 | 0.9295 |
| `large_naive_bayes_baseline_b` | Test | 0.8354 | 0.8722 | 0.7861 | 0.8269 | 0.9116 |
| `large_tfidf_logreg_main_b` | Validation | 0.9038 | 0.8923 | 0.9184 | 0.9052 | 0.9637 |
| `large_tfidf_logreg_main_b` | Test | 0.8924 | 0.8903 | 0.8951 | 0.8927 | 0.9586 |

The large TF-IDF Logistic Regression model improved on the smaller saved TF-IDF runs:

| Run | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| `small_tfidf_logreg_main_a` | 0.8404 | 0.8405 | 0.9184 |
| `small_tfidf_logreg_main_b` | 0.8514 | 0.8526 | 0.9259 |
| `small_tfidf_logreg_main_c` | 0.8472 | 0.8483 | 0.9248 |
| `medium_tfidf_logreg_main_b` | 0.8774 | 0.8787 | 0.9471 |
| `medium_tfidf_logreg_main_c` | 0.8792 | 0.8808 | 0.9463 |
| `large_tfidf_logreg_main_b` | 0.8924 | 0.8927 | 0.9586 |

The larger tier helped because it trained on more examples and evaluated on the full
25,000-review IMDb test split. This gave the vectorizer and classifier broader coverage of
review vocabulary, phrasing, and sentiment patterns than the small and medium tiers.

## Reproducing the Selected Run

```bash
python scripts/generate_subset.py --subset large --random-seed 42 --train-size 20000 --validation-size 5000 --test-size 25000
python scripts/train_model.py --subset large --variant b --model naive_bayes_baseline
python scripts/train_model.py --subset large --variant b --model tfidf_logreg_main
python scripts/visualise_model.py --model-name large_tfidf_logreg_main_b --split all
python scripts/compare_results.py --metric f1 --split test
```

The main output files are:

```text
models/large_tfidf_logreg_main_b.joblib
results/metrics/large_tfidf_logreg_main_b.json
results/figures/large_tfidf_logreg_main_b_test_metrics.png
```

## Limitations

- The model is trained on movie reviews, so it may not work well for short social media
  text, product reviews, news, or formal writing.
- The labels are only positive and negative. Neutral or mixed reviews must still be forced
  into one class.
- TF-IDF and Logistic Regression do not truly understand long context, sarcasm, or complex
  opinion changes inside a review.
- The reported winner is based on the saved experiment matrix in this repository. More
  preprocessing choices, model types, or hyperparameter tuning could change the best run.
