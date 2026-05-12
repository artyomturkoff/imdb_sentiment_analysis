# Model Card: IMDb Sentiment Classifier

## Model Details

- Task: binary sentiment classification for English movie reviews
- Dataset: Stanford IMDb dataset from Hugging Face, `stanfordnlp/imdb`
- Labels: `negative` and `positive`
- Baseline model: `naive_bayes_baseline`
- Main model: `tfidf_logreg_main`
- Example run in this card: small subset, preprocessing variant `b`

## Intended Use

This repository is for a IU course DLBAIPNLP01, NLP Project "Sentiment Analysis on Movie Reviews". The model can classify IMDb-style movie
reviews as positive or negative. It is not meant for important decisions or for sentiment
analysis in every domain.

## Data

The project uses the labelled IMDb train and test splits. Validation data is taken from the
official training split, while test data comes from the official test split.

The raw dataset is downloaded locally by Hugging Face `datasets`. It is not committed to
the repository. The files in `data/splits/` store only the selected indices, so other
people can reproduce the same splits.

## Preprocessing

The example uses variant `b`:

- HTML tags and entities are cleaned
- text is lowercased
- negation contractions are expanded, for example `didn't` becomes `did not`
- punctuation and non-letter characters are removed
- stop words are removed, but `not`, `no`, `nor`, and `never` are kept

Variant `c` also adds WordNet lemmatisation.

## Example Results

These results are from a small subset run:

- train: 2,000 reviews
- validation: 500 reviews
- test: 5,000 reviews
- preprocessing variant: `b`
- random seed: `42`

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| `naive_bayes_baseline` | Validation | 0.8240 | 0.8375 | 0.8040 | 0.8204 | 0.8763 |
| `naive_bayes_baseline` | Test | 0.8120 | 0.8482 | 0.7600 | 0.8017 | 0.8963 |
| `tfidf_logreg_main` | Validation | 0.8220 | 0.8038 | 0.8520 | 0.8272 | 0.9082 |
| `tfidf_logreg_main` | Test | 0.8514 | 0.8457 | 0.8596 | 0.8526 | 0.9259 |

The main TF-IDF Logistic Regression model gives the best test F1 score in this small run.

## Reproducing the Example

```bash
python scripts/generate_subset.py --subset small --random-seed 42 --train-size 2000 --validation-size 500 --test-size 5000
python scripts/train_model.py --subset small --variant b
python scripts/visualise_model.py --model-name small_tfidf_logreg_main_b
python scripts/compare_results.py --metric f1 --split test
```

## Limitations

- The model is trained on movie reviews, so it may not work well for short social media
  text, product reviews, or formal writing.
- The labels are only positive and negative. Neutral or mixed reviews must still be forced
  into one class.
- TF-IDF and bag-of-words models do not really understand long context, sarcasm, or complex
  opinion changes inside a review.
- Results from the small subset are useful for checking the workflow, but the large subset
  should be used for a stronger final experiment.
