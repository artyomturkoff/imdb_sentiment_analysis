# Model Card: IMDb Sentiment Classifier

## Model Details

- Task: binary sentiment classification for English movie reviews
- Primary model: TF-IDF unigrams/bigrams with Logistic Regression
- Baseline: CountVectorizer with Multinomial Naive Bayes
- Dataset: Stanford Large Movie Review Dataset (`stanfordnlp/imdb`)
- Labels: `negative` and `positive`

## Intended Use

This model is for an undergraduate NLP project. It classifies a movie review as positive
or negative and gives a confidence score.

## Training Data

The project uses the labelled portion of `aclImdb`: the official 25,000-review training
split and the official 25,000-review test split. A stratified 5,000-review validation set is
carved from the official training split for model selection.

## Preprocessing

The final variant should be selected by validation F1. The planned variants are:

- A: HTML/entity cleanup, lowercasing, whitespace normalization, negation expansion
- B: A plus stop-word removal while retaining `no`, `not`, `nor`, and `never`
- C: B plus WordNet lemmatization

## Evaluation

Final numbers are created after running:

```bash
python scripts/train_model.py --subset large --variant <a|b|c>
```

| Model | Variant | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Baseline NB | C | fill from run | fill from run | fill from run | fill from run | fill from run |
| Main LR | C | fill from run | fill from run | fill from run | fill from run | fill from run |

## Limitations

- The classifier is trained on movie reviews and should not be treated as a general
  sentiment system for other domains.
- Bag-of-words and TF-IDF features can miss sarcasm, weak negation, and reviews that
  contain both positive and negative opinions.
- The model predicts only binary sentiment; neutral or nuanced opinions are forced into
  one of two classes.

## Use Notes

This is a learning project, not a model for important decisions. Reported results should
come from the held-out official test set, without tuning after seeing test scores.
