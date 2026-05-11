# Data

Dataset: Stanford Large Movie Review Dataset (`stanfordnlp/imdb`) loaded with Hugging
Face `datasets`.

- Official train split: 25,000 labelled reviews
- Official test split: 25,000 labelled reviews
- Labels: `0 = negative`, `1 = positive`

Runtime caches are stored in `data/raw/` and ignored by git. Deterministic split indices
are stored in `data/splits/` so experiments can be recreated exactly.
