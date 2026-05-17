from src import preprocess


class FakeLemmatizer:
    def lemmatize(self, token: str) -> str:
        forms = {
            "films": "film",
            "stories": "story",
            "boring": "boring",
        }
        return forms.get(token, token)


def test_basic_cleaning_expands_negations_and_removes_html():
    cleaned = preprocess.preprocess_text(
        "I didn't love it.<br />Great &amp; funny!",
        remove_stopwords=False,
        lemmatise=False,
    )

    assert cleaned == "did not love it great funny"


def test_variant_a_keeps_stopwords_and_word_forms():
    cleaned = preprocess.get_preprocessor("a")(
        "This isn't the best film, but the stories are moving."
    )

    assert cleaned == "this is not the best film but the stories are moving"


def test_variant_b_removes_stopwords_but_keeps_negators(monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "get_stop_words",
        lambda: {"this", "is", "the", "but", "are"},
    )

    cleaned = preprocess.get_preprocessor("b")(
        "This isn't the best film, but the stories are moving."
    )

    assert cleaned == "not best film stories moving"


def test_variant_c_adds_lemmatisation(monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "get_stop_words",
        lambda: {"this", "is", "the", "but", "are"},
    )
    monkeypatch.setattr(preprocess, "get_lemmatizer", lambda: FakeLemmatizer())

    cleaned = preprocess.get_preprocessor("c")(
        "This isn't the best films, but the stories are moving."
    )

    assert cleaned == "not best film story moving"


def test_variant_c_aliases_are_not_needed(monkeypatch):
    monkeypatch.setattr(
        preprocess,
        "get_stop_words",
        lambda: {"the", "was"},
    )
    monkeypatch.setattr(preprocess, "get_lemmatizer", lambda: FakeLemmatizer())

    cleaned = preprocess.get_preprocessor("c")("The films was boring.")

    assert cleaned == "film boring"
