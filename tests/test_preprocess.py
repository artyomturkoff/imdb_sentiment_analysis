import unittest

from src.preprocess import preprocess_text


class PreprocessTextTests(unittest.TestCase):
    def test_expands_negation_contractions(self):
        cleaned = preprocess_text(
            "I didn't love it.",
            remove_stopwords=False,
            lemmatise=False,
        )
        self.assertIn("did not love", cleaned)

    def test_removes_html_breaks_and_unescapes_entities(self):
        cleaned = preprocess_text(
            "Great &amp; funny<br />Not dull.",
            remove_stopwords=False,
            lemmatise=False,
        )
        self.assertEqual(cleaned, "great funny not dull")


if __name__ == "__main__":
    unittest.main()

