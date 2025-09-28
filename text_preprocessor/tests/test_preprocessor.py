#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path

from ..core.preprocessor import TextPreprocessor
from ..config.settings import PreprocessorConfig


class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.config = PreprocessorConfig(use_huggingface_models=False)
        self.preprocessor = TextPreprocessor(self.config)

    def test_remove_metadata_and_headers(self):
        sample_text = """Title: Test Novel

Author: Test Author

CONTENTS

CHAPTER I -- The Beginning

The actual story begins here with real content.
This is the main narrative text that should be preserved."""

        result = self.preprocessor.remove_metadata_and_headers(sample_text)

        self.assertNotIn("Title:", result)
        self.assertNotIn("Author:", result)
        self.assertNotIn("CONTENTS", result)
        self.assertIn("The actual story begins", result)
        self.assertIn("main narrative text", result)

    def test_convert_scripts_to_prose(self):
        script_text = """ALICE

That's a funny game, uncle. What did you do then?

CARROLL

A red pawn took a white pawn; this way."""

        result = self.preprocessor.convert_scripts_to_prose(script_text)

        self.assertIn('"That\'s a funny game, uncle. What did you do then?," Alice said.', result)
        self.assertIn('"A red pawn took a white pawn; this way," Carroll said.', result)

    def test_normalize_dialogue(self):
        dialogue_text = 'He said"Hello there"and walked away.'

        result = self.preprocessor.normalize_dialogue(dialogue_text)

        self.assertIn('He said "Hello there" and walked away.', result)

    def test_standardize_punctuation(self):
        punct_text = "This has--long dashes and...many dots."

        result = self.preprocessor.standardize_punctuation(punct_text)

        self.assertIn("This has—long dashes", result)
        self.assertIn("and...many dots", result)

    def test_clean_whitespace(self):
        messy_text = "Line one\n\n\n\nLine two\n   \n\nLine three"

        result = self.preprocessor.clean_whitespace(messy_text)

        self.assertEqual(result.count('\n\n'), 2)
        self.assertNotIn('\n\n\n', result)

    def test_full_preprocessing_pipeline(self):
        sample_novel = """Title: Test Story

Author: John Doe

CHAPTER I

THE BEGINNING

ALICE

Hello, uncle!

CARROLL

_smiling_ Hello there, my dear.

The story continues with normal narrative text here."""

        processed_text, stats = self.preprocess_novel_from_text(sample_novel)

        self.assertNotIn("Title:", processed_text)
        self.assertNotIn("CHAPTER I", processed_text)
        self.assertIn('"Hello, uncle!," Alice said.', processed_text)
        self.assertIn('"Hello there, my dear," Carroll said.', processed_text)
        self.assertIn("The story continues", processed_text)

        self.assertIn('original_tokens', stats)
        self.assertIn('processed_tokens', stats)
        self.assertIn('tokens_removed', stats)

    def preprocess_novel_from_text(self, text: str):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = Path(f.name)

        try:
            return self.preprocessor.preprocess_novel(temp_path)
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    unittest.main()