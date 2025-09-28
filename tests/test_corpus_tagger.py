#!/usr/bin/env python3

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from corpus_tagger import SimpleNovelTagger, NovelTags, find_novels_by_criteria


class TestSimpleNovelTagger(unittest.TestCase):

    def setUp(self):
        self.tagger = SimpleNovelTagger()
        self.sample_text = """
        Alice was beginning to get very tired of sitting by her sister on the bank,
        and of having nothing to do. The young girl felt a deep love for adventure
        and mystery. Suddenly, she saw a white rabbit running past, which seemed
        very strange indeed. The rabbit spoke with fear in his voice, "I'm late!"

        This magical moment would change everything. Alice followed the mysterious
        creature down a dark hole, not knowing what dangers awaited her below.
        Her brave heart pushed her forward despite the terror she felt.

        In the strange underground world, she met many characters who seemed both
        intelligent and deceptive. The Queen of Hearts ruled with cruel authority,
        while the Cheshire Cat offered wise guidance with his mysterious smile.

        Through her journey, Alice discovered the power of friendship and the
        importance of family bonds. She learned that even in the darkest places,
        love and compassion could triumph over evil and despair.
        """

    def test_extract_basic_stats(self):
        stats = self.tagger.extract_basic_stats(self.sample_text)

        self.assertIn('word_count', stats)
        self.assertIn('sentence_count', stats)
        self.assertIn('paragraph_count', stats)
        self.assertIn('avg_sentence_length', stats)
        self.assertIn('dialogue_ratio', stats)
        self.assertIn('unique_words', stats)

        self.assertGreater(stats['word_count'], 0)
        self.assertGreater(stats['sentence_count'], 0)
        self.assertGreater(stats['unique_words'], 0)

    def test_detect_themes(self):
        themes = self.tagger.detect_themes(self.sample_text)

        self.assertIsInstance(themes, list)
        self.assertIn('adventure', themes)
        self.assertIn('mystery', themes)
        self.assertIn('romance', themes)

    def test_detect_scene_types(self):
        scene_types = self.tagger.detect_scene_types(self.sample_text)

        self.assertIsInstance(scene_types, list)
        self.assertIn('dialogue_scene', scene_types)
        self.assertIn('discovery_scene', scene_types)
        self.assertIn('emotional_scene', scene_types)

    def test_detect_character_behaviors(self):
        behaviors = self.tagger.detect_character_behaviors(self.sample_text)

        self.assertIsInstance(behaviors, list)
        self.assertIn('heroic', behaviors)
        self.assertIn('mysterious', behaviors)
        self.assertIn('intelligent', behaviors)

    def test_detect_narrative_elements(self):
        elements = self.tagger.detect_narrative_elements(self.sample_text)

        self.assertIsInstance(elements, list)
        self.assertIn('third_person', elements)
        self.assertIn('suspense', elements)

    def test_detect_social_dynamics(self):
        dynamics = self.tagger.detect_social_dynamics(self.sample_text)

        self.assertIsInstance(dynamics, list)
        self.assertIn('family', dynamics)
        self.assertIn('friendship', dynamics)
        self.assertIn('authority', dynamics)

    def test_find_characters(self):
        characters = self.tagger.find_characters(self.sample_text)

        self.assertIsInstance(characters, list)

    def test_find_locations(self):
        locations = self.tagger.find_locations(self.sample_text)

        self.assertIsInstance(locations, list)

    def test_analyze_sentiment_profile(self):
        sentiment = self.tagger.analyze_sentiment_profile(self.sample_text)

        self.assertIn('positive', sentiment)
        self.assertIn('negative', sentiment)
        self.assertIn('emotional_intensity', sentiment)

        self.assertIsInstance(sentiment['positive'], float)
        self.assertIsInstance(sentiment['negative'], float)
        self.assertGreaterEqual(sentiment['positive'], 0)
        self.assertGreaterEqual(sentiment['negative'], 0)

    def test_analyze_writing_style(self):
        style = self.tagger.analyze_writing_style(self.sample_text)

        self.assertIn('avg_word_length', style)
        self.assertIn('complex_word_ratio', style)
        self.assertIn('short_sentence_ratio', style)
        self.assertIn('exclamation_ratio', style)

        self.assertIsInstance(style['avg_word_length'], float)
        self.assertGreater(style['avg_word_length'], 0)

    def test_create_memory_compatibility_data(self):
        characters = ['Alice', 'Rabbit']
        locations = ['Wonderland', 'Bank']
        themes = ['adventure', 'mystery']

        memory_data = self.tagger.create_memory_compatibility_data(
            self.sample_text, characters, locations, themes
        )

        self.assertIn('memory_density_estimate', memory_data)
        self.assertIn('character_memory_potential', memory_data)
        self.assertIn('location_memory_potential', memory_data)
        self.assertIn('theme_memory_potential', memory_data)
        self.assertIn('suggested_memory_types', memory_data)

        self.assertEqual(memory_data['character_memory_potential'], 2)
        self.assertEqual(memory_data['location_memory_potential'], 2)
        self.assertEqual(memory_data['theme_memory_potential'], 2)

    def test_tag_novel_integration(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.sample_text)
            temp_path = Path(f.name)

        try:
            tags = self.tagger.tag_novel(temp_path)

            self.assertIsInstance(tags, NovelTags)
            self.assertIsInstance(tags.title, str)
            self.assertIsInstance(tags.themes, list)
            self.assertIsInstance(tags.scene_types, list)
            self.assertIsInstance(tags.behaviors, list)
            self.assertIsInstance(tags.narrative_elements, list)
            self.assertIsInstance(tags.social_dynamics, list)

            self.assertGreater(len(tags.themes), 0)
            self.assertGreater(len(tags.scene_types), 0)

        finally:
            temp_path.unlink()

    def test_classify_genre(self):
        themes = ['horror', 'mystery', 'supernatural']
        genres = self.tagger.classify_genre(self.sample_text, themes)

        self.assertIsInstance(genres, list)
        self.assertIn('Gothic/Horror', genres)
        self.assertIn('Mystery/Crime', genres)

    def test_theme_keyword_coverage(self):
        expected_themes = [
            'romance', 'horror', 'mystery', 'adventure', 'fantasy',
            'science_fiction', 'historical', 'war', 'supernatural', 'crime',
            'scifi', 'spy', 'thriller', 'action', 'space'
        ]

        for theme in expected_themes:
            self.assertIn(theme, self.tagger.theme_keywords)
            self.assertIsInstance(self.tagger.theme_keywords[theme], list)
            self.assertGreater(len(self.tagger.theme_keywords[theme]), 0)


class TestNovelTags(unittest.TestCase):

    def test_novel_tags_dataclass(self):
        tags = NovelTags(
            title="Test Novel",
            basic_stats={'word_count': 1000},
            themes=['adventure'],
            genre_hints=['Adventure'],
            characters=['Hero'],
            locations=['Castle'],
            sentiment_profile={'positive': 0.6, 'negative': 0.4},
            writing_style={'avg_word_length': 4.5},
            memory_compatibility={'optimal_chunk_size': 200},
            scene_types=['action_scene'],
            behaviors=['heroic'],
            narrative_elements=['third_person'],
            social_dynamics=['friendship']
        )

        self.assertEqual(tags.title, "Test Novel")
        self.assertEqual(tags.themes, ['adventure'])
        self.assertEqual(tags.scene_types, ['action_scene'])
        self.assertEqual(tags.behaviors, ['heroic'])


class TestFindNovelsByCriteria(unittest.TestCase):

    def setUp(self):
        self.sample_corpus = {
            'novels': {
                'alice_in_wonderland': {
                    'themes': ['adventure', 'fantasy'],
                    'genre_hints': ['Fantasy', 'Adventure'],
                    'basic_stats': {'word_count': 15000},
                    'characters': ['Alice', 'Rabbit']
                },
                'call_of_cthulhu': {
                    'themes': ['horror', 'supernatural'],
                    'genre_hints': ['Gothic/Horror'],
                    'basic_stats': {'word_count': 12000},
                    'characters': ['Narrator']
                },
                'spy_novel': {
                    'themes': ['spy', 'thriller', 'action'],
                    'genre_hints': ['Spy/Espionage', 'Thriller/Suspense'],
                    'basic_stats': {'word_count': 25000},
                    'characters': ['Agent']
                }
            }
        }

    @patch('builtins.open')
    @patch('json.load')
    def test_find_novels_by_theme(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({'theme': 'horror'})
        self.assertEqual(results, ['call_of_cthulhu'])

        results = find_novels_by_criteria({'theme': 'adventure'})
        self.assertEqual(results, ['alice_in_wonderland'])

    @patch('builtins.open')
    @patch('json.load')
    def test_find_novels_by_genre(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({'genre': 'Fantasy'})
        self.assertEqual(results, ['alice_in_wonderland'])

        results = find_novels_by_criteria({'genre': 'Spy/Espionage'})
        self.assertEqual(results, ['spy_novel'])

    @patch('builtins.open')
    @patch('json.load')
    def test_find_novels_by_word_count(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({'min_words': 20000})
        self.assertEqual(results, ['spy_novel'])

        results = find_novels_by_criteria({'max_words': 15000})
        self.assertIn('alice_in_wonderland', results)
        self.assertIn('call_of_cthulhu', results)

    @patch('builtins.open')
    @patch('json.load')
    def test_find_novels_by_character(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({'character': 'Alice'})
        self.assertEqual(results, ['alice_in_wonderland'])

        results = find_novels_by_criteria({'character': 'agent'})
        self.assertEqual(results, ['spy_novel'])

    @patch('builtins.open')
    @patch('json.load')
    def test_find_novels_multiple_criteria(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({
            'theme': 'spy',
            'min_words': 20000
        })
        self.assertEqual(results, ['spy_novel'])

        results = find_novels_by_criteria({
            'theme': 'adventure',
            'max_words': 20000
        })
        self.assertEqual(results, ['alice_in_wonderland'])

    @patch('builtins.open')
    @patch('json.load')
    def test_no_matches(self, mock_json_load, mock_open):
        mock_json_load.return_value = self.sample_corpus

        results = find_novels_by_criteria({'theme': 'nonexistent'})
        self.assertEqual(results, [])

        results = find_novels_by_criteria({'min_words': 100000})
        self.assertEqual(results, [])

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_missing_corpus_file(self, mock_open):
        results = find_novels_by_criteria({'theme': 'horror'})
        self.assertEqual(results, [])


class TestTaggingSystemIntegration(unittest.TestCase):

    def test_enhanced_themes_coverage(self):
        tagger = SimpleNovelTagger()

        enhanced_themes = ['scifi', 'spy', 'thriller', 'action', 'space']
        for theme in enhanced_themes:
            self.assertIn(theme, tagger.theme_keywords)

        scifi_keywords = tagger.theme_keywords['scifi']
        self.assertIn('spacecraft', scifi_keywords)
        self.assertIn('galaxy', scifi_keywords)
        self.assertIn('laboratory', scifi_keywords)

        spy_keywords = tagger.theme_keywords['spy']
        self.assertIn('agent', spy_keywords)
        self.assertIn('espionage', spy_keywords)
        self.assertIn('classified', spy_keywords)

    def test_scene_behavior_narrative_detection(self):
        tagger = SimpleNovelTagger()

        test_text = """
        The brave hero spoke with intelligence, "We must rescue them!"
        His mysterious companion nodded with a deceptive smile.
        The action scene unfolded as they fought the villainous enemy.
        This foreshadowing moment would prove crucial later.
        The family bond between them grew stronger through friendship.
        """

        scene_types = tagger.detect_scene_types(test_text)
        behaviors = tagger.detect_character_behaviors(test_text)
        narrative_elements = tagger.detect_narrative_elements(test_text)
        social_dynamics = tagger.detect_social_dynamics(test_text)

        self.assertIn('dialogue_scene', scene_types)

        self.assertIn('heroic', behaviors)
        self.assertIn('mysterious', behaviors)
        self.assertIn('villainous', behaviors)

        self.assertIn('third_person', narrative_elements)

        self.assertIn('family', social_dynamics)
        self.assertIn('friendship', social_dynamics)

    def test_tag_normalization_scoring(self):
        tagger = SimpleNovelTagger()

        short_text = "The spy was very intelligent and mysterious."
        long_text = short_text * 100  # Much longer text

        short_themes = tagger.detect_themes(short_text)
        long_themes = tagger.detect_themes(long_text)

        self.assertIsInstance(short_themes, list)
        self.assertIsInstance(long_themes, list)


if __name__ == '__main__':
    unittest.main()