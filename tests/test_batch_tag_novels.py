#!/usr/bin/env python3

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from batch_tag_novels import batch_tag_novels, tag_specific_novels
from corpus_tagger import SimpleNovelTagger


class TestBatchTagNovels(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.novels_dir = self.temp_dir / "novels"
        self.novels_dir.mkdir()

        self.sample_text = """
        This is a sample novel with adventure and mystery themes.
        The brave hero spoke with intelligence about the dangerous quest.
        In the dark forest, mysterious creatures lurked with evil intent.
        """

        self.novel_dirs = ['alice_wonderland', 'call_cthulhu', 'frankenstein']
        for novel_name in self.novel_dirs:
            novel_dir = self.novels_dir / novel_name
            novel_dir.mkdir()

            text_file = novel_dir / f"{novel_name}.txt"
            text_file.write_text(self.sample_text, encoding='utf-8')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('batch_tag_novels.Path')
    def test_batch_tag_novels_basic(self, mock_path):
        mock_path.return_value = self.novels_dir

        with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
            mock_tagger = MagicMock()
            mock_tagger_class.return_value = mock_tagger

            mock_tags = MagicMock()
            mock_tags.title = "Test Novel"
            mock_tags.themes = ['adventure', 'mystery']
            mock_tags.characters = ['Hero']
            mock_tags.locations = ['Forest']
            mock_tags.scene_types = ['action_scene']
            mock_tags.behaviors = ['heroic']
            mock_tags.narrative_elements = ['third_person']
            mock_tags.social_dynamics = ['friendship']
            mock_tags.basic_stats = {'word_count': 1000}
            mock_tags.genre_hints = ['Adventure']
            mock_tags.sentiment_profile = {'positive': 0.6}
            mock_tags.writing_style = {'avg_word_length': 4.5}
            mock_tags.memory_compatibility = {'optimal_chunk_size': 200}

            mock_tagger.tag_novel.return_value = mock_tags

            with patch('builtins.print'):
                batch_tag_novels(max_novels=2)

            self.assertGreaterEqual(mock_tagger.tag_novel.call_count, 1)

    def test_tag_specific_novels_real_directory(self):
        with patch('batch_tag_novels.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == "novels":
                    return self.novels_dir
                else:
                    return self.novels_dir / path_str

            mock_path_class.side_effect = path_side_effect

            with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
                mock_tagger = MagicMock()
                mock_tagger_class.return_value = mock_tagger

                mock_tags = MagicMock()
                mock_tags.title = "Alice Wonderland"
                mock_tags.themes = ['adventure', 'fantasy']
                mock_tags.characters = ['Alice']
                mock_tags.locations = ['Wonderland']
                mock_tags.scene_types = ['dialogue_scene']
                mock_tags.behaviors = ['heroic']
                mock_tags.narrative_elements = ['third_person']
                mock_tags.social_dynamics = ['friendship']
                mock_tags.basic_stats = {'word_count': 1000}
                mock_tags.genre_hints = ['Fantasy']
                mock_tags.sentiment_profile = {'positive': 0.7}
                mock_tags.writing_style = {'avg_word_length': 4.2}
                mock_tags.memory_compatibility = {'optimal_chunk_size': 250}

                mock_tagger.tag_novel.return_value = mock_tags

                with patch('builtins.print'):
                    tag_specific_novels(['alice_wonderland'])

                mock_tagger.tag_novel.assert_called_once()

    def test_batch_tag_novels_skip_existing(self):
        existing_novel = self.novels_dir / self.novel_dirs[0]
        tags_file = existing_novel / "tags.json"
        tags_file.write_text('{"title": "Existing"}', encoding='utf-8')

        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
                mock_tagger = MagicMock()
                mock_tagger_class.return_value = mock_tagger

                with patch('builtins.print') as mock_print:
                    batch_tag_novels(max_novels=1)

                    printed_messages = [call[0][0] for call in mock_print.call_args_list]
                    skip_messages = [msg for msg in printed_messages if "[SKIP]" in msg]
                    self.assertGreater(len(skip_messages), 0)

    def test_batch_tag_novels_no_txt_file(self):
        empty_novel_dir = self.novels_dir / "empty_novel"
        empty_novel_dir.mkdir()

        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('builtins.print') as mock_print:
                batch_tag_novels(max_novels=10)

                printed_messages = [call[0][0] for call in mock_print.call_args_list]
                error_messages = [msg for msg in printed_messages if "[ERROR]" in msg]
                self.assertGreater(len(error_messages), 0)

    def test_batch_tag_novels_error_handling(self):
        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
                mock_tagger = MagicMock()
                mock_tagger_class.return_value = mock_tagger
                mock_tagger.tag_novel.side_effect = Exception("Tagging error")

                with patch('builtins.print') as mock_print:
                    batch_tag_novels(max_novels=1)

                    printed_messages = [call[0][0] for call in mock_print.call_args_list]
                    error_messages = [msg for msg in printed_messages if "[ERROR]" in msg]
                    self.assertGreater(len(error_messages), 0)

    def test_tag_specific_novels_missing_directory(self):
        with patch('batch_tag_novels.Path') as mock_path_class:
            def path_side_effect(path_str):
                if path_str == "novels":
                    return self.novels_dir
                else:
                    mock_path = MagicMock()
                    mock_path.exists.return_value = False
                    return mock_path

            mock_path_class.side_effect = path_side_effect

            with patch('builtins.print') as mock_print:
                tag_specific_novels(['nonexistent_novel'])

                printed_messages = [call[0][0] for call in mock_print.call_args_list]
                not_found_messages = [msg for msg in printed_messages if "not found" in msg]
                self.assertGreater(len(not_found_messages), 0)

    def test_batch_tag_novels_progress_tracking(self):
        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
                mock_tagger = MagicMock()
                mock_tagger_class.return_value = mock_tagger

                mock_tags = MagicMock()
                mock_tags.title = "Test"
                mock_tags.themes = ['adventure']
                mock_tags.characters = ['Hero']
                mock_tags.locations = ['Place']
                mock_tags.scene_types = ['action_scene']
                mock_tags.behaviors = ['heroic']
                mock_tags.narrative_elements = ['third_person']
                mock_tags.social_dynamics = ['friendship']
                mock_tags.basic_stats = {'word_count': 1000}
                mock_tags.genre_hints = ['Adventure']
                mock_tags.sentiment_profile = {'positive': 0.6}
                mock_tags.writing_style = {'avg_word_length': 4.5}
                mock_tags.memory_compatibility = {'optimal_chunk_size': 200}

                mock_tagger.tag_novel.return_value = mock_tags

                with patch('builtins.print') as mock_print:
                    batch_tag_novels(max_novels=2)

                    printed_messages = [call[0][0] for call in mock_print.call_args_list]

                    progress_messages = [msg for msg in printed_messages if "Processing" in msg]
                    self.assertGreater(len(progress_messages), 0)

                    completion_messages = [msg for msg in printed_messages if "COMPLETE" in msg]
                    self.assertGreater(len(completion_messages), 0)

    def test_tag_data_structure_completeness(self):
        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('batch_tag_novels.SimpleNovelTagger') as mock_tagger_class:
                mock_tagger = MagicMock()
                mock_tagger_class.return_value = mock_tagger

                mock_tags = MagicMock()
                mock_tags.title = "Complete Novel"
                mock_tags.themes = ['adventure', 'mystery']
                mock_tags.characters = ['Hero', 'Villain']
                mock_tags.locations = ['Castle', 'Forest']
                mock_tags.scene_types = ['action_scene', 'dialogue_scene']
                mock_tags.behaviors = ['heroic', 'villainous']
                mock_tags.narrative_elements = ['third_person', 'foreshadowing']
                mock_tags.social_dynamics = ['friendship', 'rivalry']
                mock_tags.basic_stats = {'word_count': 5000, 'sentence_count': 200}
                mock_tags.genre_hints = ['Adventure', 'Mystery/Crime']
                mock_tags.sentiment_profile = {'positive': 0.6, 'negative': 0.4}
                mock_tags.writing_style = {'avg_word_length': 4.5, 'complex_word_ratio': 0.2}
                mock_tags.memory_compatibility = {'optimal_chunk_size': 200, 'suggested_memory_types': {}}

                mock_tagger.tag_novel.return_value = mock_tags

                saved_data = {}
                original_open = open

                def mock_open_wrapper(*args, **kwargs):
                    if 'tags.json' in str(args[0]) and 'w' in args[1]:
                        file_mock = MagicMock()

                        def mock_write(data):
                            saved_data['content'] = data

                        file_mock.__enter__.return_value.write = mock_write
                        return file_mock
                    else:
                        return original_open(*args, **kwargs)

                with patch('builtins.open', side_effect=mock_open_wrapper):
                    with patch('json.dump') as mock_json_dump:
                        def capture_json(data, file, **kwargs):
                            saved_data['json'] = data

                        mock_json_dump.side_effect = capture_json

                        with patch('builtins.print'):
                            batch_tag_novels(max_novels=1)

                        if 'json' in saved_data:
                            tag_data = saved_data['json']

                            required_fields = [
                                'title', 'basic_stats', 'themes', 'genre_hints',
                                'characters', 'locations', 'sentiment_profile',
                                'writing_style', 'memory_compatibility',
                                'scene_types', 'behaviors', 'narrative_elements',
                                'social_dynamics', 'tagged_date', 'source_file'
                            ]

                            for field in required_fields:
                                self.assertIn(field, tag_data, f"Missing required field: {field}")

    def test_batch_tag_novels_start_from_parameter(self):
        with patch('batch_tag_novels.Path') as mock_path:
            mock_path.return_value = self.novels_dir

            with patch('batch_tag_novels.SimpleNovelTagger'):
                with patch('builtins.print') as mock_print:
                    batch_tag_novels(max_novels=1, start_from=1)

                    printed_messages = [call[0][0] for call in mock_print.call_args_list]
                    start_messages = [msg for msg in printed_messages if "starting from index 1" in msg]
                    self.assertGreater(len(start_messages), 0)


class TestBatchTagNovelsIntegration(unittest.TestCase):

    def test_enhanced_tag_fields_integration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            novels_dir = Path(temp_dir) / "novels"
            novels_dir.mkdir()

            novel_dir = novels_dir / "test_novel"
            novel_dir.mkdir()

            novel_file = novel_dir / "test_novel.txt"
            novel_file.write_text("""
            The brave hero fought the villainous enemy in an action scene.
            They spoke with intelligence about their mysterious quest.
            The foreshadowing in this third person narrative was clear.
            Family bonds and friendship guided their journey.
            """, encoding='utf-8')

            with patch('batch_tag_novels.Path') as mock_path_class:
                def path_side_effect(path_str):
                    if path_str == "novels":
                        return novels_dir
                    else:
                        return novels_dir / path_str

                mock_path_class.side_effect = path_side_effect

                with patch('builtins.print'):
                    tag_specific_novels(['test_novel'])

                tags_file = novel_dir / "tags.json"
                self.assertTrue(tags_file.exists())

                with open(tags_file, 'r', encoding='utf-8') as f:
                    tag_data = json.load(f)

                enhanced_fields = ['scene_types', 'behaviors', 'narrative_elements', 'social_dynamics']
                for field in enhanced_fields:
                    self.assertIn(field, tag_data, f"Enhanced field {field} missing from saved data")
                    self.assertIsInstance(tag_data[field], list, f"Field {field} should be a list")


if __name__ == '__main__':
    unittest.main()