#!/usr/bin/env python3

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from corpus_tagger import SimpleNovelTagger
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
from batch_tag_novels import tag_specific_novels


class TestTaggingIntegration(unittest.TestCase):
    """Integration tests for the complete tagging system"""

    def setUp(self):
        self.sample_novel_text = """
        Alice was a brave and mysterious hero who loved dangerous adventures.
        She spoke with intelligence to the frightened rabbit in the dark forest.
        "We must rescue them!" Alice said with determination and compassion.
        The villainous queen ruled with cruel authority over the magical kingdom.
        Their friendship would be tested in this terrifying supernatural place.
        The foreshadowing in this third-person narrative was unmistakable.
        The family bonds between the characters grew stronger through love.
        """

    def test_end_to_end_tagging_and_memory_integration(self):
        """Test complete workflow from tagging to memory enhancement"""
        with tempfile.TemporaryDirectory() as temp_dir:
            novels_dir = Path(temp_dir) / "novels"
            novels_dir.mkdir()

            novel_dir = novels_dir / "test_adventure"
            novel_dir.mkdir()

            novel_file = novel_dir / "test_adventure.txt"
            novel_file.write_text(self.sample_novel_text, encoding='utf-8')

            with patch('batch_tag_novels.Path') as mock_path_class:
                def path_side_effect(path_str):
                    if path_str == "novels":
                        return novels_dir
                    else:
                        return novels_dir / path_str

                mock_path_class.side_effect = path_side_effect

                with patch('builtins.print'):
                    tag_specific_novels(['test_adventure'])

                tags_file = novel_dir / "tags.json"
                self.assertTrue(tags_file.exists())

                with open(tags_file, 'r', encoding='utf-8') as f:
                    tag_data = json.load(f)

                required_fields = [
                    'themes', 'scene_types', 'behaviors',
                    'narrative_elements', 'social_dynamics'
                ]
                for field in required_fields:
                    self.assertIn(field, tag_data)
                    self.assertIsInstance(tag_data[field], list)

                config = MemoryConfig(memory_chunk_size=20, max_memories_per_novel=10)
                memory_system = EpisodicMemorySystem(config)

                with patch('episodic_memory_system.Path') as mock_mem_path:
                    mock_mem_path.return_value = novel_file.parent / "novels"
                    result = memory_system.build_memory_for_model("test_model", novel_file)

                self.assertIn('memories', result)
                self.assertIn('analysis', result)
                self.assertGreater(len(result['memories']), 0)

                memories = result['memories']
                enhanced_keywords = []
                for memory in memories:
                    enhanced_keywords.extend(memory.keywords)

                expected_enhancements = ['Alice', 'adventure', 'mysterious', 'heroic']
                found_enhancements = [kw for kw in expected_enhancements
                                     if any(expected in enhanced_keywords for expected in [kw])]

                self.assertGreater(len(found_enhancements), 0,
                                  "Memory should be enhanced with tag data")

    def test_tag_consistency_across_components(self):
        """Test that tag categories are consistent across all components"""
        tagger = SimpleNovelTagger()

        enhanced_themes = ['scifi', 'spy', 'thriller', 'action', 'space']
        for theme in enhanced_themes:
            self.assertIn(theme, tagger.theme_keywords,
                         f"Theme {theme} missing from tagger keywords")

        config = MemoryConfig()
        memory_system = EpisodicMemorySystem(config)

        test_tags = {
            'themes': enhanced_themes,
            'characters': ['TestChar'],
            'locations': ['TestLoc']
        }

        test_memory = type('Memory', (), {
            'content': 'Test space agent content',
            'keywords': [],
            'memory_type': 'test',
            'importance_score': 0.5
        })()

        enhanced_memories = memory_system.enhance_memories_with_tags([test_memory], test_tags)
        self.assertEqual(len(enhanced_memories), 1)

    def test_novel_tags_dataclass_completeness(self):
        """Test that NovelTags dataclass includes all enhanced fields"""
        from corpus_tagger import NovelTags

        required_fields = [
            'title', 'basic_stats', 'themes', 'genre_hints',
            'characters', 'locations', 'sentiment_profile',
            'writing_style', 'memory_compatibility',
            'scene_types', 'behaviors', 'narrative_elements', 'social_dynamics'
        ]

        tags = NovelTags(
            title="Test",
            basic_stats={},
            themes=[],
            genre_hints=[],
            characters=[],
            locations=[],
            sentiment_profile={},
            writing_style={},
            memory_compatibility={},
            scene_types=[],
            behaviors=[],
            narrative_elements=[],
            social_dynamics=[]
        )

        for field in required_fields:
            self.assertTrue(hasattr(tags, field), f"NovelTags missing field: {field}")

    def test_tag_detection_thresholds(self):
        """Test that tag detection uses appropriate thresholds"""
        tagger = SimpleNovelTagger()

        minimal_text = "The spy was brave."
        extensive_text = minimal_text * 100

        minimal_themes = tagger.detect_themes(minimal_text)
        extensive_themes = tagger.detect_themes(extensive_text)

        self.assertIsInstance(minimal_themes, list)
        self.assertIsInstance(extensive_themes, list)

        minimal_behaviors = tagger.detect_character_behaviors(minimal_text)
        extensive_behaviors = tagger.detect_character_behaviors(extensive_text)

        self.assertIsInstance(minimal_behaviors, list)
        self.assertIsInstance(extensive_behaviors, list)

    def test_memory_tag_enhancement_performance(self):
        """Test that memory enhancement doesn't degrade performance significantly"""
        config = MemoryConfig()
        memory_system = EpisodicMemorySystem(config)

        test_memories = []
        for i in range(50):
            memory = type('Memory', (), {
                'content': f'Test memory content {i} with adventure and mystery',
                'keywords': [f'keyword{i}'],
                'memory_type': 'test',
                'importance_score': 0.5
            })()
            test_memories.append(memory)

        large_tag_data = {
            'themes': ['adventure', 'mystery', 'horror', 'scifi', 'spy'],
            'characters': [f'Char{i}' for i in range(20)],
            'locations': [f'Loc{i}' for i in range(15)],
            'scene_types': ['action_scene', 'dialogue_scene'],
            'behaviors': ['heroic', 'mysterious'],
            'narrative_elements': ['third_person'],
            'social_dynamics': ['friendship']
        }

        import time
        start_time = time.time()
        enhanced_memories = memory_system.enhance_memories_with_tags(test_memories, large_tag_data)
        end_time = time.time()

        self.assertEqual(len(enhanced_memories), 50)
        self.assertLess(end_time - start_time, 5.0, "Tag enhancement should complete within 5 seconds")


if __name__ == '__main__':
    unittest.main()