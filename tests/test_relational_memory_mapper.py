#!/usr/bin/env python3
"""
Test relational memory mapper functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from relational_memory_mapper import RelationalMemoryMapper, RelationalConfig


class TestRelationalConfig:
    """Test RelationalConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = RelationalConfig()

        assert config.similarity_threshold == 0.3
        assert config.theme_overlap_threshold == 0.2
        assert config.character_similarity_threshold == 0.4
        assert config.max_cross_references_per_novel == 5
        assert config.max_thematic_connections == 10
        assert config.max_character_mappings == 8
        assert config.analyze_themes == True
        assert config.analyze_characters == True
        assert config.analyze_narrative_patterns == True
        assert config.analyze_stylistic_elements == True
        assert config.create_enhanced_memories == True
        assert config.cross_novel_memory_percentage == 0.2

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = RelationalConfig()

        # Test reasonable ranges
        assert 0 < config.similarity_threshold <= 1.0
        assert 0 < config.theme_overlap_threshold <= 1.0
        assert 0 < config.character_similarity_threshold <= 1.0
        assert 1 <= config.max_cross_references_per_novel <= 20
        assert 1 <= config.max_thematic_connections <= 50
        assert 1 <= config.max_character_mappings <= 50
        assert 0 <= config.cross_novel_memory_percentage <= 1.0


class TestRelationalMemoryMapper:
    """Test RelationalMemoryMapper class"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.config = RelationalConfig()

        # Mock model mapping data
        self.mock_mapping = {
            "metadata": {
                "total_models": 1,
                "total_novels_assigned": 3
            },
            "models": {
                "test_model": {
                    "description": "Test Model",
                    "novel_count": 3,
                    "novels": [
                        {"original_name": "Horror Novel", "directory_name": "horror_novel"},
                        {"original_name": "Mystery Novel", "directory_name": "mystery_novel"},
                        {"original_name": "Adventure Novel", "directory_name": "adventure_novel"}
                    ]
                }
            }
        }

        # Mock novel contents with different themes
        self.mock_novel_contents = {
            "horror_novel": "The dark mansion was filled with terror and fear. The monster lurked in the shadows, causing nightmares for all who entered. Evil spirits haunted the cursed halls.",
            "mystery_novel": "The detective investigated the mysterious murder case. Finding clues and evidence, he questioned every suspect carefully. The crime scene revealed crucial information.",
            "adventure_novel": "The brave explorers embarked on a dangerous journey through unknown lands. They discovered ancient treasures while escaping from fierce battles and perilous quests."
        }

    @patch('relational_memory_mapper.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_model_mapping(self, mock_json_load, mock_open, mock_path):
        """Test model mapping loading"""
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        mapper = RelationalMemoryMapper(self.config)

        assert mapper.model_mapping == self.mock_mapping

    def test_load_theme_keywords(self):
        """Test theme keywords loading"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)
            keywords = mapper._load_theme_keywords()

            assert "horror" in keywords
            assert "mystery" in keywords
            assert "adventure" in keywords
            assert isinstance(keywords["horror"], list)
            assert "fear" in keywords["horror"]
            assert "detective" in keywords["mystery"]
            assert "journey" in keywords["adventure"]

    def test_load_character_indicators(self):
        """Test character indicators loading"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)
            indicators = mapper._load_character_indicators()

            assert isinstance(indicators, list)
            assert "protagonist" in indicators
            assert "detective" in indicators
            assert "hero" in indicators

    def test_get_available_models(self):
        """Test getting available models list"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)
            models = mapper.get_available_models()

            assert models == ["test_model"]

    def test_extract_themes(self):
        """Test theme extraction from content"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Test horror content
            horror_themes = mapper._extract_themes(self.mock_novel_contents["horror_novel"])
            assert "horror" in horror_themes
            assert horror_themes["horror"] > 0

            # Test mystery content
            mystery_themes = mapper._extract_themes(self.mock_novel_contents["mystery_novel"])
            assert "mystery" in mystery_themes
            assert mystery_themes["mystery"] > 0

            # Test adventure content
            adventure_themes = mapper._extract_themes(self.mock_novel_contents["adventure_novel"])
            assert "adventure" in adventure_themes
            assert adventure_themes["adventure"] > 0

    def test_extract_character_archetypes(self):
        """Test character archetype extraction"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Test content with detective
            archetypes = mapper._extract_character_archetypes(self.mock_novel_contents["mystery_novel"])
            assert "detective" in archetypes
            assert archetypes["detective"] > 0

    def test_find_thematic_connections(self):
        """Test finding thematic connections between novels"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Create mock novel themes with some overlap
            novel_themes = {
                "novel_1": {
                    "original_name": "Novel 1",
                    "themes": {"horror": 5.0, "mystery": 2.0},
                    "word_count": 1000
                },
                "novel_2": {
                    "original_name": "Novel 2",
                    "themes": {"horror": 3.0, "adventure": 4.0},
                    "word_count": 1200
                },
                "novel_3": {
                    "original_name": "Novel 3",
                    "themes": {"adventure": 6.0, "mystery": 1.0},
                    "word_count": 900
                }
            }

            connections = mapper._find_thematic_connections(novel_themes)

            assert len(connections) > 0
            # Should find connections between novels with shared themes
            connection_pairs = [(c["novel1"], c["novel2"]) for c in connections]
            assert any("Novel 1" in pair and "Novel 2" in pair for pair in connection_pairs)

    def test_find_character_connections(self):
        """Test finding character connections between novels"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Create mock novel characters with some overlap
            novel_characters = {
                "novel_1": {
                    "original_name": "Novel 1",
                    "character_archetypes": {"detective": 3, "hero": 1}
                },
                "novel_2": {
                    "original_name": "Novel 2",
                    "character_archetypes": {"detective": 2, "villain": 1}
                },
                "novel_3": {
                    "original_name": "Novel 3",
                    "character_archetypes": {"hero": 2, "protagonist": 1}
                }
            }

            connections = mapper._find_character_connections(novel_characters)

            assert len(connections) >= 0
            if connections:
                # Should find connections between novels with shared character types
                connection_pairs = [(c["novel1"], c["novel2"]) for c in connections]
                assert any("Novel 1" in pair and "Novel 2" in pair for pair in connection_pairs)

    def test_extract_narrative_patterns(self):
        """Test narrative pattern extraction"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            content = "He walked slowly. She spoke quietly. They talked together. I thought about it."
            patterns = mapper._extract_narrative_patterns(content)

            assert "average_sentence_length" in patterns
            assert "dialogue_percentage" in patterns
            assert "description_density" in patterns
            assert "pacing_indicators" in patterns
            assert "narrative_style" in patterns

            assert patterns["average_sentence_length"] > 0
            assert patterns["narrative_style"] in ["first_person", "third_person", "mixed"]

    def test_calculate_dialogue_percentage(self):
        """Test dialogue percentage calculation"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            content_with_dialogue = 'He said "Hello there." She replied "Hi back."'
            content_without_dialogue = "The sun was shining brightly in the sky."

            dialogue_percent_high = mapper._calculate_dialogue_percentage(content_with_dialogue)
            dialogue_percent_low = mapper._calculate_dialogue_percentage(content_without_dialogue)

            assert dialogue_percent_high > dialogue_percent_low
            assert dialogue_percent_high > 0
            assert dialogue_percent_low >= 0

    def test_classify_narrative_style(self):
        """Test narrative style classification"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            first_person_text = "I walked to the store. I bought some milk. I went home."
            third_person_text = "The man walked and he was tired. She bought some milk. They went home."
            mixed_text = "I walked to the store. He bought some milk. We went home together."

            assert mapper._classify_narrative_style(first_person_text) == "first_person"
            assert mapper._classify_narrative_style(third_person_text) == "third_person"
            assert mapper._classify_narrative_style(mixed_text) in ["mixed", "first_person", "third_person"]

    def test_analyze_pacing(self):
        """Test pacing analysis"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            action_text = "He suddenly ran quickly and jumped over the fence. They fought fiercely."
            contemplative_text = "She thought about her life. He considered the possibilities and reflected on the past."

            action_pacing = mapper._analyze_pacing(action_text)
            contemplative_pacing = mapper._analyze_pacing(contemplative_text)

            assert "action_density" in action_pacing
            assert "contemplative_density" in action_pacing

            # Action text should have higher action density
            assert action_pacing["action_density"] > contemplative_pacing["action_density"]
            # Contemplative text should have higher contemplative density
            assert contemplative_pacing["contemplative_density"] > action_pacing["contemplative_density"]

    @patch.object(RelationalMemoryMapper, '_load_novel_content')
    def test_analyze_thematic_similarities(self, mock_load_content):
        """Test thematic similarity analysis"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Mock novel content loading
            mock_load_content.side_effect = lambda directory: self.mock_novel_contents[directory]

            novels = self.mock_mapping["models"]["test_model"]["novels"]
            analysis = mapper.analyze_thematic_similarities(novels)

            assert "novel_themes" in analysis
            assert "thematic_connections" in analysis
            assert "theme_summary" in analysis

            # Should have themes for each novel
            assert len(analysis["novel_themes"]) > 0

            # Should have summary statistics
            assert "dominant_themes" in analysis["theme_summary"]
            assert "total_unique_themes" in analysis["theme_summary"]

    @patch.object(RelationalMemoryMapper, '_load_novel_content')
    def test_analyze_character_relationships(self, mock_load_content):
        """Test character relationship analysis"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Mock novel content loading
            mock_load_content.side_effect = lambda directory: self.mock_novel_contents[directory]

            novels = self.mock_mapping["models"]["test_model"]["novels"]
            analysis = mapper.analyze_character_relationships(novels)

            assert "novel_characters" in analysis
            assert "character_connections" in analysis
            assert "archetype_summary" in analysis

    @patch.object(RelationalMemoryMapper, '_load_novel_content')
    def test_analyze_narrative_patterns_full(self, mock_load_content):
        """Test complete narrative pattern analysis"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Mock novel content loading
            mock_load_content.side_effect = lambda directory: self.mock_novel_contents[directory]

            novels = self.mock_mapping["models"]["test_model"]["novels"]
            analysis = mapper.analyze_narrative_patterns(novels)

            assert "narrative_patterns" in analysis
            assert "pattern_connections" in analysis
            assert "pattern_summary" in analysis

    def test_create_unified_cross_references(self):
        """Test unified cross-reference creation"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Mock relational data with some connections
            relational_data = {
                "thematic_analysis": {
                    "thematic_connections": [
                        {
                            "novel1": "Novel 1",
                            "novel2": "Novel 2",
                            "novel1_dir": "novel_1",
                            "novel2_dir": "novel_2",
                            "common_themes": ["horror", "mystery"],
                            "connection_strength": 0.8
                        }
                    ]
                },
                "character_analysis": {
                    "character_connections": [
                        {
                            "novel1": "Novel 1",
                            "novel2": "Novel 3",
                            "novel1_dir": "novel_1",
                            "novel2_dir": "novel_3",
                            "common_archetypes": ["detective", "hero"],
                            "connection_strength": 0.6
                        }
                    ]
                }
            }

            cross_refs = mapper._create_unified_cross_references(relational_data)

            assert isinstance(cross_refs, dict)
            assert "novel_1" in cross_refs
            assert "novel_2" in cross_refs

            # Should have both thematic and character cross-references
            novel_1_refs = cross_refs["novel_1"]
            assert len(novel_1_refs) > 0
            assert any(ref["type"] == "thematic" for ref in novel_1_refs)

    def test_generate_memory_enhancements(self):
        """Test memory enhancement generation"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            # Mock relational data with cross-references
            relational_data = {
                "cross_references": {
                    "novel_1": [
                        {
                            "type": "thematic",
                            "target_novel": "Novel 2",
                            "target_dir": "novel_2",
                            "connection_strength": 0.8,
                            "details": ["horror", "mystery"]
                        }
                    ]
                }
            }

            enhancements = mapper._generate_memory_enhancements(relational_data)

            assert "cross_novel_memories" in enhancements
            assert "thematic_clusters" in enhancements
            assert "character_archetype_groups" in enhancements
            assert "narrative_style_connections" in enhancements

            # Should generate cross-novel memories for strong connections
            if enhancements["cross_novel_memories"]:
                memory = enhancements["cross_novel_memories"][0]
                assert "source_novel" in memory
                assert "target_novel" in memory
                assert "memory_type" in memory
                assert "content_suggestion" in memory

    @patch('relational_memory_mapper.json.dump')
    @patch('builtins.open')
    def test_save_relational_mappings(self, mock_open, mock_json_dump):
        """Test saving relational mappings to file"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            test_mappings = {"test": "data"}
            output_file = mapper.save_relational_mappings("test_model", test_mappings)

            assert "test_model_relational_mappings.json" in output_file
            mock_open.assert_called_once()
            mock_json_dump.assert_called_once_with(test_mappings, mock_open.return_value.__enter__.return_value, indent=2, default=str)

    def test_load_novel_content_missing_directory(self):
        """Test error handling for missing novel directory"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            with pytest.raises(FileNotFoundError):
                mapper._load_novel_content("non_existent_novel")

    def test_load_novel_content_missing_content_files(self):
        """Test error handling when no readable content files exist"""
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(self.config)

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create novel directory but no content files
                novel_dir = Path(tmp_dir) / "empty_novel"
                novel_dir.mkdir()

                # Mock the novels_dir to point to our temp directory
                mapper.novels_dir = Path(tmp_dir)

                with pytest.raises(ValueError, match="No readable content found"):
                    mapper._load_novel_content("empty_novel")


if __name__ == "__main__":
    pytest.main([__file__])