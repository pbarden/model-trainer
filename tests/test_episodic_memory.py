"""
Test the episodic memory system functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from episodic_memory_system import EpisodicMemorySystem, MemoryConfig, Memory


class TestMemoryConfig:
    """Test memory configuration"""

    def test_default_config(self):
        """Test default memory configuration values"""
        config = MemoryConfig()

        assert config.memory_chunk_size == 35
        assert config.overlap_ratio == 0.3
        assert config.max_memories_per_novel == 250
        assert config.max_retrieved_memories == 5
        assert config.relevance_threshold == 0.1
        assert config.use_simple_similarity == True

    def test_config_customization(self):
        """Test custom memory configuration"""
        config = MemoryConfig(
            memory_chunk_size=50,
            max_memories_per_novel=100,
            relevance_threshold=0.2
        )

        assert config.memory_chunk_size == 50
        assert config.max_memories_per_novel == 100
        assert config.relevance_threshold == 0.2


class TestMemory:
    """Test individual memory objects"""

    def test_memory_creation(self):
        """Test episodic memory object creation"""
        memory = Memory(
            content="Test memory content",
            memory_type="character",
            keywords=["test", "memory"],
            context="test context",
            emotional_tone="neutral",
            importance_score=0.5,
            chapter_position=1
        )

        assert memory.content == "Test memory content"
        assert memory.memory_type == "character"
        assert memory.keywords == ["test", "memory"]
        assert memory.context == "test context"
        assert memory.emotional_tone == "neutral"
        assert memory.importance_score == 0.5
        assert memory.chapter_position == 1

    def test_memory_serialization(self):
        """Test memory can be serialized for storage"""
        memory = Memory(
            content="Test content",
            memory_type="location",
            keywords=["place"],
            context="scene",
            emotional_tone="dark",
            importance_score=0.8,
            chapter_position=5
        )

        memory_dict = {
            'content': memory.content,
            'memory_type': memory.memory_type,
            'keywords': memory.keywords,
            'context': memory.context,
            'emotional_tone': memory.emotional_tone,
            'importance_score': memory.importance_score,
            'chapter_position': memory.chapter_position
        }

        assert isinstance(json.dumps(memory_dict), str)


class TestEpisodicMemorySystem:
    """Test the episodic memory system"""

    def test_system_initialization(self):
        """Test memory system can be initialized"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        assert system.config == config
        assert hasattr(system, 'extractor')
        assert hasattr(system, 'retriever')

    def test_text_chunking(self):
        """Test text chunking functionality"""
        config = MemoryConfig(memory_chunk_size=10, overlap_ratio=0.2)
        system = EpisodicMemorySystem(config)

        test_text = "This is a very long test text that contains many words and should definitely be chunked into multiple smaller pieces for proper memory processing. It has enough content to trigger the chunking mechanism of the episodic memory system."
        chunks = system.extractor._create_chunks(test_text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_memory_extraction_from_text(self):
        """Test extracting memories from text"""
        config = MemoryConfig(max_memories_per_novel=10)
        system = EpisodicMemorySystem(config)

        test_text = """
        The detective walked into the dimly lit room. "Something isn't right here," he muttered.
        The ancient desk was covered in dust, and strange symbols were carved into its surface.
        He felt a chill run down his spine as he approached the mysterious artifacts.
        """

        memories = system.extractor.extract_memories(test_text, "test_novel")

        assert isinstance(memories, list)
        assert len(memories) >= 0
        assert len(memories) <= config.max_memories_per_novel
        assert all(isinstance(memory, Memory) for memory in memories)

    def test_emotional_tone_detection(self):
        """Test emotional tone detection"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        positive_text = "The beautiful sunset filled everyone with joy and happiness."
        negative_text = "The terrible storm brought fear and destruction to the village."
        neutral_text = "The meeting was scheduled for three o'clock in the afternoon."

        pos_tone = system.extractor._detect_emotional_tone(positive_text)
        neg_tone = system.extractor._detect_emotional_tone(negative_text)
        neu_tone = system.extractor._detect_emotional_tone(neutral_text)

        assert pos_tone in ["positive", "neutral", "joy", "happiness", "excited", "happy"]
        assert neg_tone in ["negative", "neutral", "fear", "dark", "angry", "sad"]
        assert neu_tone in ["neutral", "calm"]

    def test_build_memory_for_model(self):
        """Test building memory for a model"""
        config = MemoryConfig(max_memories_per_novel=5)
        system = EpisodicMemorySystem(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "test_novel.txt"
            novel_path.write_text("This is a test novel. The hero walked through the forest.")

            result = system.build_memory_for_model("test_model", novel_path)

            assert isinstance(result, dict)
            assert "memories" in result
            assert "analysis" in result

    def test_memory_retrieval(self):
        """Test memory retrieval functionality"""
        config = MemoryConfig(max_retrieved_memories=3)
        system = EpisodicMemorySystem(config)

        test_memories = [
            Memory("Castle on the hill", "location", ["castle", "hill"], "", "", 0.8, 1),
            Memory("Knight in armor", "character", ["knight", "armor"], "", "", 0.6, 2),
            Memory("Dark forest path", "location", ["forest", "dark"], "", "", 0.7, 3),
            Memory("Ancient sword", "object", ["sword", "ancient"], "", "", 0.5, 4)
        ]

        query = "Tell me about the castle"
        retrieved = system.retriever.retrieve_memories(query, test_memories)

        assert isinstance(retrieved, list)
        assert len(retrieved) <= config.max_retrieved_memories

    def test_load_memory_for_model(self):
        """Test loading memory for a model"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        result = system.load_memory_for_model("nonexistent_model")

        assert isinstance(result, bool)

    def test_activate_memories(self):
        """Test activating memories with a prompt"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "test_novel.txt"
            novel_path.write_text("The wizard cast a spell in the enchanted forest.")

            system.build_memory_for_model("test_model", novel_path)

            memories, stats = system.activate_memories("test_model", "Tell me about magic")

            assert isinstance(memories, list)
            assert isinstance(stats, dict)

    def test_importance_calculation(self):
        """Test importance score calculation"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        important_text = "The king declared war against the neighboring kingdom."
        mundane_text = "It was a sunny day."

        important_score = system.extractor._calculate_importance(important_text)
        mundane_score = system.extractor._calculate_importance(mundane_text)

        assert isinstance(important_score, float)
        assert isinstance(mundane_score, float)
        assert 0.0 <= important_score <= 1.0
        assert 0.0 <= mundane_score <= 1.0

    def test_character_memory_extraction(self):
        """Test character memory extraction"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        character_text = "John walked into the room with a determined expression on his face."
        memories = system.extractor._extract_character_memories(character_text, 0.5)

        assert isinstance(memories, list)
        assert all(isinstance(memory, Memory) for memory in memories)
        if memories:
            assert memories[0].memory_type == "character"

    def test_location_memory_extraction(self):
        """Test location memory extraction"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        location_text = "The ancient library was filled with dusty books and mysterious shadows."
        memories = system.extractor._extract_location_memories(location_text, 0.5)

        assert isinstance(memories, list)
        assert all(isinstance(memory, Memory) for memory in memories)
        if memories:
            assert memories[0].memory_type == "location"

    def test_dialogue_memory_extraction(self):
        """Test dialogue memory extraction"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        dialogue_text = '"Hello there," she said with a warm smile.'
        memories = system.extractor._extract_dialogue_memories(dialogue_text, 0.5)

        assert isinstance(memories, list)
        assert all(isinstance(memory, Memory) for memory in memories)
        if memories:
            assert memories[0].memory_type == "dialogue"


class TestEnhancedMemoryIntegration:
    """Test enhanced memory integration with tagging system"""

    def test_load_novel_tags(self):
        """Test loading tags.json from novel directory"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        sample_tags = {
            "title": "Test Novel",
            "themes": ["adventure", "mystery", "horror"],
            "characters": ["Alice", "Rabbit"],
            "locations": ["Wonderland", "Forest"],
            "scene_types": ["action_scene", "dialogue_scene"],
            "behaviors": ["heroic", "mysterious"],
            "narrative_elements": ["third_person", "foreshadowing"],
            "social_dynamics": ["family", "friendship"]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_dir = Path(temp_dir) / "test_novel"
            novel_dir.mkdir()

            novel_file = novel_dir / "test_novel.txt"
            novel_file.write_text("Sample novel content")

            tags_file = novel_dir / "tags.json"
            with open(tags_file, 'w') as f:
                json.dump(sample_tags, f)

            loaded_tags = system.load_novel_tags(novel_file)

            assert loaded_tags is not None
            assert loaded_tags["title"] == "Test Novel"
            assert "adventure" in loaded_tags["themes"]
            assert "Alice" in loaded_tags["characters"]
            assert "action_scene" in loaded_tags["scene_types"]

    def test_load_novel_tags_missing_file(self):
        """Test loading tags when tags.json doesn't exist"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_file = Path(temp_dir) / "novel.txt"
            novel_file.write_text("Sample content")

            loaded_tags = system.load_novel_tags(novel_file)
            assert loaded_tags is None

    def test_enhance_memories_with_tags(self):
        """Test enhancing memories with tag data"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        sample_memories = [
            Memory(
                content="Alice spoke to the mysterious rabbit about their adventure",
                memory_type="character",
                keywords=["conversation"],
                context="dialogue scene",
                emotional_tone=0.6,
                importance_score=0.8,
                chapter_position=0.3
            ),
            Memory(
                content="The dark forest was filled with terrifying creatures",
                memory_type="location",
                keywords=["setting"],
                context="description",
                emotional_tone=0.2,
                importance_score=0.7,
                chapter_position=0.5
            )
        ]

        tags_data = {
            "characters": ["Alice", "Rabbit"],
            "locations": ["Forest", "Wonderland"],
            "themes": ["horror", "adventure", "mystery"],
            "scene_types": ["dialogue_scene", "action_scene"],
            "behaviors": ["mysterious", "heroic"],
            "narrative_elements": ["foreshadowing"],
            "social_dynamics": ["friendship"]
        }

        enhanced_memories = system.enhance_memories_with_tags(sample_memories, tags_data)

        assert len(enhanced_memories) == 2
        assert all(isinstance(memory, Memory) for memory in enhanced_memories)

        character_memory = enhanced_memories[0]
        assert "Alice" in character_memory.keywords
        assert character_memory.importance_score >= sample_memories[0].importance_score

        location_memory = enhanced_memories[1]
        assert "Forest" in location_memory.keywords
        assert location_memory.importance_score >= sample_memories[1].importance_score

    def test_theme_keyword_enhancement(self):
        """Test theme-based keyword enhancement"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        memory_with_horror = Memory(
            content="The terrifying creature filled everyone with fear and dread",
            memory_type="description",
            keywords=["creature"],
            context="horror scene",
            emotional_tone=0.1,
            importance_score=0.6,
            chapter_position=0.4
        )

        tags_data = {
            "themes": ["horror", "thriller"],
            "characters": [],
            "locations": []
        }

        enhanced_memories = system.enhance_memories_with_tags([memory_with_horror], tags_data)
        enhanced_memory = enhanced_memories[0]

        horror_keywords = ['fear', 'terror', 'dark', 'evil']
        thriller_keywords = ['chase', 'tension', 'suspense', 'danger']

        found_horror_keywords = [kw for kw in horror_keywords if kw in enhanced_memory.keywords]
        found_thriller_keywords = [kw for kw in thriller_keywords if kw in enhanced_memory.keywords]

        assert len(found_horror_keywords) > 0 or len(found_thriller_keywords) > 0

    def test_enhanced_theme_keywords_coverage(self):
        """Test coverage of enhanced theme keywords"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        memory_content = Memory(
            content="The space agent used advanced technology on an alien planet",
            memory_type="description",
            keywords=["space"],
            context="scifi scene",
            emotional_tone=0.5,
            importance_score=0.7,
            chapter_position=0.2
        )

        tags_data = {
            "themes": ["scifi", "spy", "space"],
            "characters": [],
            "locations": []
        }

        enhanced_memories = system.enhance_memories_with_tags([memory_content], tags_data)
        enhanced_memory = enhanced_memories[0]

        expected_scifi_keywords = ['space', 'technology', 'alien']
        expected_spy_keywords = ['agent']

        found_keywords = enhanced_memory.keywords
        scifi_matches = [kw for kw in expected_scifi_keywords if kw in found_keywords]
        spy_matches = [kw for kw in expected_spy_keywords if kw in found_keywords]

        assert len(scifi_matches) > 0 or len(spy_matches) > 0

    def test_build_memory_with_tag_integration(self):
        """Test full memory building with tag integration"""
        config = MemoryConfig(memory_chunk_size=20, max_memories_per_novel=10)
        system = EpisodicMemorySystem(config)

        sample_text = """
        Alice was a brave and mysterious character who loved adventure.
        She traveled through the magical forest filled with danger and wonder.
        The intelligent rabbit spoke with fear about the approaching evil.
        Their friendship would be tested in this terrifying place.
        """

        sample_tags = {
            "title": "Alice Adventure",
            "themes": ["adventure", "mystery", "horror"],
            "characters": ["Alice", "Rabbit"],
            "locations": ["Forest"],
            "scene_types": ["dialogue_scene", "action_scene"],
            "behaviors": ["heroic", "mysterious", "intelligent", "fearful"],
            "narrative_elements": ["third_person"],
            "social_dynamics": ["friendship"]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_dir = Path(temp_dir) / "alice_adventure"
            novel_dir.mkdir()

            novel_file = novel_dir / "alice_adventure.txt"
            novel_file.write_text(sample_text)

            tags_file = novel_dir / "tags.json"
            with open(tags_file, 'w') as f:
                json.dump(sample_tags, f)

            result = system.build_memory_for_model("alice_model", novel_file)

            assert "memories" in result
            assert "analysis" in result
            assert len(result["memories"]) > 0

            memories = result["memories"]
            assert all(isinstance(memory, Memory) for memory in memories)

            enhanced_keywords = []
            for memory in memories:
                enhanced_keywords.extend(memory.keywords)

            assert "Alice" in enhanced_keywords or "Rabbit" in enhanced_keywords

    def test_memory_importance_boosting(self):
        """Test that tag matches boost memory importance"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        character_memory = Memory(
            content="Alice walked through the mysterious garden",
            memory_type="character",
            keywords=["walk"],
            context="action",
            emotional_tone=0.5,
            importance_score=0.5,
            chapter_position=0.3
        )

        non_character_memory = Memory(
            content="The weather was pleasant that day",
            memory_type="description",
            keywords=["weather"],
            context="setting",
            emotional_tone=0.6,
            importance_score=0.5,
            chapter_position=0.4
        )

        tags_data = {
            "characters": ["Alice"],
            "locations": ["Garden"],
            "themes": ["mystery"]
        }

        enhanced_memories = system.enhance_memories_with_tags(
            [character_memory, non_character_memory], tags_data
        )

        enhanced_character = enhanced_memories[0]
        enhanced_non_character = enhanced_memories[1]

        assert enhanced_character.importance_score >= character_memory.importance_score
        assert enhanced_non_character.importance_score == non_character_memory.importance_score

    def test_tag_integration_with_different_memory_types(self):
        """Test tag enhancement across different memory types"""
        config = MemoryConfig()
        system = EpisodicMemorySystem(config)

        memories = [
            Memory("Alice said hello", "character", ["greeting"], "dialogue", 0.5, 0.5, 0.1),
            Memory("Dark forest ahead", "location", ["trees"], "setting", 0.3, 0.6, 0.2),
            Memory("Magic spell cast", "action", ["magic"], "event", 0.7, 0.8, 0.3),
            Memory("Feeling of dread", "emotion", ["fear"], "internal", 0.2, 0.4, 0.4)
        ]

        tags_data = {
            "characters": ["Alice"],
            "locations": ["Forest"],
            "themes": ["fantasy", "horror"],
            "behaviors": ["mysterious"],
            "scene_types": ["dialogue_scene"],
            "narrative_elements": ["third_person"],
            "social_dynamics": ["friendship"]
        }

        enhanced_memories = system.enhance_memories_with_tags(memories, tags_data)

        assert len(enhanced_memories) == 4

        for i, enhanced_memory in enumerate(enhanced_memories):
            original_memory = memories[i]
            assert enhanced_memory.memory_type == original_memory.memory_type
            assert len(enhanced_memory.keywords) >= len(original_memory.keywords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])