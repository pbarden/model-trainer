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

        test_text = "This is a test text that should be chunked into smaller pieces for memory processing."
        chunks = system.extractor._create_chunks(test_text)

        assert len(chunks) > 1
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])