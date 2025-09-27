#!/usr/bin/env python3
"""
Test utility functions and helper classes
"""

import pytest
import tempfile
import json
from pathlib import Path

from model_tea_utils import (
    FileSystemUtils, TextProcessingUtils, TrainingUtils,
    QualityMetrics, MemorySystemUtils, ErrorHandling
)


class TestFileSystemUtils:
    """Test filesystem utility functions"""

    def test_ensure_directory(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "test_dir" / "nested"
            result = FileSystemUtils.ensure_directory(test_path)

            assert result.exists()
            assert result.is_dir()
            assert result == test_path

    def test_safe_json_save_and_load(self):
        """Test JSON save and load operations"""
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"

            # Test save
            success = FileSystemUtils.safe_json_save(test_data, file_path)
            assert success
            assert file_path.exists()

            # Test load
            loaded_data = FileSystemUtils.safe_json_load(file_path)
            assert loaded_data == test_data

    def test_safe_json_load_missing_file(self):
        """Test loading non-existent JSON file"""
        missing_file = Path("non_existent_file.json")
        result = FileSystemUtils.safe_json_load(missing_file)
        assert result is None


class TestTextProcessingUtils:
    """Test text processing utilities"""

    def test_create_progressive_chunks(self):
        """Test progressive chunk creation"""
        text = "This is a test. Another sentence. And another one. Final sentence."

        # Test different iterations
        chunks_iter_0 = TextProcessingUtils.create_progressive_chunks(text, 0, base_size=10)
        chunks_iter_2 = TextProcessingUtils.create_progressive_chunks(text, 2, base_size=10)

        assert len(chunks_iter_0) > 0
        assert len(chunks_iter_2) > 0
        # Later iterations should have fewer, larger chunks
        assert len(chunks_iter_0) >= len(chunks_iter_2)

    def test_calculate_text_stats(self):
        """Test text statistics calculation"""
        text = "This is a test sentence. Another sentence here."
        stats = TextProcessingUtils.calculate_text_stats(text)

        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "character_count" in stats
        assert "estimated_reading_time" in stats

        assert stats["word_count"] > 0
        assert stats["sentence_count"] >= 1
        assert stats["character_count"] > 0


class TestTrainingUtils:
    """Test training utility functions"""

    def test_calculate_learning_rate(self):
        """Test learning rate calculation"""
        start_lr = 5e-5
        end_lr = 1e-5
        total_iterations = 6

        # Test first iteration
        lr_0 = TrainingUtils.calculate_learning_rate(0, total_iterations, start_lr, end_lr)
        assert lr_0 == start_lr

        # Test last iteration
        lr_last = TrainingUtils.calculate_learning_rate(total_iterations-1, total_iterations, start_lr, end_lr)
        assert lr_last == end_lr

        # Test middle iteration
        lr_mid = TrainingUtils.calculate_learning_rate(2, total_iterations, start_lr, end_lr)
        assert end_lr < lr_mid < start_lr

    def test_calculate_chunk_size(self):
        """Test chunk size calculation"""
        base_size = 200

        size_0 = TrainingUtils.calculate_chunk_size(0, base_size)
        size_2 = TrainingUtils.calculate_chunk_size(2, base_size)

        assert size_0 == base_size
        assert size_2 > size_0

    def test_format_training_time(self):
        """Test time formatting"""
        # Test seconds
        assert "s" in TrainingUtils.format_training_time(45)

        # Test minutes
        assert "m" in TrainingUtils.format_training_time(120)

        # Test hours
        assert "h" in TrainingUtils.format_training_time(3700)


class TestQualityMetrics:
    """Test quality assessment functions"""

    def test_assess_text_quality(self):
        """Test text quality assessment"""
        text = "This is a well-written sentence with good vocabulary and structure."
        metrics = QualityMetrics.assess_text_quality(text)

        assert "quality_score" in metrics
        assert "repetition_ratio" in metrics
        assert "vocabulary_diversity" in metrics

        assert 0 <= metrics["quality_score"] <= 1
        assert metrics["vocabulary_diversity"] > 0

    def test_calculate_perplexity_from_loss(self):
        """Test perplexity calculation"""
        loss = 2.0
        perplexity = QualityMetrics.calculate_perplexity_from_loss(loss)

        assert perplexity > 0
        # e^2 ≈ 7.39
        assert abs(perplexity - 7.39) < 0.1


class TestMemorySystemUtils:
    """Test memory system utilities"""

    def test_check_memory_system_available(self):
        """Test memory system availability check"""
        # This should return a boolean
        available = MemorySystemUtils.check_memory_system_available()
        assert isinstance(available, bool)

    def test_create_memory_fallback(self):
        """Test memory fallback creation"""
        fallback = MemorySystemUtils.create_memory_fallback()

        assert isinstance(fallback, dict)
        assert "system_status" in fallback
        assert fallback["system_status"] == "unavailable"


class TestErrorHandling:
    """Test error handling utilities"""

    def test_handle_training_error(self):
        """Test training error handling"""
        error = ValueError("Test error")
        context = "test_context"

        result = ErrorHandling.handle_training_error(error, context)

        assert isinstance(result, dict)
        assert "error" in result
        assert "context" in result
        assert "status" in result
        assert result["context"] == context

    def test_handle_memory_error(self):
        """Test memory error handling"""
        error = ImportError("Memory module not found")
        context = "memory_test"

        result = ErrorHandling.handle_memory_error(error, context)

        assert isinstance(result, dict)
        assert result["status"] == "memory_disabled"


if __name__ == "__main__":
    pytest.main([__file__])