#!/usr/bin/env python3
"""
Test configuration classes and utility functions
"""

import pytest
from dataclasses import dataclass
from pathlib import Path

from model_tea_utils import ModelTeaConfig, validate_system_setup
from iterative_novel_trainer import IterativeConfig
from quality_validator import ValidationConfig


class TestModelTeaConfig:
    """Test ModelTeaConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ModelTeaConfig()

        assert config.base_model == "distilgpt2"
        assert config.iterations_per_novel == 12
        assert config.max_memories_per_novel == 250
        assert config.memory_chunk_size == 35
        assert config.memory_retrieval_limit == 5

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = ModelTeaConfig()

        # Test reasonable ranges
        assert 1 <= config.iterations_per_novel <= 15
        assert 50 <= config.max_memories_per_novel <= 1000
        assert 10 <= config.memory_chunk_size <= 100
        assert 1 <= config.memory_retrieval_limit <= 10


class TestIterativeConfig:
    """Test IterativeConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = IterativeConfig()

        assert config.base_model == "distilgpt2"
        assert config.iterations_per_novel == 12
        assert config.learning_rate_start == 5e-5
        assert config.learning_rate_end == 5e-6

    def test_learning_rate_progression(self):
        """Test learning rate makes sense"""
        config = IterativeConfig()

        assert config.learning_rate_start > config.learning_rate_end
        assert config.learning_rate_end > 0


class TestValidationConfig:
    """Test ValidationConfig dataclass"""

    def test_default_config(self):
        """Test default validation configuration"""
        config = ValidationConfig()

        assert config.perplexity_threshold > 0
        assert config.quality_threshold > 0
        assert 0 < config.quality_threshold < 1


class TestSystemValidation:
    """Test system setup validation"""

    def test_validate_system_setup(self):
        """Test system validation function"""
        validation = validate_system_setup()

        assert isinstance(validation, dict)
        assert "directories" in validation
        assert "dependencies" in validation
        assert "memory_system" in validation

        # All values should be boolean
        for key, value in validation.items():
            assert isinstance(value, bool), f"{key} should be boolean, got {type(value)}"


if __name__ == "__main__":
    pytest.main([__file__])