#!/usr/bin/env python3
"""
Test quality validator functionality
"""

import pytest
from unittest.mock import Mock, patch

from quality_validator import QualityValidator, ValidationConfig


class TestValidationConfig:
    """Test ValidationConfig dataclass"""

    def test_default_config(self):
        """Test default validation configuration"""
        config = ValidationConfig()

        assert config.perplexity_threshold == 50.0
        assert config.quality_threshold == 0.8
        assert config.max_repetition_penalty == 1.2
        assert config.temperature_range == (0.7, 1.0)
        assert config.sample_length == 100


class TestQualityValidator:
    """Test QualityValidator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = ValidationConfig(
            perplexity_threshold=25.0,
            quality_threshold=0.7,
            sample_length=50
        )
        self.validator = QualityValidator(self.config)

    def test_validator_initialization(self):
        """Test validator initialization"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        assert validator.config == config
        assert validator.perplexity_history == []
        assert validator.generation_quality_history == []

    def test_should_continue_training(self):
        """Test training continuation decision"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Should always continue for comprehensive analysis
        for iteration in range(10):
            assert validator.should_continue_training(iteration) == True

    def test_get_training_analysis(self):
        """Test training analysis generation"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Add some test data
        validator.perplexity_history = [10.0, 8.0, 6.0]
        validator.generation_quality_history = [0.8, 0.85, 0.9]

        analysis = validator.get_training_analysis(2)

        assert "iteration" in analysis
        assert "current_perplexity" in analysis
        assert "current_quality" in analysis
        assert "perplexity_trend" in analysis
        assert "quality_trend" in analysis
        assert "convergence_status" in analysis

        assert analysis["iteration"] == 3
        assert analysis["current_perplexity"] == 6.0
        assert analysis["current_quality"] == 0.9

    def test_calculate_trend(self):
        """Test trend calculation"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test improving trend
        improving_values = [10.0, 8.0, 6.0, 4.0]
        trend = validator._calculate_trend(improving_values)
        assert trend == "improving"

        # Test degrading trend
        degrading_values = [4.0, 6.0, 8.0, 10.0]
        trend = validator._calculate_trend(degrading_values)
        assert trend == "degrading"

        # Test insufficient data
        insufficient_values = [10.0, 8.0]
        trend = validator._calculate_trend(insufficient_values)
        assert trend == "insufficient_data"

    def test_is_improving(self):
        """Test improvement detection"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test improving case
        validator.perplexity_history = [10.0, 8.0]
        validator.generation_quality_history = [0.8, 0.9]
        assert validator._is_improving() == True

        # Test non-improving case
        validator.perplexity_history = [5.0, 8.0]
        validator.generation_quality_history = [0.9, 0.8]
        assert validator._is_improving() == False

    def test_check_overfitting_risk(self):
        """Test overfitting detection"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test no overfitting
        validator.generation_quality_history = [0.8, 0.85, 0.9]
        assert validator._check_overfitting_risk() == False

        # Test overfitting pattern
        validator.generation_quality_history = [0.9, 0.85, 0.8]
        assert validator._check_overfitting_risk() == True

        # Test insufficient data
        validator.generation_quality_history = [0.8, 0.9]
        assert validator._check_overfitting_risk() == False

    def test_assess_training_stability(self):
        """Test training stability assessment"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test very stable
        validator.perplexity_history = [10.0, 10.1, 9.9, 10.0]
        validator.generation_quality_history = [0.8, 0.81, 0.79, 0.8]
        stability = validator._assess_training_stability()
        assert stability in ["very_stable", "stable", "moderately_stable", "unstable"]

        # Test insufficient data
        validator.perplexity_history = [10.0, 8.0]
        stability = validator._assess_training_stability()
        assert stability == "insufficient_data"

    def test_reset(self):
        """Test validator reset"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        validator.perplexity_history = [10.0, 8.0]
        validator.generation_quality_history = [0.8, 0.9]

        validator.reset()

        assert validator.perplexity_history == []
        assert validator.generation_quality_history == []

    def test_get_summary_statistics(self):
        """Test summary statistics generation"""
        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test with no data
        summary = validator.get_summary_statistics()
        assert "error" in summary

        # Test with data
        validator.perplexity_history = [10.0, 8.0, 6.0]
        validator.generation_quality_history = [0.8, 0.85, 0.9]

        summary = validator.get_summary_statistics()

        assert "total_iterations" in summary
        assert "perplexity_stats" in summary
        assert "quality_stats" in summary
        assert "overall_assessment" in summary

        assert summary["total_iterations"] == 3


if __name__ == "__main__":
    pytest.main([__file__])