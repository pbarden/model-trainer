#!/usr/bin/env python3
"""
Test modular architecture components
Simplified tests focusing on core functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import the modular components
from combined_model_trainer import CombinedModelConfig
from relational_memory_mapper import RelationalConfig
from master_training_pipeline import PipelineConfig, PipelineStage


class TestModularArchitectureConfigs:
    """Test all configuration classes in the modular architecture"""

    def test_combined_model_config_defaults(self):
        """Test CombinedModelConfig defaults"""
        config = CombinedModelConfig()

        assert config.combine_novels_method == "concatenate"
        assert config.min_novels_required == 2
        assert config.max_iterations == 6
        assert config.combined_memories_count == 350
        assert config.cross_novel_memories == True

    def test_relational_config_defaults(self):
        """Test RelationalConfig defaults"""
        config = RelationalConfig()

        assert config.similarity_threshold == 0.3
        assert config.theme_overlap_threshold == 0.2
        assert config.max_cross_references_per_novel == 5
        assert config.analyze_themes == True
        assert config.analyze_characters == True

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig defaults"""
        config = PipelineConfig()

        assert config.run_individual_training == True
        assert config.run_combined_training == True
        assert config.run_relational_mapping == True
        assert config.run_validation == True
        assert config.stop_on_error == False

    def test_pipeline_stages_enum(self):
        """Test PipelineStage enum values"""
        assert PipelineStage.INDIVIDUAL_TRAINING.value == "individual_training"
        assert PipelineStage.COMBINED_TRAINING.value == "combined_training"
        assert PipelineStage.RELATIONAL_MAPPING.value == "relational_mapping"
        assert PipelineStage.VALIDATION.value == "validation"
        assert PipelineStage.REPORTING.value == "reporting"


class TestModularArchitectureImports:
    """Test that all modular architecture components can be imported"""

    def test_import_combined_trainer(self):
        """Test importing combined model trainer"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig

        # Should be able to create config
        config = CombinedModelConfig()
        assert config is not None

    def test_import_relational_mapper(self):
        """Test importing relational memory mapper"""
        from relational_memory_mapper import RelationalMemoryMapper, RelationalConfig

        # Should be able to create config
        config = RelationalConfig()
        assert config is not None

    def test_import_master_pipeline(self):
        """Test importing master training pipeline"""
        from master_training_pipeline import MasterTrainingPipeline, PipelineConfig

        # Should be able to create config
        config = PipelineConfig()
        assert config is not None

    def test_import_iterative_trainer(self):
        """Test importing enhanced iterative trainer"""
        from iterative_novel_trainer import IterativeTrainer, IterativeConfig

        # Should be able to create config
        config = IterativeConfig()
        assert config is not None


class TestModularArchitectureBasicFunctionality:
    """Test basic functionality without complex mocking"""

    @pytest.fixture
    def mock_model_mapping(self):
        """Fixture providing mock model mapping"""
        return {
            "metadata": {"total_models": 1},
            "models": {
                "test_model": {
                    "description": "Test Model",
                    "novel_count": 2,
                    "novels": [
                        {"original_name": "Novel 1", "directory_name": "novel_1"},
                        {"original_name": "Novel 2", "directory_name": "novel_2"}
                    ]
                }
            }
        }

    def test_combined_trainer_basic_initialization(self, mock_model_mapping):
        """Test basic combined trainer initialization"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig

        config = CombinedModelConfig()

        # Mock the model mapping loading
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=mock_model_mapping):
            trainer = CombinedModelTrainer(config)

            assert trainer.config == config
            assert trainer.model_mapping == mock_model_mapping

            # Test basic methods
            models = trainer.get_available_models()
            assert "test_model" in models

    def test_relational_mapper_basic_initialization(self, mock_model_mapping):
        """Test basic relational mapper initialization"""
        from relational_memory_mapper import RelationalMemoryMapper, RelationalConfig

        config = RelationalConfig()

        # Mock the model mapping loading
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=mock_model_mapping):
            mapper = RelationalMemoryMapper(config)

            assert mapper.config == config
            assert mapper.model_mapping == mock_model_mapping

            # Test basic methods
            models = mapper.get_available_models()
            assert "test_model" in models

    def test_master_pipeline_basic_initialization(self, mock_model_mapping):
        """Test basic master pipeline initialization"""
        from master_training_pipeline import MasterTrainingPipeline, PipelineConfig

        config = PipelineConfig()

        # Mock the model mapping loading
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=mock_model_mapping):
            pipeline = MasterTrainingPipeline(config)

            assert pipeline.config == config
            assert pipeline.model_mapping == mock_model_mapping

            # Test basic methods
            models = pipeline.get_available_models()
            assert "test_model" in models

    def test_text_processing_functionality(self):
        """Test text processing functionality used across components"""
        from model_tea_utils import TextProcessingUtils

        text = "This is a test sentence. Another sentence here. And a third one."

        # Test progressive chunks
        chunks_0 = TextProcessingUtils.create_progressive_chunks(text, 0, base_size=10)
        chunks_2 = TextProcessingUtils.create_progressive_chunks(text, 2, base_size=10)

        assert len(chunks_0) > 0
        assert len(chunks_2) > 0
        # Later iterations should generally have fewer, larger chunks
        assert len(chunks_0) >= len(chunks_2)

    def test_quality_validation_integration(self):
        """Test quality validation integration"""
        from quality_validator import QualityValidator, ValidationConfig

        config = ValidationConfig()
        validator = QualityValidator(config)

        # Test basic functionality
        assert validator.should_continue_training(0) == True
        assert validator.should_continue_training(5) == True

        # Test analysis with some mock data
        validator.perplexity_history = [10.0, 8.0, 6.0]
        validator.generation_quality_history = [0.8, 0.85, 0.9]

        analysis = validator.get_training_analysis(2)
        assert "iteration" in analysis
        assert "current_perplexity" in analysis
        assert "current_quality" in analysis

    def test_file_system_utilities(self):
        """Test file system utilities used across components"""
        from model_tea_utils import FileSystemUtils

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test directory creation
            test_dir = Path(tmp_dir) / "test_subdir"
            result = FileSystemUtils.ensure_directory(test_dir)

            assert result.exists()
            assert result.is_dir()

            # Test JSON operations
            test_data = {"test": "data", "number": 42}
            test_file = test_dir / "test.json"

            success = FileSystemUtils.safe_json_save(test_data, test_file)
            assert success
            assert test_file.exists()

            loaded_data = FileSystemUtils.safe_json_load(test_file)
            assert loaded_data == test_data


class TestModularArchitectureIntegration:
    """Test integration between modular components"""

    @pytest.fixture
    def mock_mapping_with_multiple_models(self):
        """Fixture with multiple models for integration testing"""
        return {
            "metadata": {"total_models": 2},
            "models": {
                "model_a": {
                    "description": "Model A",
                    "novels": [
                        {"original_name": "Novel A1", "directory_name": "novel_a1"},
                        {"original_name": "Novel A2", "directory_name": "novel_a2"}
                    ]
                },
                "model_b": {
                    "description": "Model B",
                    "novels": [
                        {"original_name": "Novel B1", "directory_name": "novel_b1"},
                        {"original_name": "Novel B2", "directory_name": "novel_b2"}
                    ]
                }
            }
        }

    def test_consistent_model_lists(self, mock_mapping_with_multiple_models):
        """Test that all components see the same model lists"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig
        from relational_memory_mapper import RelationalMemoryMapper, RelationalConfig
        from master_training_pipeline import MasterTrainingPipeline, PipelineConfig

        # Create all components with same mapping
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=mock_mapping_with_multiple_models):
            with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=mock_mapping_with_multiple_models):
                with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=mock_mapping_with_multiple_models):

                    combined_trainer = CombinedModelTrainer(CombinedModelConfig())
                    mapper = RelationalMemoryMapper(RelationalConfig())
                    pipeline = MasterTrainingPipeline(PipelineConfig())

                    # All should see the same models
                    combined_models = combined_trainer.get_available_models()
                    mapper_models = mapper.get_available_models()
                    pipeline_models = pipeline.get_available_models()

                    assert set(combined_models) == set(mapper_models)
                    assert set(mapper_models) == set(pipeline_models)
                    assert "model_a" in combined_models
                    assert "model_b" in combined_models

    def test_novel_list_consistency(self, mock_mapping_with_multiple_models):
        """Test that components return consistent novel lists"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig
        from master_training_pipeline import MasterTrainingPipeline, PipelineConfig

        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=mock_mapping_with_multiple_models):
            with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=mock_mapping_with_multiple_models):

                combined_trainer = CombinedModelTrainer(CombinedModelConfig())
                pipeline = MasterTrainingPipeline(PipelineConfig())

                # Both should return the same novels for the same model
                combined_novels = combined_trainer.get_novels_for_model("model_a")
                pipeline_novels = pipeline.get_novels_for_model("model_a")

                assert len(combined_novels) == len(pipeline_novels)
                for i, novel in enumerate(combined_novels):
                    assert novel["original_name"] == pipeline_novels[i]["original_name"]
                    assert novel["directory_name"] == pipeline_novels[i]["directory_name"]

    def test_configuration_inheritance(self):
        """Test that configurations are properly inherited and compatible"""
        from iterative_novel_trainer import IterativeConfig
        from combined_model_trainer import CombinedModelConfig

        # Both should have compatible settings
        iterative_config = IterativeConfig()
        combined_config = CombinedModelConfig()

        # Same base model
        assert iterative_config.base_model == combined_config.base_config.base_model

        # Compatible learning rates
        assert iterative_config.learning_rate_start == combined_config.learning_rate_start
        assert iterative_config.learning_rate_end == combined_config.learning_rate_end

        # Compatible iteration counts
        assert iterative_config.iterations_per_novel == combined_config.max_iterations


class TestModularArchitectureErrorHandling:
    """Test error handling across modular components"""

    def test_missing_model_mapping_error(self):
        """Test error handling when model mapping is missing"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig

        # Mock missing file
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                CombinedModelTrainer(CombinedModelConfig())

    def test_invalid_model_key_error(self):
        """Test error handling for invalid model keys"""
        from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig

        mock_mapping = {"models": {"valid_model": {"novels": []}}}

        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=mock_mapping):
            trainer = CombinedModelTrainer(CombinedModelConfig())

            with pytest.raises(ValueError, match="not found"):
                trainer.get_novels_for_model("invalid_model")

    def test_memory_system_fallback(self):
        """Test memory system fallback behavior"""
        from model_tea_utils import MemorySystemUtils

        # This should always work, even if memory system is unavailable
        available = MemorySystemUtils.check_memory_system_available()
        assert isinstance(available, bool)

        if not available:
            fallback = MemorySystemUtils.create_memory_fallback()
            assert isinstance(fallback, dict)
            assert fallback["system_status"] == "unavailable"


if __name__ == "__main__":
    pytest.main([__file__])