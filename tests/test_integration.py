#!/usr/bin/env python3
"""
Integration tests for complete Model Tea workflows
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import all the main components
from iterative_novel_trainer import IterativeTrainer, IterativeConfig
from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig
from relational_memory_mapper import RelationalMemoryMapper, RelationalConfig
from master_training_pipeline import MasterTrainingPipeline, PipelineConfig
from model_tea_utils import validate_system_setup, ModelTeaConfig


class TestSystemIntegration:
    """Test system-wide integration"""

    def test_system_validation(self):
        """Test system validation passes"""
        validation = validate_system_setup()

        assert isinstance(validation, dict)
        assert "directories" in validation
        assert "dependencies" in validation
        assert "memory_system" in validation

        # At least some components should be available
        assert any(validation.values())

    def test_configuration_compatibility(self):
        """Test that all configuration classes are compatible"""
        # Create all config types
        model_tea_config = ModelTeaConfig()
        iterative_config = IterativeConfig()
        combined_config = CombinedModelConfig()
        relational_config = RelationalConfig()
        pipeline_config = PipelineConfig()

        # Test that they all have reasonable defaults
        assert model_tea_config.base_model == "distilgpt2"
        assert iterative_config.base_model == "distilgpt2"
        assert combined_config.max_iterations == 6
        assert relational_config.similarity_threshold > 0
        assert pipeline_config.run_individual_training == True

        # Test that learning rates are compatible
        assert iterative_config.learning_rate_start == combined_config.learning_rate_start
        assert iterative_config.learning_rate_end == combined_config.learning_rate_end

    def test_module_imports(self):
        """Test that all modules can be imported without errors"""
        # Test that we can create instances of all main classes
        try:
            with patch('builtins.open'), patch('json.load', return_value={"models": {}}):
                with patch('pathlib.Path.exists', return_value=True):
                    # These should not raise import errors
                    trainer = IterativeTrainer(IterativeConfig())
                    combined_trainer = CombinedModelTrainer(CombinedModelConfig())
                    mapper = RelationalMemoryMapper(RelationalConfig())
                    pipeline = MasterTrainingPipeline(PipelineConfig())

                    assert trainer is not None
                    assert combined_trainer is not None
                    assert mapper is not None
                    assert pipeline is not None
        except ImportError as e:
            pytest.fail(f"Module import failed: {e}")


class TestWorkflowIntegration:
    """Test complete workflow integration"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        # Mock model mapping for all tests
        self.mock_mapping = {
            "metadata": {
                "total_models": 1,
                "total_novels_assigned": 2
            },
            "models": {
                "test_model": {
                    "description": "Test Model",
                    "novel_count": 2,
                    "novels": [
                        {"original_name": "Test Novel 1", "directory_name": "test_novel_1"},
                        {"original_name": "Test Novel 2", "directory_name": "test_novel_2"}
                    ]
                }
            }
        }

        # Mock novel contents
        self.mock_contents = {
            "test_novel_1": "This is the content of the first test novel. It contains adventure and mystery themes.",
            "test_novel_2": "This is the content of the second test novel. It also has adventure themes and some horror elements."
        }

    @patch('iterative_novel_trainer.IterativeTrainer')
    @patch('combined_model_trainer.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_individual_to_combined_workflow(self, mock_json_load, mock_open, mock_path, mock_trainer_class):
        """Test workflow from individual training to combined model"""
        # Setup mocks
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        # Mock file reading for novel contents
        def mock_file_read(file_path, *args, **kwargs):
            mock_file = Mock()
            if "test_novel_1" in str(file_path):
                mock_file.read.return_value = self.mock_contents["test_novel_1"]
            elif "test_novel_2" in str(file_path):
                mock_file.read.return_value = self.mock_contents["test_novel_2"]
            else:
                mock_file.read.return_value = ""
            return mock_file

        mock_open.side_effect = mock_file_read

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer._train_with_content.return_value = {
            "iterations": [
                {"iteration": 1, "perplexity": 20.0, "quality_score": 0.8},
                {"iteration": 2, "perplexity": 15.0, "quality_score": 0.9}
            ],
            "final_quality": 0.9,
            "training_time": 600
        }
        mock_trainer_class.return_value = mock_trainer

        # Test combined training workflow
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            combined_trainer = CombinedModelTrainer(CombinedModelConfig())

            # Check novel status
            novels = combined_trainer.get_novels_for_model("test_model")
            assert len(novels) == 2

            # Test combining content
            combined_content = combined_trainer.combine_novel_contents(novels, "concatenate")
            assert "Test Novel 1" in combined_content
            assert "Test Novel 2" in combined_content
            assert self.mock_contents["test_novel_1"] in combined_content
            assert self.mock_contents["test_novel_2"] in combined_content

            # Test training (mocked)
            with patch.object(combined_trainer, '_create_combined_memories'):
                result = combined_trainer.train_combined_model("test_model")

                assert result["model_type"] == "combined"
                assert result["model_key"] == "test_model"
                assert "novels_included" in result
                assert len(result["novels_included"]) == 2

    @patch('relational_memory_mapper.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_combined_to_relational_workflow(self, mock_json_load, mock_open, mock_path):
        """Test workflow from combined model to relational mapping"""
        # Setup mocks
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        # Mock file reading for novel contents
        def mock_file_read(file_path, *args, **kwargs):
            mock_file = Mock()
            if "test_novel_1" in str(file_path):
                mock_file.read.return_value = self.mock_contents["test_novel_1"]
            elif "test_novel_2" in str(file_path):
                mock_file.read.return_value = self.mock_contents["test_novel_2"]
            else:
                mock_file.read.return_value = ""
            return mock_file

        mock_open.side_effect = mock_file_read

        # Test relational mapping workflow
        with patch.object(RelationalMemoryMapper, '_load_model_mapping', return_value=self.mock_mapping):
            mapper = RelationalMemoryMapper(RelationalConfig())

            # Test thematic analysis
            novels = mapper.get_available_models()
            assert "test_model" in novels

            # Test building relational memory table
            with patch.object(mapper, 'save_relational_mappings', return_value="test_output.json"):
                relational_data = mapper.build_relational_memory_table("test_model")

                assert "model_key" in relational_data
                assert relational_data["model_key"] == "test_model"
                assert "novel_count" in relational_data
                assert relational_data["novel_count"] == 2

                # Should contain analysis results
                if mapper.config.analyze_themes:
                    assert "thematic_analysis" in relational_data
                if mapper.config.analyze_characters:
                    assert "character_analysis" in relational_data
                if mapper.config.analyze_narrative_patterns:
                    assert "narrative_analysis" in relational_data

    @patch('master_training_pipeline.subprocess.run')
    @patch('master_training_pipeline.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_complete_pipeline_workflow(self, mock_json_load, mock_open, mock_path, mock_subprocess):
        """Test complete pipeline workflow from start to finish"""
        # Setup mocks
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        # Mock successful subprocess runs
        mock_subprocess.return_value = Mock(returncode=0, stdout="Success", stderr="")

        # Test complete pipeline
        pipeline = MasterTrainingPipeline(PipelineConfig())

        # Mock all internal checks
        with patch.object(pipeline, 'check_individual_novels_status') as mock_check_individual:
            with patch.object(pipeline, 'check_combined_model_status') as mock_check_combined:
                with patch.object(pipeline, 'check_relational_mappings_status') as mock_check_mappings:
                    with patch.object(pipeline, '_validate_combined_model') as mock_validate_model:
                        with patch.object(pipeline, '_validate_memory_system') as mock_validate_memory:
                            with patch.object(pipeline, '_validate_integration') as mock_validate_integration:
                                with patch.object(pipeline, '_save_pipeline_results'):

                                    # Setup mock returns
                                    mock_check_individual.return_value = {
                                        "total_novels": 2,
                                        "trained_novels": 0,
                                        "untrained_novels": 2
                                    }
                                    mock_check_combined.return_value = {"model_exists": False}
                                    mock_check_mappings.return_value = {"mappings_exist": False}
                                    mock_validate_model.return_value = {"status": "passed"}
                                    mock_validate_memory.return_value = {"status": "passed"}
                                    mock_validate_integration.return_value = {"status": "passed"}

                                    # Execute pipeline
                                    result = pipeline.execute_full_pipeline("test_model")

                                    # Verify pipeline executed all stages
                                    assert result["model_key"] == "test_model"
                                    assert "stages" in result
                                    assert "total_duration" in result

                                    # Check that all expected stages were executed
                                    stages = result["stages"]
                                    expected_stages = ["stage_1", "stage_2", "stage_3", "stage_4"]
                                    for stage in expected_stages:
                                        assert stage in stages

    def test_error_propagation_workflow(self):
        """Test that errors propagate properly through the workflow"""
        # Test with invalid model mapping
        with pytest.raises(FileNotFoundError):
            with patch('pathlib.Path.exists', return_value=False):
                CombinedModelTrainer(CombinedModelConfig())

        # Test with invalid model key
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(CombinedModelConfig())
            with pytest.raises(ValueError):
                trainer.get_novels_for_model("non_existent_model")

    @patch('builtins.open')
    @patch('json.load')
    @patch('pathlib.Path.exists', return_value=True)
    def test_configuration_persistence_workflow(self, mock_path, mock_json_load, mock_open):
        """Test that configuration persists through workflow steps"""
        mock_json_load.return_value = self.mock_mapping

        # Create pipeline with custom config
        config = PipelineConfig()
        config.max_iterations = 8  # Custom value
        config.force_retrain_combined = True

        pipeline = MasterTrainingPipeline(config)

        # Verify configuration is maintained
        assert pipeline.config.max_iterations == 8
        assert pipeline.config.force_retrain_combined == True

        # Test that configuration affects behavior
        assert pipeline.config.run_individual_training == True
        assert pipeline.config.run_combined_training == True


class TestDataFlowIntegration:
    """Test data flow between components"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.test_mapping = {
            "models": {
                "flow_test": {
                    "novels": [
                        {"original_name": "Flow Novel 1", "directory_name": "flow_novel_1"},
                        {"original_name": "Flow Novel 2", "directory_name": "flow_novel_2"}
                    ]
                }
            }
        }

    @patch('combined_model_trainer.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_novel_list_consistency(self, mock_json_load, mock_open, mock_path):
        """Test that novel lists are consistent across components"""
        mock_json_load.return_value = self.test_mapping
        mock_path.return_value.exists.return_value = True

        # Test that all components see the same novels
        combined_trainer = CombinedModelTrainer(CombinedModelConfig())
        mapper = RelationalMemoryMapper(RelationalConfig())
        pipeline = MasterTrainingPipeline(PipelineConfig())

        # Get novels from each component
        combined_novels = combined_trainer.get_novels_for_model("flow_test")
        pipeline_novels = pipeline.get_novels_for_model("flow_test")

        # Should be identical
        assert len(combined_novels) == len(pipeline_novels)
        for i, novel in enumerate(combined_novels):
            assert novel["original_name"] == pipeline_novels[i]["original_name"]
            assert novel["directory_name"] == pipeline_novels[i]["directory_name"]

    @patch('relational_memory_mapper.json.dump')
    @patch('builtins.open')
    @patch('json.load')
    @patch('pathlib.Path.exists', return_value=True)
    def test_output_format_consistency(self, mock_path, mock_json_load, mock_open, mock_json_dump):
        """Test that output formats are consistent between components"""
        mock_json_load.return_value = self.test_mapping

        # Test that all components produce compatible output formats
        combined_trainer = CombinedModelTrainer(CombinedModelConfig())
        mapper = RelationalMemoryMapper(RelationalConfig())

        # Mock training result
        training_result = {
            "model_key": "flow_test",
            "novels_included": ["Flow Novel 1", "Flow Novel 2"],
            "training_time": 600,
            "final_quality": 0.9
        }

        # Mock relational mapping result
        with patch.object(mapper, '_load_novel_content', return_value="test content"):
            relational_result = mapper.build_relational_memory_table("flow_test")

            # Both should have compatible model_key fields
            assert "model_key" in relational_result
            assert relational_result["model_key"] == "flow_test"

            # Both should reference the same number of novels
            assert relational_result["novel_count"] == len(training_result["novels_included"])


class TestMemorySystemIntegration:
    """Test memory system integration across components"""

    def test_memory_system_availability_check(self):
        """Test memory system availability checking"""
        from model_tea_utils import MemorySystemUtils

        # This should not raise an exception
        available = MemorySystemUtils.check_memory_system_available()
        assert isinstance(available, bool)

        # If not available, should provide fallback
        if not available:
            fallback = MemorySystemUtils.create_memory_fallback()
            assert isinstance(fallback, dict)
            assert fallback["system_status"] == "unavailable"

    @patch('combined_model_trainer.MEMORY_SYSTEM_AVAILABLE', True)
    @patch('combined_model_trainer.EpisodicMemorySystem')
    def test_memory_system_integration_mock(self, mock_memory_system):
        """Test memory system integration with mocking"""
        # Mock memory system
        mock_memory_instance = Mock()
        mock_memory_system.return_value = mock_memory_instance

        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value={"models": {}}):
            trainer = CombinedModelTrainer(CombinedModelConfig())

            # Test memory creation method exists and can be called
            test_content = "Test novel content for memory creation"
            test_novels = [{"original_name": "Test", "directory_name": "test"}]
            test_results = {"training_time": 600}

            # This should not raise an exception
            try:
                trainer._create_combined_memories("test_model", test_content, test_novels, test_results)
            except Exception as e:
                # Memory creation might fail due to mocking, but the method should exist
                assert "create_memories_from_content" in str(e) or "save_to_directory" in str(e)


if __name__ == "__main__":
    pytest.main([__file__])