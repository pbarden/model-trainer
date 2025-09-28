#!/usr/bin/env python3
"""
Test master training pipeline functionality
"""

import pytest
import tempfile
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from model_tea.core.training.master_pipeline import MasterTrainingPipeline, PipelineConfig, PipelineStage


class TestPipelineConfig:
    """Test PipelineConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = PipelineConfig()

        assert config.run_individual_training == True
        assert config.run_combined_training == True
        assert config.run_relational_mapping == True
        assert config.run_validation == True
        assert config.generate_report == True
        assert config.force_retrain_individual == False
        assert config.force_retrain_combined == False
        assert config.stop_on_error == False
        assert config.save_intermediate_results == True
        assert config.verbose_logging == True
        assert config.results_directory == "pipeline_results"

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = PipelineConfig()

        # Test reasonable ranges
        assert 1 <= config.max_parallel_novels <= 10
        assert isinstance(config.individual_training_parallel, bool)
        assert isinstance(config.skip_if_exists, bool)


class TestPipelineStage:
    """Test PipelineStage enum"""

    def test_pipeline_stages(self):
        """Test pipeline stage enumeration"""
        assert PipelineStage.INDIVIDUAL_TRAINING.value == "individual_training"
        assert PipelineStage.COMBINED_TRAINING.value == "combined_training"
        assert PipelineStage.RELATIONAL_MAPPING.value == "relational_mapping"
        assert PipelineStage.VALIDATION.value == "validation"
        assert PipelineStage.REPORTING.value == "reporting"


class TestMasterTrainingPipeline:
    """Test MasterTrainingPipeline class"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.config = PipelineConfig()

        # Mock model mapping data
        self.mock_mapping = {
            "metadata": {
                "total_models": 2,
                "total_novels_assigned": 4
            },
            "models": {
                "test_model_1": {
                    "description": "Test Model 1",
                    "novel_count": 2,
                    "novels": [
                        {"original_name": "Test Novel 1", "directory_name": "test_novel_1"},
                        {"original_name": "Test Novel 2", "directory_name": "test_novel_2"}
                    ]
                },
                "test_model_2": {
                    "description": "Test Model 2",
                    "novel_count": 2,
                    "novels": [
                        {"original_name": "Test Novel 3", "directory_name": "test_novel_3"},
                        {"original_name": "Test Novel 4", "directory_name": "test_novel_4"}
                    ]
                }
            }
        }

    @patch('master_training_pipeline.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_model_mapping(self, mock_json_load, mock_open, mock_path):
        """Test model mapping loading"""
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        pipeline = MasterTrainingPipeline(self.config)

        assert pipeline.model_mapping == self.mock_mapping

    def test_get_available_models(self):
        """Test getting available models list"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)
            models = pipeline.get_available_models()

            assert models == ["test_model_1", "test_model_2"]

    def test_get_novels_for_model(self):
        """Test getting novels for specific model"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)
            novels = pipeline.get_novels_for_model("test_model_1")

            assert len(novels) == 2
            assert novels[0]["original_name"] == "Test Novel 1"
            assert novels[1]["original_name"] == "Test Novel 2"

    def test_get_novels_for_invalid_model(self):
        """Test error handling for invalid model"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
                pipeline.get_novels_for_model("invalid_model")

    def test_check_individual_novels_status(self):
        """Test checking individual novel training status"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock that all individual novels are already trained
            with patch('pathlib.Path.exists', return_value=True):
                status = pipeline.check_individual_novels_status("test_model_1")

                assert status["total_novels"] == 2
                assert status["trained_novels"] == 2
                assert status["untrained_novels"] == 0
                assert status["novel_status"]["test_novel_1"]["trained"] == True
                assert status["novel_status"]["test_novel_2"]["trained"] == True

    def test_check_combined_model_status(self):
        """Test checking combined model status"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            mock_results = {
                "training_time": 600,
                "final_quality": 0.9,
                "model_type": "combined"
            }

            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open=True):
                    with patch('json.load', return_value=mock_results):
                        status = pipeline.check_combined_model_status("test_model_1")

                        assert status["model_exists"] == True
                        assert status["results_exist"] == True
                        assert status["training_completed"] == True
                        assert status["training_time"] == 600
                        assert status["final_quality"] == 0.9

    def test_check_relational_mappings_status(self):
        """Test checking relational mappings status"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            mock_mappings = {
                "novel_count": 2,
                "cross_references": {
                    "novel_1": [{"type": "thematic", "target": "novel_2"}],
                    "novel_2": [{"type": "character", "target": "novel_1"}]
                }
            }

            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open=True):
                    with patch('json.load', return_value=mock_mappings):
                        status = pipeline.check_relational_mappings_status("test_model_1")

                        assert status["mappings_exist"] == True
                        assert status["mapping_completed"] == True
                        assert status["novel_count"] == 2
                        assert status["cross_references"] == 2

    @patch('master_training_pipeline.subprocess.run')
    def test_stage_1_individual_training_mock(self, mock_subprocess):
        """Test stage 1 individual training with mocked subprocess"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock successful subprocess run
            mock_subprocess.return_value = Mock(returncode=0, stdout="Training completed", stderr="")

            # Mock individual novel status check
            mock_status = {
                "total_novels": 2,
                "trained_novels": 0,
                "untrained_novels": 2,
                "novel_status": {
                    "test_novel_1": {"trained": False},
                    "test_novel_2": {"trained": False}
                }
            }

            with patch.object(pipeline, 'check_individual_novels_status', return_value=mock_status):
                with patch.object(pipeline, 'get_novels_for_model', return_value=self.mock_mapping["models"]["test_model_1"]["novels"]):
                    result = pipeline.stage_1_individual_training("test_model_1")

                    assert result["stage"] == "individual_training"
                    assert result["model_key"] == "test_model_1"
                    assert "start_time" in result
                    assert "duration" in result

    @patch('master_training_pipeline.subprocess.run')
    def test_stage_2_combined_training_mock(self, mock_subprocess):
        """Test stage 2 combined training with mocked subprocess"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock successful subprocess run
            mock_subprocess.return_value = Mock(returncode=0, stdout="Combined training completed", stderr="")

            # Mock combined model status (not exists)
            mock_status = {"model_exists": False, "results_exist": False}

            with patch.object(pipeline, 'check_combined_model_status', return_value=mock_status):
                result = pipeline.stage_2_combined_training("test_model_1")

                assert result["stage"] == "combined_training"
                assert result["model_key"] == "test_model_1"
                assert result["status"] == "completed"
                assert "duration" in result

    @patch('master_training_pipeline.subprocess.run')
    def test_stage_3_relational_mapping_mock(self, mock_subprocess):
        """Test stage 3 relational mapping with mocked subprocess"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock successful subprocess run
            mock_subprocess.return_value = Mock(returncode=0, stdout="Relational mapping completed", stderr="")

            # Mock mappings status (not exists)
            mock_status = {"mappings_exist": False, "mapping_completed": False}

            with patch.object(pipeline, 'check_relational_mappings_status', return_value=mock_status):
                result = pipeline.stage_3_relational_mapping("test_model_1")

                assert result["stage"] == "relational_mapping"
                assert result["model_key"] == "test_model_1"
                assert result["status"] == "completed"
                assert "duration" in result

    def test_stage_4_validation_testing(self):
        """Test stage 4 validation testing"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock validation methods
            with patch.object(pipeline, '_validate_combined_model', return_value={"status": "passed"}):
                with patch.object(pipeline, '_validate_memory_system', return_value={"status": "passed"}):
                    with patch.object(pipeline, '_validate_integration', return_value={"status": "passed"}):
                        result = pipeline.stage_4_validation_testing("test_model_1")

                        assert result["stage"] == "validation"
                        assert result["model_key"] == "test_model_1"
                        assert result["status"] == "passed"
                        assert "validation_results" in result
                        assert "duration" in result

    @patch('master_training_pipeline.Path')
    def test_validate_combined_model(self, mock_path):
        """Test combined model validation"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock model directory exists
            def path_exists_side_effect(path_obj):
                return True

            # Mock required files exist
            def path_file_exists_side_effect(path_obj):
                filename = str(path_obj).split('/')[-1]
                return filename in ["config.json", "pytorch_model.bin", "tokenizer.json"]

            with patch.object(Path, 'exists', side_effect=path_exists_side_effect):
                with patch.object(Path, '__truediv__') as mock_div:
                    mock_file_path = Mock()
                    mock_file_path.exists.side_effect = path_file_exists_side_effect
                    mock_div.return_value = mock_file_path

                    result = pipeline._validate_combined_model("test_model_1")

                    assert result["status"] == "passed"
                    assert "model_path" in result

    @patch('master_training_pipeline.Path')
    def test_validate_memory_system(self, mock_path):
        """Test memory system validation"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock paths exist
            def path_exists_side_effect(path_obj):
                return True

            with patch.object(Path, 'exists', side_effect=path_exists_side_effect):
                result = pipeline._validate_memory_system("test_model_1")

                assert "status" in result
                assert "checks" in result
                assert len(result["checks"]) > 0

    def test_validate_integration(self):
        """Test integration validation"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            mock_results = {
                "novels_included": ["Novel 1", "Novel 2"]
            }

            with patch.object(pipeline, 'get_novels_for_model', return_value=self.mock_mapping["models"]["test_model_1"]["novels"]):
                with patch('builtins.open', mock_open=True):
                    with patch('json.load', return_value=mock_results):
                        with patch.object(Path, 'exists', return_value=True):
                            result = pipeline._validate_integration("test_model_1")

                            assert "status" in result
                            assert "expected_novels" in result
                            assert result["expected_novels"] == 2

    def test_execute_full_pipeline_mock(self):
        """Test full pipeline execution with mocking"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock all stage methods
            stage_1_result = {"stage": "individual_training", "status": "completed", "duration": 300}
            stage_2_result = {"stage": "combined_training", "status": "completed", "duration": 600}
            stage_3_result = {"stage": "relational_mapping", "status": "completed", "duration": 180}
            stage_4_result = {"stage": "validation", "status": "passed", "duration": 60}

            with patch.object(pipeline, 'stage_1_individual_training', return_value=stage_1_result):
                with patch.object(pipeline, 'stage_2_combined_training', return_value=stage_2_result):
                    with patch.object(pipeline, 'stage_3_relational_mapping', return_value=stage_3_result):
                        with patch.object(pipeline, 'stage_4_validation_testing', return_value=stage_4_result):
                            with patch.object(pipeline, '_save_pipeline_results'):
                                result = pipeline.execute_full_pipeline("test_model_1")

                                assert result["model_key"] == "test_model_1"
                                assert result["overall_status"] == "success"
                                assert "stages" in result
                                assert "total_duration" in result
                                assert len(result["stages"]) == 4

    def test_execute_all_models_pipeline_mock(self):
        """Test executing pipeline for all models"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            # Mock successful pipeline execution for each model
            def mock_execute_pipeline(model_key):
                return {
                    "model_key": model_key,
                    "overall_status": "success",
                    "total_duration": 1200
                }

            with patch.object(pipeline, 'execute_full_pipeline', side_effect=mock_execute_pipeline):
                with patch.object(pipeline, '_save_summary_report'):
                    results = pipeline.execute_all_models_pipeline()

                    assert "models" in results
                    assert "summary" in results
                    assert len(results["models"]) == 2
                    assert "test_model_1" in results["models"]
                    assert "test_model_2" in results["models"]

                    # Check summary
                    summary = results["summary"]
                    assert summary["total_models"] == 2
                    assert summary["successful"] == 2
                    assert summary["failed"] == 0

    def test_generate_summary(self):
        """Test summary generation"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            model_results = {
                "model_1": {"overall_status": "success"},
                "model_2": {"overall_status": "failed"},
                "model_3": {"overall_status": "partial"},
                "model_4": {"overall_status": "error"}
            }

            summary = pipeline._generate_summary(model_results)

            assert summary["total_models"] == 4
            assert summary["successful"] == 1
            assert summary["failed"] == 1
            assert summary["partial"] == 1
            assert summary["errors"] == 1

    @patch('master_training_pipeline.json.dump')
    @patch('builtins.open')
    def test_save_pipeline_results(self, mock_open, mock_json_dump):
        """Test saving pipeline results"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            test_results = {"test": "data"}
            pipeline._save_pipeline_results("test_model", test_results)

            mock_open.assert_called_once()
            mock_json_dump.assert_called_once()

    @patch('master_training_pipeline.json.dump')
    @patch('builtins.open')
    def test_save_summary_report(self, mock_open, mock_json_dump):
        """Test saving summary report"""
        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(self.config)

            test_results = {"summary": "data"}
            pipeline._save_summary_report(test_results)

            mock_open.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_stage_skip_behavior(self):
        """Test stage skipping behavior"""
        config = PipelineConfig()
        config.run_individual_training = False
        config.run_combined_training = False

        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(config)

            with patch.object(pipeline, '_save_pipeline_results'):
                result = pipeline.execute_full_pipeline("test_model_1")

                # Should only run relational mapping and validation
                assert len(result["stages"]) == 2
                assert "stage_3" in result["stages"]  # relational mapping
                assert "stage_4" in result["stages"]  # validation

    def test_error_handling_stop_on_error(self):
        """Test error handling with stop_on_error enabled"""
        config = PipelineConfig()
        config.stop_on_error = True

        with patch.object(MasterTrainingPipeline, '_load_model_mapping', return_value=self.mock_mapping):
            pipeline = MasterTrainingPipeline(config)

            # Mock stage 1 failure
            stage_1_result = {"stage": "individual_training", "status": "error", "error": "Test error"}

            with patch.object(pipeline, 'stage_1_individual_training', return_value=stage_1_result):
                with patch.object(pipeline, '_save_pipeline_results'):
                    result = pipeline.execute_full_pipeline("test_model_1")

                    assert result["overall_status"] == "failed"
                    assert result["failed_at_stage"] == "individual_training"
                    assert len(result["stages"]) == 1  # Should stop after first stage


if __name__ == "__main__":
    pytest.main([__file__])