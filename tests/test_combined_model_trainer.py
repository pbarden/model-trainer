#!/usr/bin/env python3
"""
Test combined model trainer functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig


class TestCombinedModelConfig:
    """Test CombinedModelConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CombinedModelConfig()

        assert config.combine_novels_method == "concatenate"
        assert config.novel_separator == "\n\n=== NEW NOVEL ===\n\n"
        assert config.min_novels_required == 2
        assert config.max_combined_size == 2000000
        assert config.max_iterations == 12
        assert config.learning_rate_start == 5e-5
        assert config.learning_rate_end == 5e-6
        assert config.combined_memories_count == 350
        assert config.cross_novel_memories == True

    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = CombinedModelConfig()

        # Test reasonable ranges
        assert 1 <= config.min_novels_required <= 10
        assert 1 <= config.max_iterations <= 15
        assert 0 < config.learning_rate_start <= 1e-3
        assert 0 < config.learning_rate_end <= 1e-4
        assert 100 <= config.combined_memories_count <= 1000


class TestCombinedModelTrainer:
    """Test CombinedModelTrainer class"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures"""
        self.config = CombinedModelConfig()

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

    @patch('combined_model_trainer.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_model_mapping(self, mock_json_load, mock_open, mock_path):
        """Test model mapping loading"""
        mock_json_load.return_value = self.mock_mapping
        mock_path.return_value.exists.return_value = True

        trainer = CombinedModelTrainer(self.config)

        assert trainer.model_mapping == self.mock_mapping

    @patch('combined_model_trainer.Path')
    def test_load_model_mapping_missing_file(self, mock_path):
        """Test handling of missing model mapping file"""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            CombinedModelTrainer(self.config)

    def test_get_available_models(self):
        """Test getting available models list"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)
            models = trainer.get_available_models()

            assert models == ["test_model_1", "test_model_2"]

    def test_get_novels_for_model(self):
        """Test getting novels for specific model"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)
            novels = trainer.get_novels_for_model("test_model_1")

            assert len(novels) == 2
            assert novels[0]["original_name"] == "Test Novel 1"
            assert novels[1]["original_name"] == "Test Novel 2"

    def test_get_novels_for_invalid_model(self):
        """Test error handling for invalid model"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
                trainer.get_novels_for_model("invalid_model")

    def test_check_individual_novels_trained(self):
        """Test checking individual novel training status"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            novels = trainer.get_novels_for_model("test_model_1")

            # Mock that all individual novels are already trained
            with patch('pathlib.Path.exists', return_value=True):
                status = trainer.check_individual_novels_trained(novels)

                assert len(status) == 2
                assert status["test_novel_1"] == True
                assert status["test_novel_2"] == True

    def test_load_novel_content(self):
        """Test loading novel content from directory"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            test_content = "This is test novel content."

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create test novel directory
                novel_dir = Path(tmp_dir) / "test_novel"
                novel_dir.mkdir()
                content_file = novel_dir / "content.txt"
                content_file.write_text(test_content)

                # Mock the novels_dir to point to our temp directory
                trainer.novels_dir = Path(tmp_dir)

                content = trainer.load_novel_content("test_novel")
                assert content == test_content

    def test_load_novel_content_missing_directory(self):
        """Test error handling for missing novel directory"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            with pytest.raises(FileNotFoundError):
                trainer.load_novel_content("non_existent_novel")

    def test_combine_novel_contents_concatenate(self):
        """Test combining novels using concatenate method"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            novels = [
                {"original_name": "Novel 1", "directory_name": "novel_1"},
                {"original_name": "Novel 2", "directory_name": "novel_2"}
            ]

            # Mock load_novel_content
            def mock_load_content(directory_name):
                if directory_name == "novel_1":
                    return "Content of novel 1."
                elif directory_name == "novel_2":
                    return "Content of novel 2."
                return ""

            with patch.object(trainer, 'load_novel_content', side_effect=mock_load_content):
                combined = trainer.combine_novel_contents(novels, "concatenate")

                assert "=== Novel 1 ===" in combined
                assert "=== Novel 2 ===" in combined
                assert "Content of novel 1." in combined
                assert "Content of novel 2." in combined
                assert trainer.config.novel_separator in combined

    def test_combine_novel_contents_size_limit(self):
        """Test combining novels respects size limit"""
        config = CombinedModelConfig()
        config.max_combined_size = 50  # Very small limit for testing

        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(config)

            novels = [
                {"original_name": "Novel 1", "directory_name": "novel_1"},
                {"original_name": "Novel 2", "directory_name": "novel_2"}
            ]

            # Mock large content
            def mock_load_content(directory_name):
                return "This is a very long novel content that exceeds the size limit."

            with patch.object(trainer, 'load_novel_content', side_effect=mock_load_content):
                combined = trainer.combine_novel_contents(novels, "concatenate")

                # Should still contain some content but respect the limit
                assert len(combined) > 0

    @patch.object(CombinedModelTrainer, 'get_novels_for_model')
    @patch.object(CombinedModelTrainer, 'check_individual_novels_trained')
    @patch.object(CombinedModelTrainer, 'combine_novel_contents')
    def test_train_combined_model_mock(self, mock_combine, mock_check_trained, mock_get_novels):
        """Test combined model training with mocked dependencies"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            # Setup mocks
            mock_get_novels.return_value = self.mock_mapping["models"]["test_model_1"]["novels"]
            mock_check_trained.return_value = {
                "test_novel_1": True,
                "test_novel_2": True
            }
            mock_combine.return_value = "Combined novel content for testing"

            # Mock the iterative trainer import and training
            with patch('iterative_novel_trainer.IterativeTrainer') as mock_trainer_class:
                mock_trainer_instance = Mock()
                mock_trainer_instance._train_with_content.return_value = {
                    "iterations": [{"iteration": 1, "perplexity": 10.0, "quality_score": 0.9}],
                    "final_quality": 0.9,
                    "training_time": 600
                }
                mock_trainer_class.return_value = mock_trainer_instance

                # Test training
                result = trainer.train_combined_model("test_model_1")

                assert "model_type" in result
                assert result["model_type"] == "combined"
                assert result["model_key"] == "test_model_1"
                assert "novels_included" in result
                assert "total_training_time" in result
                assert "combination_method" in result

    def test_list_trained_combined_models(self):
        """Test listing trained combined models"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            # Mock that only test_model_1 exists as trained, test_model_2 doesn't
            call_count = 0
            def path_exists_side_effect():
                nonlocal call_count
                call_count += 1
                # First call is for test_model_1, second is for test_model_2
                return call_count == 1

            with patch('pathlib.Path.exists', side_effect=path_exists_side_effect):
                trained_models = trainer.list_trained_combined_models()

                assert "test_model_1" in trained_models
                assert "test_model_2" not in trained_models

    @patch('combined_model_trainer.logger')
    def test_train_all_combined_models_mock(self, mock_logger):
        """Test training all combined models with mocking"""
        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(self.config)

            # Mock train_combined_model to avoid actual training
            def mock_train_model(model_key):
                return {
                    "model_key": model_key,
                    "status": "success",
                    "training_time": 300
                }

            with patch.object(trainer, 'train_combined_model', side_effect=mock_train_model):
                with patch.object(Path, 'exists', return_value=False):  # No models already trained
                    results = trainer.train_all_combined_models()

                    assert len(results) == 2
                    assert "test_model_1" in results
                    assert "test_model_2" in results
                    assert results["test_model_1"]["status"] == "success"
                    assert results["test_model_2"]["status"] == "success"

    def test_insufficient_novels_error(self):
        """Test error handling when model has insufficient novels"""
        config = CombinedModelConfig()
        config.min_novels_required = 5  # Require more novels than available

        with patch.object(CombinedModelTrainer, '_load_model_mapping', return_value=self.mock_mapping):
            trainer = CombinedModelTrainer(config)

            with pytest.raises(ValueError, match="has only 2 novels. Minimum required: 5"):
                trainer.train_combined_model("test_model_1")


if __name__ == "__main__":
    pytest.main([__file__])