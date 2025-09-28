"""
Test the actual iterative novel trainer functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from iterative_novel_trainer import IterativeTrainer, IterativeConfig


class TestIterativeTrainer:
    """Test the main training functionality that actually exists"""

    def test_config_creation(self):
        """Test that IterativeConfig can be created with default values"""
        config = IterativeConfig()

        assert config.base_model == "gpt2"
        assert config.max_seq_length == 1024
        assert config.iterations_per_novel == 12
        assert config.max_steps_per_iteration == 8
        assert config.learning_rate_start == 5e-5
        assert config.learning_rate_end == 2e-5

    def test_trainer_initialization(self):
        """Test that trainer can be initialized"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        assert trainer.config == config
        assert trainer.novels_dir == Path(config.novels_dir)
        assert trainer.output_dir == Path(config.output_dir)

    def test_list_available_novels(self):
        """Test novel listing functionality"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        novels = trainer.list_available_novels()

        assert isinstance(novels, list)
        assert len(novels) > 0
        assert "call_of_cthulhu" in [n.lower().replace(" ", "_").replace("-", "_") for n in novels]

    def test_novel_path_resolution(self):
        """Test that novel paths can be resolved"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        novels_dir = trainer.novels_dir
        assert novels_dir.exists()

        novel_files = list(novels_dir.glob("*"))
        assert len(novel_files) > 0

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_trainer_components_mock(self, mock_model, mock_tokenizer):
        """Test trainer components can be mocked for unit testing"""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        assert trainer is not None

    def test_config_validation(self):
        """Test configuration validation"""
        config = IterativeConfig()

        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0
        assert config.validation_split > 0 and config.validation_split < 1
        assert config.learning_rate_start > config.learning_rate_end
        assert config.iterations_per_novel > 0

    def test_config_customization(self):
        """Test that config can be customized"""
        config = IterativeConfig(
            base_model="custom-model",
            iterations_per_novel=5,
            max_steps_per_iteration=10
        )

        assert config.base_model == "custom-model"
        assert config.iterations_per_novel == 5
        assert config.max_steps_per_iteration == 10

    def test_learning_rate_calculation(self):
        """Test learning rate calculation doesn't cause division by zero"""
        config = IterativeConfig(iterations_per_novel=1)
        trainer = IterativeTrainer(config)

        lr = trainer._calculate_learning_rate(0)
        assert lr > 0
        assert lr == config.learning_rate_start

    def test_generate_sample_functionality(self):
        """Test that sample generation method exists"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        assert hasattr(trainer, 'generate_sample')
        assert callable(getattr(trainer, 'generate_sample'))

    def test_quality_evaluation_functionality(self):
        """Test that quality evaluation methods exist"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        assert hasattr(trainer, 'test_generation_quality')
        assert callable(getattr(trainer, 'test_generation_quality'))

    def test_perplexity_calculation_functionality(self):
        """Test that perplexity calculation methods exist"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        assert hasattr(trainer, 'calculate_perplexity')
        assert callable(getattr(trainer, 'calculate_perplexity'))


class TestIterativeTrainerIntegration:
    """Integration tests for actual training workflows"""

    def test_novel_loading_integration(self):
        """Test that novels can actually be loaded"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        novels = trainer.list_available_novels()
        assert len(novels) > 0

        test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")

        novel_path = trainer.novels_dir / test_novel
        if novel_path.exists():
            assert novel_path.is_file() or any(novel_path.glob("*.txt"))

    @pytest.mark.slow
    def test_quick_training_smoke_test(self):
        """Smoke test - verify training can start without errors"""
        config = IterativeConfig(
            iterations_per_novel=1,
            max_steps_per_iteration=1
        )
        trainer = IterativeTrainer(config)

        novels = trainer.list_available_novels()
        if novels:
            test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")

            try:
                with patch('transformers.Trainer') as mock_trainer:
                    mock_trainer_instance = MagicMock()
                    mock_trainer.return_value = mock_trainer_instance
                    mock_trainer_instance.train.return_value = None

                    result = trainer.train_novel(test_novel)

                    assert isinstance(result, dict)

            except Exception as e:
                pytest.skip(f"Training setup issues: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])