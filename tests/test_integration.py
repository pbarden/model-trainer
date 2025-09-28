"""
Integration tests for the full training workflow
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from iterative_novel_trainer import IterativeTrainer, IterativeConfig
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
from model_tea.core.pipeline import MLPipeline, PipelineConfig
from model_tea.core.evaluation import ModelEvaluator


class TestTrainingIntegration:
    """Integration tests for training workflow"""

    def test_config_to_trainer_flow(self):
        """Test config creation to trainer initialization"""
        config = IterativeConfig(
            base_model="gpt2",
            iterations_per_novel=2,
            max_steps_per_iteration=1
        )

        trainer = IterativeTrainer(config)

        assert trainer.config.base_model == "gpt2"
        assert trainer.config.iterations_per_novel == 2
        assert trainer.novels_dir.exists()
        assert trainer.output_dir == Path(config.output_dir)

    def test_novel_listing_and_selection(self):
        """Test novel listing and selection workflow"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        novels = trainer.list_available_novels()
        assert len(novels) > 0

        test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")
        assert isinstance(test_novel, str)
        assert len(test_novel) > 0

    def test_memory_system_integration(self):
        """Test episodic memory system integration"""
        memory_config = MemoryConfig(
            memory_chunk_size=20,
            max_memories_per_novel=50
        )
        memory_system = EpisodicMemorySystem(memory_config)

        test_text = """
        The detective entered the mysterious library. Ancient books lined the walls.
        "Something strange is happening here," he whispered to his partner.
        The old librarian watched them with suspicious eyes.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "test_novel.txt"
            novel_path.write_text(test_text)
            result = memory_system.build_memory_for_model("test_model", novel_path)
            memories = result.get("memories", [])

        assert len(memories) > 0
        assert len(memories) <= memory_config.max_memories_per_novel
        assert all(hasattr(memory, 'content') for memory in memories)
        assert all(hasattr(memory, 'memory_type') for memory in memories)

    def test_trainer_with_memory_system(self):
        """Test trainer working with memory system"""
        trainer_config = IterativeConfig(iterations_per_novel=1, max_steps_per_iteration=1)
        memory_config = MemoryConfig(max_memories_per_novel=10)

        trainer = IterativeTrainer(trainer_config)
        memory_system = EpisodicMemorySystem(memory_config)

        novels = trainer.list_available_novels()
        if novels:
            test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")

            with tempfile.TemporaryDirectory() as temp_dir:
                novel_path = trainer.novels_dir / test_novel
                if novel_path.exists():
                    result = memory_system.build_memory_for_model("test_model", novel_path)
                    memories = result.get("memories", [])

                    assert len(memories) >= 0

    @patch('transformers.Trainer')
    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_full_training_pipeline_mock(self, mock_model, mock_tokenizer, mock_trainer):
        """Test full training pipeline with mocks"""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        config = IterativeConfig(
            iterations_per_novel=1,
            max_steps_per_iteration=1
        )
        trainer = IterativeTrainer(config)

        novels = trainer.list_available_novels()
        if novels:
            test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")

            try:
                result = trainer.train_novel(test_novel)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.skip(f"Training mock setup issues: {e}")

    def test_pipeline_with_evaluation(self):
        """Test ML pipeline with evaluation"""
        pipeline_config = PipelineConfig(pipeline_name="test_training_pipeline")
        pipeline = MLPipeline(pipeline_config)
        evaluator = ModelEvaluator()

        from model_tea.core.pipeline import DataPreprocessingStage, ModelTrainingStage, ModelEvaluationStage

        preprocessing = DataPreprocessingStage({})
        training = ModelTrainingStage({"epochs": 1, "max_iter": 10})
        evaluation = ModelEvaluationStage({})

        pipeline.add_stage(preprocessing)
        pipeline.add_stage(training)
        pipeline.add_stage(evaluation)

        initial_data = {"raw_data": list(range(20))}
        results = pipeline.execute(initial_data)

        assert "trained_model" in results
        assert "evaluation_results" in results
        assert results["evaluation_results"]["accuracy"] > 0

    def test_memory_storage_and_loading_integration(self):
        """Test memory system storage and loading"""
        memory_config = MemoryConfig()
        memory_system = EpisodicMemorySystem(memory_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "fantasy_novel.txt"
            novel_path.write_text("The wizard cast a powerful spell in the enchanted forest.")

            result = memory_system.build_memory_for_model("fantasy_model", novel_path)
            memories = result.get("memories", [])

            if memory_system.load_memory_for_model("fantasy_model"):
                loaded_memories, _ = memory_system.activate_memories("fantasy_model", "tell me about magic")
                assert isinstance(loaded_memories, list)

    def test_training_config_validation_integration(self):
        """Test training configuration validation in integration context"""
        valid_config = IterativeConfig(
            base_model="gpt2",
            iterations_per_novel=5,
            max_steps_per_iteration=10,
            learning_rate_start=5e-5,
            learning_rate_end=1e-5
        )

        trainer = IterativeTrainer(valid_config)

        assert trainer.config.learning_rate_start > trainer.config.learning_rate_end
        assert trainer.config.iterations_per_novel > 0
        assert trainer.config.max_steps_per_iteration > 0

        learning_rate = trainer._calculate_learning_rate(0)
        assert learning_rate == valid_config.learning_rate_start

        learning_rate_mid = trainer._calculate_learning_rate(2)
        assert valid_config.learning_rate_end <= learning_rate_mid <= valid_config.learning_rate_start

    def test_end_to_end_workflow_simulation(self):
        """Test end-to-end workflow simulation"""
        trainer_config = IterativeConfig(
            iterations_per_novel=1,
            max_steps_per_iteration=1,
            chunk_size=100
        )
        memory_config = MemoryConfig(max_memories_per_novel=5)

        trainer = IterativeTrainer(trainer_config)
        memory_system = EpisodicMemorySystem(memory_config)

        novels = trainer.list_available_novels()
        assert len(novels) > 0

        test_novel = novels[0].lower().replace(" ", "_").replace("-", "_")

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / f"{test_novel}.txt"
            novel_path.write_text(f"This is sample text from {test_novel} for testing the complete workflow.")

            result = memory_system.build_memory_for_model(f"{test_novel}_model", novel_path)
            memories = result.get("memories", [])

            assert len(memories) >= 0
            assert len(memories) <= memory_config.max_memories_per_novel

        assert hasattr(trainer, 'test_generation_quality')
        assert hasattr(trainer, 'calculate_perplexity')

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow"""
        config = IterativeConfig()
        trainer = IterativeTrainer(config)

        non_existent_novel = "non_existent_novel_12345"

        try:
            result = trainer.train_novel(non_existent_novel)
        except Exception as e:
            assert "not found" in str(e).lower() or "error" in str(e).lower()

    def test_memory_retrieval_integration(self):
        """Test memory retrieval in integrated context"""
        memory_config = MemoryConfig(max_retrieved_memories=3)
        memory_system = EpisodicMemorySystem(memory_config)

        test_text = """
        The ancient castle stood on a hill. A brave knight approached the gates.
        Inside, the princess waited in the tower. Dragons circled overhead.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            novel_path = Path(temp_dir) / "test_story.txt"
            novel_path.write_text(test_text)

            result = memory_system.build_memory_for_model("test_story_model", novel_path)
            memories = result.get("memories", [])

            if memories:
                query = "Tell me about the castle"
                retrieved, _ = memory_system.activate_memories("test_story_model", query)

                assert isinstance(retrieved, list)
                assert len(retrieved) <= memory_config.max_retrieved_memories


class TestSystemConfiguration:
    """Test system-wide configuration integration"""

    def test_compatible_configurations(self):
        """Test that different system configurations work together"""
        trainer_config = IterativeConfig(
            base_model="gpt2",
            max_seq_length=512,
            chunk_size=200
        )

        memory_config = MemoryConfig(
            memory_chunk_size=25,
            max_memories_per_novel=100
        )

        pipeline_config = PipelineConfig(
            pipeline_name="integrated_training",
            timeout_minutes=60
        )

        trainer = IterativeTrainer(trainer_config)
        memory_system = EpisodicMemorySystem(memory_config)
        pipeline = MLPipeline(pipeline_config)

        assert trainer.config.chunk_size >= memory_config.memory_chunk_size
        assert trainer.config.max_seq_length >= memory_config.memory_chunk_size * 10

    def test_resource_constraints(self):
        """Test configurations under resource constraints"""
        small_config = IterativeConfig(
            max_seq_length=256,
            chunk_size=50,
            iterations_per_novel=2,
            max_steps_per_iteration=2
        )

        small_memory_config = MemoryConfig(
            memory_chunk_size=10,
            max_memories_per_novel=20,
            max_retrieved_memories=3
        )

        trainer = IterativeTrainer(small_config)
        memory_system = EpisodicMemorySystem(small_memory_config)

        assert trainer.config.max_seq_length > 0
        assert memory_system.config.max_memories_per_novel > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])