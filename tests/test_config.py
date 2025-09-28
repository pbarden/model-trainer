"""
Test configuration and validation across all systems
"""

import pytest
import tempfile
from pathlib import Path
from dataclasses import fields

from iterative_novel_trainer import IterativeConfig
from episodic_memory_system import MemoryConfig
from model_tea.core.pipeline import PipelineConfig
from model_tea.deployment.serving import ServingConfig


class TestIterativeConfig:
    """Test iterative training configuration"""

    def test_default_values(self):
        """Test all default configuration values"""
        config = IterativeConfig()

        assert config.base_model == "gpt2"
        assert config.max_seq_length == 1024
        assert config.iterations_per_novel == 12
        assert config.max_steps_per_iteration == 8
        assert config.learning_rate_start == 5e-5
        assert config.learning_rate_end == 2e-5
        assert config.chunk_size == 512
        assert config.chunk_overlap == 0.1
        assert config.validation_split == 0.1
        assert config.novels_dir == "novels"
        assert config.output_dir == "output"

    def test_custom_values(self):
        """Test custom configuration values"""
        config = IterativeConfig(
            base_model="gpt2-medium",
            max_seq_length=2048,
            iterations_per_novel=20,
            max_steps_per_iteration=15,
            learning_rate_start=1e-4,
            learning_rate_end=1e-5,
            chunk_size=1024,
            validation_split=0.2
        )

        assert config.base_model == "gpt2-medium"
        assert config.max_seq_length == 2048
        assert config.iterations_per_novel == 20
        assert config.max_steps_per_iteration == 15
        assert config.learning_rate_start == 1e-4
        assert config.learning_rate_end == 1e-5
        assert config.chunk_size == 1024
        assert config.validation_split == 0.2

    def test_learning_rate_validation(self):
        """Test learning rate configuration constraints"""
        config = IterativeConfig(
            learning_rate_start=1e-4,
            learning_rate_end=1e-5
        )

        assert config.learning_rate_start > config.learning_rate_end

        config_equal = IterativeConfig(
            learning_rate_start=1e-4,
            learning_rate_end=1e-4
        )
        assert config_equal.learning_rate_start == config_equal.learning_rate_end

    def test_positive_value_constraints(self):
        """Test that positive values are maintained"""
        config = IterativeConfig(
            max_seq_length=512,
            iterations_per_novel=5,
            max_steps_per_iteration=3,
            chunk_size=256
        )

        assert config.max_seq_length > 0
        assert config.iterations_per_novel > 0
        assert config.max_steps_per_iteration > 0
        assert config.chunk_size > 0

    def test_validation_split_range(self):
        """Test validation split is in valid range"""
        config_low = IterativeConfig(validation_split=0.05)
        config_high = IterativeConfig(validation_split=0.3)

        assert 0 < config_low.validation_split < 1
        assert 0 < config_high.validation_split < 1

    def test_chunk_overlap_range(self):
        """Test chunk overlap is in valid range"""
        config = IterativeConfig(chunk_overlap=0.2)

        assert 0 <= config.chunk_overlap < 1

    def test_path_configuration(self):
        """Test path configuration"""
        config = IterativeConfig(
            novels_dir="custom_novels",
            output_dir="custom_output"
        )

        assert config.novels_dir == "custom_novels"
        assert config.output_dir == "custom_output"


class TestMemoryConfig:
    """Test episodic memory configuration"""

    def test_default_values(self):
        """Test default memory configuration"""
        config = MemoryConfig()

        assert config.memory_chunk_size == 35
        assert config.overlap_ratio == 0.3
        assert config.max_memories_per_novel == 250
        assert config.max_retrieved_memories == 5
        assert config.relevance_threshold == 0.1
        assert config.use_simple_similarity == True

    def test_custom_values(self):
        """Test custom memory configuration"""
        config = MemoryConfig(
            memory_chunk_size=50,
            overlap_ratio=0.2,
            max_memories_per_novel=500,
            max_retrieved_memories=10,
            relevance_threshold=0.2,
            use_simple_similarity=False
        )

        assert config.memory_chunk_size == 50
        assert config.overlap_ratio == 0.2
        assert config.max_memories_per_novel == 500
        assert config.max_retrieved_memories == 10
        assert config.relevance_threshold == 0.2
        assert config.use_simple_similarity == False

    def test_positive_constraints(self):
        """Test positive value constraints"""
        config = MemoryConfig(
            memory_chunk_size=25,
            max_memories_per_novel=100,
            max_retrieved_memories=3
        )

        assert config.memory_chunk_size > 0
        assert config.max_memories_per_novel > 0
        assert config.max_retrieved_memories > 0

    def test_ratio_constraints(self):
        """Test ratio constraints"""
        config = MemoryConfig(
            overlap_ratio=0.25,
            relevance_threshold=0.15
        )

        assert 0 <= config.overlap_ratio < 1
        assert 0 <= config.relevance_threshold <= 1

    def test_memory_limits(self):
        """Test memory limit constraints"""
        config = MemoryConfig(
            max_memories_per_novel=1000,
            max_retrieved_memories=20
        )

        assert config.max_retrieved_memories <= config.max_memories_per_novel


class TestPipelineConfig:
    """Test ML pipeline configuration"""

    def test_default_values(self):
        """Test default pipeline configuration"""
        config = PipelineConfig(pipeline_name="test_pipeline")

        assert config.pipeline_name == "test_pipeline"
        assert config.version == "2.0.0"
        assert config.timeout_minutes == 120
        assert config.retry_attempts == 3
        assert config.parallel_execution == False
        assert config.checkpoint_enabled == True

    def test_custom_values(self):
        """Test custom pipeline configuration"""
        config = PipelineConfig(
            pipeline_name="custom_pipeline",
            version="3.0.0",
            timeout_minutes=180,
            retry_attempts=5,
            parallel_execution=True,
            checkpoint_enabled=False
        )

        assert config.pipeline_name == "custom_pipeline"
        assert config.version == "3.0.0"
        assert config.timeout_minutes == 180
        assert config.retry_attempts == 5
        assert config.parallel_execution == True
        assert config.checkpoint_enabled == False

    def test_checkpoint_directory(self):
        """Test checkpoint directory configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PipelineConfig(
                pipeline_name="test",
                checkpoint_dir=temp_dir
            )

            assert config.checkpoint_dir == Path(temp_dir)

    def test_positive_constraints(self):
        """Test positive value constraints"""
        config = PipelineConfig(
            pipeline_name="test",
            timeout_minutes=60,
            retry_attempts=2
        )

        assert config.timeout_minutes > 0
        assert config.retry_attempts >= 0

    def test_required_fields(self):
        """Test required configuration fields"""
        config = PipelineConfig(pipeline_name="required_test")
        assert config.pipeline_name is not None
        assert len(config.pipeline_name) > 0


class TestServingConfig:
    """Test model serving configuration"""

    def test_default_values(self):
        """Test default serving configuration"""
        config = ServingConfig(model_name="test_model", version="1.0.0")

        assert config.model_name == "test_model"
        assert config.version == "1.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.max_batch_size == 32
        assert config.timeout_seconds == 30
        assert config.workers == 1
        assert config.enable_metrics == True
        assert config.enable_logging == True
        assert config.cors_enabled == False
        assert config.auth_enabled == False

    def test_custom_values(self):
        """Test custom serving configuration"""
        config = ServingConfig(
            model_name="custom_model",
            version="2.0.0",
            host="127.0.0.1",
            port=9000,
            max_batch_size=64,
            timeout_seconds=60,
            workers=4,
            enable_metrics=False,
            cors_enabled=True,
            auth_enabled=True
        )

        assert config.model_name == "custom_model"
        assert config.version == "2.0.0"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_batch_size == 64
        assert config.timeout_seconds == 60
        assert config.workers == 4
        assert config.enable_metrics == False
        assert config.cors_enabled == True
        assert config.auth_enabled == True

    def test_port_range(self):
        """Test valid port range"""
        config_low = ServingConfig(model_name="test", version="1.0", port=1024)
        config_high = ServingConfig(model_name="test", version="1.0", port=65535)

        assert 1 <= config_low.port <= 65535
        assert 1 <= config_high.port <= 65535

    def test_positive_constraints(self):
        """Test positive value constraints"""
        config = ServingConfig(
            model_name="test",
            version="1.0",
            max_batch_size=16,
            timeout_seconds=45,
            workers=2
        )

        assert config.max_batch_size > 0
        assert config.timeout_seconds > 0
        assert config.workers > 0

    def test_endpoint_paths(self):
        """Test endpoint path configuration"""
        config = ServingConfig(
            model_name="test",
            version="1.0",
            health_check_path="/custom_health",
            prediction_path="/custom_predict",
            metadata_path="/custom_metadata"
        )

        assert config.health_check_path == "/custom_health"
        assert config.prediction_path == "/custom_predict"
        assert config.metadata_path == "/custom_metadata"

    def test_rate_limiting(self):
        """Test rate limiting configuration"""
        config = ServingConfig(
            model_name="test",
            version="1.0",
            rate_limit_requests_per_minute=100
        )

        assert config.rate_limit_requests_per_minute == 100

    def test_required_fields(self):
        """Test required configuration fields"""
        config = ServingConfig(model_name="required_test", version="1.0.0")
        assert config.model_name is not None
        assert config.version is not None
        assert len(config.model_name) > 0
        assert len(config.version) > 0


class TestConfigurationCompatibility:
    """Test configuration compatibility across systems"""

    def test_trainer_memory_compatibility(self):
        """Test trainer and memory config compatibility"""
        trainer_config = IterativeConfig(
            chunk_size=512,
            max_seq_length=1024
        )

        memory_config = MemoryConfig(
            memory_chunk_size=25,
            max_memories_per_novel=200
        )

        assert trainer_config.chunk_size >= memory_config.memory_chunk_size
        assert trainer_config.max_seq_length >= memory_config.memory_chunk_size * 10

    def test_serving_memory_compatibility(self):
        """Test serving and memory config compatibility"""
        serving_config = ServingConfig(
            model_name="test",
            version="1.0",
            max_batch_size=32,
            timeout_seconds=30
        )

        memory_config = MemoryConfig(
            max_retrieved_memories=5,
            max_memories_per_novel=200
        )

        assert serving_config.max_batch_size >= memory_config.max_retrieved_memories

    def test_pipeline_trainer_compatibility(self):
        """Test pipeline and trainer config compatibility"""
        pipeline_config = PipelineConfig(
            pipeline_name="training_pipeline",
            timeout_minutes=120,
            retry_attempts=3
        )

        trainer_config = IterativeConfig(
            iterations_per_novel=12,
            max_steps_per_iteration=8
        )

        total_steps = trainer_config.iterations_per_novel * trainer_config.max_steps_per_iteration
        timeout_seconds = pipeline_config.timeout_minutes * 60

        assert timeout_seconds > total_steps

    def test_cross_system_validation(self):
        """Test validation across all system configurations"""
        trainer_config = IterativeConfig(chunk_size=256, max_seq_length=512)
        memory_config = MemoryConfig(memory_chunk_size=20, max_memories_per_novel=100)
        pipeline_config = PipelineConfig(pipeline_name="integrated", timeout_minutes=60)
        serving_config = ServingConfig(model_name="integrated_model", version="1.0", max_batch_size=16)

        assert trainer_config.chunk_size > memory_config.memory_chunk_size
        assert serving_config.max_batch_size > memory_config.max_retrieved_memories
        assert pipeline_config.timeout_minutes > 0
        assert all([
            trainer_config.chunk_size > 0,
            memory_config.memory_chunk_size > 0,
            serving_config.max_batch_size > 0
        ])


class TestConfigurationEdgeCases:
    """Test configuration edge cases and boundaries"""

    def test_minimum_values(self):
        """Test minimum viable configuration values"""
        trainer_config = IterativeConfig(
            iterations_per_novel=1,
            max_steps_per_iteration=1,
            chunk_size=50,
            max_seq_length=100
        )

        memory_config = MemoryConfig(
            memory_chunk_size=5,
            max_memories_per_novel=1,
            max_retrieved_memories=1
        )

        assert trainer_config.iterations_per_novel >= 1
        assert memory_config.max_memories_per_novel >= 1
        assert memory_config.max_retrieved_memories >= 1

    def test_maximum_reasonable_values(self):
        """Test maximum reasonable configuration values"""
        trainer_config = IterativeConfig(
            iterations_per_novel=100,
            max_steps_per_iteration=50,
            chunk_size=2048,
            max_seq_length=4096
        )

        memory_config = MemoryConfig(
            memory_chunk_size=100,
            max_memories_per_novel=10000,
            max_retrieved_memories=50
        )

        assert trainer_config.chunk_size <= trainer_config.max_seq_length
        assert memory_config.max_retrieved_memories <= memory_config.max_memories_per_novel

    def test_boundary_conditions(self):
        """Test boundary condition configurations"""
        config_zero_overlap = IterativeConfig(chunk_overlap=0.0)
        config_max_overlap = IterativeConfig(chunk_overlap=0.99)

        assert config_zero_overlap.chunk_overlap == 0.0
        assert config_max_overlap.chunk_overlap < 1.0

        memory_config_min_threshold = MemoryConfig(relevance_threshold=0.0)
        memory_config_max_threshold = MemoryConfig(relevance_threshold=1.0)

        assert memory_config_min_threshold.relevance_threshold == 0.0
        assert memory_config_max_threshold.relevance_threshold == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])