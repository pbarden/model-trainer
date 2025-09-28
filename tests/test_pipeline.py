"""
Test the ML pipeline functionality
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_tea.core.pipeline import (
    PipelineConfig, MLPipeline, PipelineStage, StageStatus,
    DataPreprocessingStage, ModelTrainingStage, ModelEvaluationStage
)


class TestPipelineConfig:
    """Test pipeline configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = PipelineConfig(pipeline_name="test_pipeline")

        assert config.pipeline_name == "test_pipeline"
        assert config.version == "2.0.0"
        assert config.timeout_minutes == 120
        assert config.retry_attempts == 3
        assert config.parallel_execution == False
        assert config.checkpoint_enabled == True

    def test_config_customization(self):
        """Test custom configuration"""
        config = PipelineConfig(
            pipeline_name="custom_pipeline",
            version="1.0.0",
            timeout_minutes=60,
            retry_attempts=2
        )

        assert config.pipeline_name == "custom_pipeline"
        assert config.version == "1.0.0"
        assert config.timeout_minutes == 60


class TestPipelineStage:
    """Test pipeline stage functionality"""

    def test_stage_initialization(self):
        """Test stage can be initialized"""
        stage = DataPreprocessingStage({})

        assert stage.name == "data_preprocessing"
        assert stage.status == StageStatus.PENDING
        assert stage.dependencies == []

    def test_stage_dependencies(self):
        """Test stage dependency checking"""
        stage = ModelTrainingStage({})

        assert stage.dependencies == ["data_preprocessing"]
        assert not stage.can_execute([])
        assert stage.can_execute(["data_preprocessing"])

    def test_stage_execution(self):
        """Test stage execution"""
        stage = DataPreprocessingStage({})

        input_data = {"raw_data": [1, 2, 3, 4, 5]}
        result = stage.run(input_data)

        assert stage.status == StageStatus.COMPLETED
        assert "processed_data" in result
        assert "preprocessing_stats" in result


class TestDataPreprocessingStage:
    """Test data preprocessing stage"""

    def test_preprocessing_execution(self):
        """Test preprocessing stage execution"""
        stage = DataPreprocessingStage({"normalize": True})

        input_data = {"raw_data": [1, 2, 3, 4, 5]}
        result = stage.execute(input_data)

        assert "processed_data" in result
        assert "preprocessing_stats" in result
        assert result["preprocessing_stats"]["num_samples"] == 5

    def test_preprocessing_no_data(self):
        """Test preprocessing with no input data"""
        stage = DataPreprocessingStage({})

        with pytest.raises(ValueError, match="No raw_data provided"):
            stage.execute({})


class TestModelTrainingStage:
    """Test model training stage"""

    def test_training_execution(self):
        """Test training stage execution"""
        stage = ModelTrainingStage({"epochs": 5, "max_iter": 100})

        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        input_data = {"processed_data": {"X": X, "y": y}}

        result = stage.execute(input_data)

        assert "trained_model" in result
        assert "training_metrics" in result
        assert result["training_metrics"]["epochs"] == 5

    def test_training_fallback_data(self):
        """Test training with fallback synthetic data"""
        stage = ModelTrainingStage({})

        input_data = {"processed_data": "invalid_data"}
        result = stage.execute(input_data)

        assert "trained_model" in result
        assert "training_metrics" in result

    def test_training_no_data(self):
        """Test training with no processed data"""
        stage = ModelTrainingStage({})

        with pytest.raises(ValueError, match="No processed_data available"):
            stage.execute({})


class TestModelEvaluationStage:
    """Test model evaluation stage"""

    def test_evaluation_execution(self):
        """Test evaluation stage execution"""
        from sklearn.linear_model import LogisticRegression

        stage = ModelEvaluationStage({})
        model = LogisticRegression()

        input_data = {"trained_model": model}
        result = stage.execute(input_data)

        assert "evaluation_results" in result
        assert "model_approved" in result
        assert result["evaluation_results"]["accuracy"] == 0.95

    def test_evaluation_no_model(self):
        """Test evaluation with no trained model"""
        stage = ModelEvaluationStage({})

        with pytest.raises(ValueError, match="No trained_model available"):
            stage.execute({})


class TestMLPipeline:
    """Test ML pipeline orchestration"""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        assert pipeline.config == config
        assert len(pipeline.stages) == 0
        assert len(pipeline.execution_order) == 0

    def test_add_stages(self):
        """Test adding stages to pipeline"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        preprocessing = DataPreprocessingStage({})
        training = ModelTrainingStage({})
        evaluation = ModelEvaluationStage({})

        pipeline.add_stage(preprocessing)
        pipeline.add_stage(training)
        pipeline.add_stage(evaluation)

        assert len(pipeline.stages) == 3
        assert "data_preprocessing" in pipeline.stages
        assert "model_training" in pipeline.stages
        assert "model_evaluation" in pipeline.stages

    def test_execution_order(self):
        """Test pipeline execution order based on dependencies"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        preprocessing = DataPreprocessingStage({})
        training = ModelTrainingStage({})
        evaluation = ModelEvaluationStage({})

        pipeline.add_stage(preprocessing)
        pipeline.add_stage(training)
        pipeline.add_stage(evaluation)

        expected_order = ["data_preprocessing", "model_training", "model_evaluation"]
        assert pipeline.execution_order == expected_order

    def test_pipeline_execution(self):
        """Test full pipeline execution"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        preprocessing = DataPreprocessingStage({})
        training = ModelTrainingStage({"epochs": 3})
        evaluation = ModelEvaluationStage({})

        pipeline.add_stage(preprocessing)
        pipeline.add_stage(training)
        pipeline.add_stage(evaluation)

        initial_data = {"raw_data": [1, 2, 3, 4, 5]}
        results = pipeline.execute(initial_data)

        assert "processed_data" in results
        assert "trained_model" in results
        assert "evaluation_results" in results

    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        class CircularStage(PipelineStage):
            def __init__(self, name, deps):
                super().__init__(name, deps)
            def execute(self, inputs):
                return {}

        stage1 = CircularStage("stage1", ["stage2"])
        stage2 = CircularStage("stage2", ["stage1"])

        # Adding stages in the right order to avoid missing dependency error first
        # We expect a circular dependency detection, so we'll add both and test exception
        pipeline.stages["stage1"] = stage1
        pipeline.stages["stage2"] = stage2

        with pytest.raises(ValueError, match="Circular dependency detected"):
            pipeline._update_execution_order()

    def test_missing_dependency(self):
        """Test missing dependency detection"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        training = ModelTrainingStage({})

        with pytest.raises(ValueError, match="Dependency .* not found"):
            pipeline.add_stage(training)

    def test_get_stage_status(self):
        """Test getting stage status"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        preprocessing = DataPreprocessingStage({})
        pipeline.add_stage(preprocessing)

        status = pipeline.get_stage_status()
        assert "data_preprocessing" in status
        assert status["data_preprocessing"] == StageStatus.PENDING

    def test_execution_summary(self):
        """Test getting execution summary"""
        config = PipelineConfig(pipeline_name="test_pipeline")
        pipeline = MLPipeline(config)

        preprocessing = DataPreprocessingStage({})
        pipeline.add_stage(preprocessing)

        summary = pipeline.get_execution_summary()
        assert summary["pipeline_name"] == "test_pipeline"
        assert summary["total_stages"] == 1
        assert summary["completed_stages"] == 0
        assert len(summary["stage_details"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])