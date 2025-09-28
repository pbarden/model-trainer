# Model Tea Core Framework

The `model_tea/core/` module provides the foundational ML pipeline infrastructure for the Model Tea framework, including pipeline orchestration, evaluation systems, and extensible stage management.

## Overview

Model Tea Core implements:
1. **Pipeline Management**: Orchestrated execution of ML workflows
2. **Stage-based Architecture**: Modular, dependency-aware processing stages
3. **Evaluation Framework**: Comprehensive model assessment and metrics
4. **Error Handling**: Robust error recovery and status tracking
5. **Extensibility**: Plugin architecture for custom stages and metrics

## Core Modules

### Pipeline Module (`pipeline.py`)

#### Classes

##### `PipelineConfig`

Configuration for ML pipeline execution.

```python
@dataclass
class PipelineConfig:
    pipeline_name: str                          # Pipeline identifier
    version: str = "2.0.0"                     # Pipeline version
    description: Optional[str] = None           # Pipeline description
    tags: List[str] = field(default_factory=list)  # Classification tags
    timeout_minutes: int = 120                  # Execution timeout
    retry_attempts: int = 3                     # Retry count on failure
    parallel_execution: bool = False            # Enable parallel stages
    checkpoint_enabled: bool = True             # Enable checkpointing
    checkpoint_dir: Optional[Path] = None       # Checkpoint directory
```

##### `StageStatus`

Enumeration of pipeline stage states.

```python
class StageStatus(Enum):
    PENDING = "pending"        # Stage not yet started
    RUNNING = "running"        # Stage currently executing
    COMPLETED = "completed"    # Stage finished successfully
    FAILED = "failed"          # Stage encountered error
    SKIPPED = "skipped"        # Stage was skipped
```

##### `PipelineStage`

Abstract base class for pipeline stages.

```python
class PipelineStage(ABC):
    def __init__(self, name: str, dependencies: Optional[List[str]] = None)
```

**Key Methods:**
- `execute(inputs: Dict[str, Any]) -> Any`: Main stage execution logic
- `can_execute(completed_stages: List[str]) -> bool`: Dependency checking
- `run(inputs: Dict[str, Any]) -> Any`: Wrapper with error handling

**Properties:**
- `status`: Current stage status
- `duration`: Execution time in seconds
- `error_message`: Error details if failed

##### `MLPipeline`

Main pipeline orchestrator.

```python
class MLPipeline:
    def __init__(self, config: PipelineConfig)
```

**Key Methods:**

###### `add_stage(stage: PipelineStage)`

Adds a stage to the pipeline with automatic dependency resolution.

**Example:**
```python
pipeline = MLPipeline(PipelineConfig(pipeline_name="training"))
pipeline.add_stage(DataPreprocessingStage())
pipeline.add_stage(ModelTrainingStage())
pipeline.add_stage(ModelEvaluationStage())
```

###### `execute(initial_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Executes the complete pipeline with dependency management.

**Parameters:**
- `initial_inputs`: Starting data for the pipeline

**Returns:**
- Dictionary containing all stage outputs

###### `get_stage_status() -> Dict[str, StageStatus]`

Returns current status of all stages.

###### `get_execution_summary() -> Dict[str, Any]`

Provides detailed execution summary with metrics.

#### Built-in Stages

##### `DataPreprocessingStage`

Handles data preprocessing operations.

```python
class DataPreprocessingStage(PipelineStage):
    def __init__(self, preprocessing_config: Dict[str, Any])
    def execute(self, inputs: Dict[str, Any]) -> Any
```

**Input Requirements:**
- `raw_data`: Input data to preprocess

**Outputs:**
- `processed_data`: Cleaned and prepared data
- `preprocessing_stats`: Processing statistics

##### `ModelTrainingStage`

Manages model training operations.

```python
class ModelTrainingStage(PipelineStage):
    def __init__(self, training_config: Dict[str, Any])
    def execute(self, inputs: Dict[str, Any]) -> Any
```

**Dependencies:** `["data_preprocessing"]`

**Input Requirements:**
- `processed_data`: Preprocessed training data

**Outputs:**
- `trained_model`: Trained model object
- `training_metrics`: Training performance metrics

##### `ModelEvaluationStage`

Handles model evaluation and assessment.

```python
class ModelEvaluationStage(PipelineStage):
    def __init__(self, evaluation_config: Dict[str, Any])
    def execute(self, inputs: Dict[str, Any]) -> Any
```

**Dependencies:** `["model_training"]`

**Input Requirements:**
- `trained_model`: Model to evaluate

**Outputs:**
- `evaluation_results`: Comprehensive evaluation metrics
- `model_approved`: Boolean approval status

## Evaluation Module (`evaluation.py`)

### Classes

##### `EvaluationMetrics`

Container for evaluation metrics with serialization support.

```python
@dataclass
class EvaluationMetrics:
    accuracy: Optional[float] = None            # Classification accuracy
    precision: Optional[float] = None           # Precision score
    recall: Optional[float] = None              # Recall score
    f1_score: Optional[float] = None            # F1 score
    auc_roc: Optional[float] = None             # ROC AUC score
    loss: Optional[float] = None                # Model loss
    custom_metrics: Optional[Dict[str, float]] = None  # Custom metrics
```

**Methods:**
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `__str__() -> str`: Human-readable representation

##### `MetricCalculator`

Abstract base class for metric calculators.

```python
class MetricCalculator(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float
```

##### `AccuracyCalculator`

Calculates classification accuracy.

```python
class AccuracyCalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float
```

##### `PrecisionCalculator`

Calculates precision with multiple averaging strategies.

```python
class PrecisionCalculator(MetricCalculator):
    def __init__(self, average: str = 'binary')  # 'binary', 'macro', 'weighted'
```

##### `RecallCalculator`

Calculates recall with multiple averaging strategies.

```python
class RecallCalculator(MetricCalculator):
    def __init__(self, average: str = 'binary')  # 'binary', 'macro', 'weighted'
```

##### `ModelEvaluator`

Comprehensive model evaluation framework.

```python
class ModelEvaluator:
    def __init__(self)
```

**Key Methods:**

###### `evaluate(model: Any, test_data: Any, metrics: Optional[List[str]] = None) -> EvaluationMetrics`

Performs comprehensive model evaluation.

**Parameters:**
- `model`: Model to evaluate
- `test_data`: Test dataset (dict with 'X' and 'y' keys, or tuple)
- `metrics`: List of metrics to calculate

**Returns:**
- `EvaluationMetrics` object with calculated scores

**Example:**
```python
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, {"X": X_test, "y": y_test},
                            ["accuracy", "precision", "recall"])
print(f"Accuracy: {metrics.accuracy:.3f}")
```

###### `add_custom_metric(name: str, calculator: MetricCalculator)`

Adds custom metric calculator.

**Example:**
```python
class CustomF2Calculator(MetricCalculator):
    def calculate(self, y_true, y_pred):
        # F2 score implementation
        return f2_score

evaluator.add_custom_metric("f2_score", CustomF2Calculator())
```

###### `cross_validate(model: Any, data: Any, cv_folds: int = 5, metrics: Optional[List[str]] = None) -> Dict[str, List[float]]`

Performs k-fold cross-validation.

**Parameters:**
- `model`: Model to validate
- `data`: Full dataset for cross-validation
- `cv_folds`: Number of folds
- `metrics`: Metrics to calculate

**Returns:**
- Dictionary mapping metric names to lists of scores

**Example:**
```python
cv_results = evaluator.cross_validate(model, data, cv_folds=5,
                                    metrics=["accuracy", "f1_score"])
print(f"CV Accuracy: {np.mean(cv_results['accuracy']):.3f} ± {np.std(cv_results['accuracy']):.3f}")
```

## Pipeline Usage Examples

### Basic Pipeline Creation

```python
from model_tea.core.pipeline import MLPipeline, PipelineConfig
from model_tea.core.pipeline import DataPreprocessingStage, ModelTrainingStage, ModelEvaluationStage

# Create pipeline configuration
config = PipelineConfig(
    pipeline_name="novel_training_pipeline",
    timeout_minutes=180,
    checkpoint_enabled=True
)

# Initialize pipeline
pipeline = MLPipeline(config)

# Add stages
preprocessing_config = {"normalize": True, "remove_outliers": True}
training_config = {"epochs": 10, "learning_rate": 0.001}
evaluation_config = {"test_size": 0.2}

pipeline.add_stage(DataPreprocessingStage(preprocessing_config))
pipeline.add_stage(ModelTrainingStage(training_config))
pipeline.add_stage(ModelEvaluationStage(evaluation_config))

# Execute pipeline
results = pipeline.execute({"raw_data": raw_dataset})

print(f"Training completed: {results['model_approved']}")
print(f"Final accuracy: {results['evaluation_results']['accuracy']}")
```

### Custom Pipeline Stage

```python
class FeatureEngineeringStage(PipelineStage):
    def __init__(self, feature_config: Dict[str, Any]):
        super().__init__("feature_engineering", dependencies=["data_preprocessing"])
        self.config = feature_config

    def execute(self, inputs: Dict[str, Any]) -> Any:
        processed_data = inputs["processed_data"]

        # Custom feature engineering logic
        engineered_features = self._create_features(processed_data)
        feature_importance = self._calculate_importance(engineered_features)

        return {
            "engineered_data": engineered_features,
            "feature_importance": feature_importance,
            "feature_count": len(engineered_features.columns)
        }

    def _create_features(self, data):
        # Implementation details
        pass

    def _calculate_importance(self, features):
        # Implementation details
        pass

# Add to pipeline
pipeline.add_stage(FeatureEngineeringStage({"create_polynomials": True}))
```

### Advanced Evaluation

```python
from model_tea.core.evaluation import ModelEvaluator, EvaluationMetrics

# Initialize evaluator
evaluator = ModelEvaluator()

# Add custom metrics
class PearsonCorrelationCalculator(MetricCalculator):
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.corrcoef(y_true, y_pred)[0, 1]

evaluator.add_custom_metric("pearson_correlation", PearsonCorrelationCalculator())

# Comprehensive evaluation
metrics = evaluator.evaluate(
    model=trained_model,
    test_data={"X": X_test, "y": y_test},
    metrics=["accuracy", "precision", "recall", "pearson_correlation"]
)

# Cross-validation
cv_results = evaluator.cross_validate(
    model=model,
    data={"X": X_full, "y": y_full},
    cv_folds=10,
    metrics=["accuracy", "f1_score"]
)

# Results analysis
print("Single Evaluation:")
print(metrics)

print("\nCross-Validation Results:")
for metric, scores in cv_results.items():
    print(f"{metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

## Pipeline Monitoring

### Status Tracking

```python
# During execution, monitor progress
while pipeline.status != "completed":
    stage_status = pipeline.get_stage_status()
    for stage_name, status in stage_status.items():
        print(f"{stage_name}: {status.value}")

    time.sleep(5)

# Get final summary
summary = pipeline.get_execution_summary()
print(f"Total stages: {summary['total_stages']}")
print(f"Completed: {summary['completed_stages']}")
print(f"Failed: {summary['failed_stages']}")
```

### Error Handling

```python
try:
    results = pipeline.execute(initial_data)
except Exception as e:
    # Get detailed error information
    summary = pipeline.get_execution_summary()

    for stage_detail in summary['stage_details']:
        if 'error' in stage_detail:
            print(f"Stage {stage_detail['name']} failed: {stage_detail['error']}")

    # Implement recovery strategy
    if "data_preprocessing" in [s['name'] for s in summary['stage_details'] if s['status'] == 'failed']:
        # Retry with different preprocessing parameters
        pass
```

## Performance Optimization

### Parallel Execution

```python
config = PipelineConfig(
    pipeline_name="parallel_pipeline",
    parallel_execution=True  # Enable parallel stage execution
)

# Independent stages will run in parallel
pipeline.add_stage(IndependentStageA())
pipeline.add_stage(IndependentStageB())
pipeline.add_stage(DependentStage(dependencies=["stage_a", "stage_b"]))
```

### Checkpointing

```python
config = PipelineConfig(
    pipeline_name="long_pipeline",
    checkpoint_enabled=True,
    checkpoint_dir=Path("checkpoints")
)

# Pipeline will automatically save state after each stage
# Can resume from last checkpoint on failure
```

### Memory Management

```python
class MemoryEfficientStage(PipelineStage):
    def execute(self, inputs: Dict[str, Any]) -> Any:
        # Process data in chunks to manage memory
        for chunk in self._chunk_data(inputs["large_dataset"]):
            processed_chunk = self._process_chunk(chunk)
            yield processed_chunk  # Stream processing

        # Cleanup intermediate results
        self._cleanup_temporary_files()
```

## Extension Points

### Custom Stages

```python
class CustomMLStage(PipelineStage):
    def __init__(self, custom_config: Dict[str, Any]):
        super().__init__("custom_ml_stage", dependencies=["preprocessing"])
        self.config = custom_config

    def execute(self, inputs: Dict[str, Any]) -> Any:
        # Custom ML logic
        return custom_results
```

### Custom Evaluators

```python
class DomainSpecificEvaluator(ModelEvaluator):
    def __init__(self, domain_config: Dict[str, Any]):
        super().__init__()
        self.domain_config = domain_config

        # Add domain-specific metrics
        self.add_custom_metric("domain_score", DomainScoreCalculator())

    def evaluate_with_domain_knowledge(self, model, test_data):
        # Domain-specific evaluation logic
        return enhanced_metrics
```

## Integration with Training System

### Novel Training Pipeline

```python
# Integration with iterative novel trainer
from iterative_novel_trainer import IterativeTrainer

class NovelTrainingStage(PipelineStage):
    def __init__(self, training_config: IterativeConfig):
        super().__init__("novel_training")
        self.trainer = IterativeTrainer(training_config)

    def execute(self, inputs: Dict[str, Any]) -> Any:
        novel_name = inputs["novel_name"]
        results = self.trainer.train_novel(novel_name)

        return {
            "trained_model": results["model"],
            "training_metrics": results["metrics"],
            "final_quality": results["quality_score"]
        }

# Create novel training pipeline
pipeline = MLPipeline(PipelineConfig(pipeline_name="novel_training"))
pipeline.add_stage(NovelTrainingStage(IterativeConfig()))
pipeline.add_stage(ModelEvaluationStage({}))
```

## Error Recovery

### Retry Strategies

```python
class RetryableStage(PipelineStage):
    def __init__(self, max_retries: int = 3):
        super().__init__("retryable_stage")
        self.max_retries = max_retries

    def execute(self, inputs: Dict[str, Any]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self._attempt_execution(inputs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
```

### Graceful Degradation

```python
class FallbackStage(PipelineStage):
    def execute(self, inputs: Dict[str, Any]) -> Any:
        try:
            return self._primary_method(inputs)
        except Exception as e:
            self.logger.warning(f"Primary method failed: {e}, using fallback")
            return self._fallback_method(inputs)
```

## See Also

- [Iterative Novel Trainer](iterative_trainer.md)
- [Model Tea Serving](model_tea_serving.md)
- [API Reference - Pipeline](../api/pipeline.md)
- [API Reference - Evaluation](../api/evaluation.md)
- [Feature Documentation - Pipeline Integration](../features/pipeline_integration.md)