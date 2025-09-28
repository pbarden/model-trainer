# API Reference

This directory contains comprehensive API documentation for all modules, classes, and functions in the Model Tea Iterative Novel Training System.

## Core APIs

### [Iterative Training API](iterative_trainer_api.md)
Complete API reference for the main training system including `IterativeTrainer` and `IterativeConfig`.

### [Episodic Memory API](memory_system_api.md)
API documentation for the memory system including `EpisodicMemorySystem`, `MemoryConfig`, and `Memory` classes.

### [Quality Validation API](quality_validator_api.md)
API reference for quality assessment and validation components.

## Model Tea Framework APIs

### [Pipeline API](pipeline_api.md)
ML pipeline framework API including `MLPipeline`, `PipelineStage`, and built-in stages.

### [Evaluation API](evaluation_api.md)
Model evaluation and metrics API including `ModelEvaluator` and metric calculators.

### [Serving API](serving_api.md)
Model serving and deployment API including `ModelServer` and `ServingConfig`.

## Utility APIs

### [Model Tea Utils API](utils_api.md)
Shared utilities and helper functions API reference.

### [Configuration API](configuration_api.md)
Configuration management and validation API.

## Quick Reference

### Class Hierarchy

```
IterativeTrainer
├── IterativeConfig
├── EpisodicMemorySystem
│   ├── MemoryConfig
│   ├── Memory
│   ├── MemoryExtractor
│   └── MemoryRetriever
└── QualityValidator
    └── ValidationConfig

MLPipeline
├── PipelineConfig
└── PipelineStage
    ├── DataPreprocessingStage
    ├── ModelTrainingStage
    └── ModelEvaluationStage

ModelEvaluator
├── EvaluationMetrics
└── MetricCalculator
    ├── AccuracyCalculator
    ├── PrecisionCalculator
    └── RecallCalculator

ModelServer
└── ServingConfig
```

### Key Enumerations

```python
class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ServerStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
```

## Usage Examples

### Basic Training

```python
from iterative_novel_trainer import IterativeTrainer, IterativeConfig

# Configure training
config = IterativeConfig(
    base_model="gpt2",
    iterations_per_novel=10,
    learning_rate_start=5e-5
)

# Initialize trainer
trainer = IterativeTrainer(config)

# Train on novel
results = trainer.train_novel("alice_in_wonderland")
```

### Memory System Usage

```python
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig

# Configure memory system
config = MemoryConfig(
    max_memories_per_novel=200,
    max_retrieved_memories=5
)

# Initialize memory system
memory_system = EpisodicMemorySystem(config)

# Build memories from novel
result = memory_system.build_memory_for_model("model_name", novel_path)

# Activate memories for prompt
memories, stats = memory_system.activate_memories("model_name", "prompt")
```

### Pipeline Usage

```python
from model_tea.core.pipeline import MLPipeline, PipelineConfig
from model_tea.core.pipeline import DataPreprocessingStage, ModelTrainingStage

# Configure pipeline
config = PipelineConfig(pipeline_name="training_pipeline")

# Initialize pipeline
pipeline = MLPipeline(config)

# Add stages
pipeline.add_stage(DataPreprocessingStage({}))
pipeline.add_stage(ModelTrainingStage({}))

# Execute pipeline
results = pipeline.execute({"raw_data": data})
```

## Error Handling

### Common Exceptions

```python
# Training errors
class TrainingError(Exception):
    """Raised when training fails"""
    pass

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass

# Memory errors
class MemoryError(Exception):
    """Raised when memory operations fail"""
    pass

class MemoryNotFoundError(Exception):
    """Raised when requested memories don't exist"""
    pass

# Pipeline errors
class PipelineError(Exception):
    """Raised when pipeline execution fails"""
    pass

class StageError(Exception):
    """Raised when individual stage fails"""
    pass
```

### Error Handling Patterns

```python
try:
    results = trainer.train_novel("novel_name")
except TrainingError as e:
    logger.error(f"Training failed: {e}")
    # Implement recovery strategy
except ModelLoadError as e:
    logger.error(f"Model loading failed: {e}")
    # Try alternative model
```

## Type Annotations

The API uses comprehensive type annotations for better development experience:

```python
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

def train_novel(self, novel_name: str) -> Dict[str, Any]: ...
def activate_memories(self, model_name: str, prompt: str) -> Tuple[List[Memory], Dict[str, Any]]: ...
def execute(self, initial_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
```

## Versioning

The API follows semantic versioning:

- **Major version**: Breaking changes to public API
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes, backward compatible

Current version: **2.0.0**

## Compatibility

### Python Version Support

- **Python 3.8+**: Minimum supported version
- **Python 3.9**: Recommended version
- **Python 3.10+**: Full feature support

### Dependency Compatibility

- **PyTorch**: 1.9+
- **Transformers**: 4.0+
- **NumPy**: 1.20+
- **scikit-learn**: 1.0+

## Testing

All API components include comprehensive test coverage:

```bash
# Run API tests
python -m pytest tests/test_iterative_trainer.py -v
python -m pytest tests/test_episodic_memory.py -v
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_evaluation.py -v
python -m pytest tests/test_serving.py -v
```

## Navigation

- [Module Documentation](../modules/README.md)
- [Feature Documentation](../features/README.md)
- [Installation Guide](../guides/installation.md)
- [Training Guide](../guides/training.md)