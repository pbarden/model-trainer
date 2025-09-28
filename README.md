# Model Tea

A production-ready machine learning framework for model training, deployment, and monitoring.

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Model Training

```python
import numpy as np
from model_tea import ModelManager, ModelRegistry, TrainingCoordinator, TrainingConfig

# Create mock model and data for demonstration
class SimpleModel:
    def __init__(self):
        self.weights = np.random.rand(10)

model = SimpleModel()
train_data = np.random.rand(100, 10)
val_data = np.random.rand(20, 10)

# Setup model management
registry = ModelRegistry("./model_registry")
manager = ModelManager(registry)

# Configure training
config = TrainingConfig(
    model_name="sentiment_classifier",
    batch_size=32,
    learning_rate=0.001,
    epochs=10
)

# Train and save model
trainer = TrainingCoordinator(config)
training_history = trainer.train(model, train_data, val_data)

# Save with metadata
version = manager.save_model(
    model=model,
    model_id="sentiment_classifier",
    version="1.0.0",
    metadata={
        "algorithm": "transformer",
        "accuracy": 0.94,
        "framework": "pytorch"
    }
)
```

### Model Deployment

```python
from datetime import datetime
from model_tea import CanaryDeployment, ModelServer, ServingConfig
from model_tea.deployment.strategies import DeploymentInfo, DeploymentStatus, DeploymentConfig

# Create a simple mock model
class MockModel:
    def predict(self, data):
        return {"prediction": "positive", "confidence": 0.95}

model = MockModel()

# Configure model serving
serving_config = ServingConfig(
    model_name="sentiment_classifier",
    version="1.0.0",
    host="0.0.0.0",
    port=8080,
    max_batch_size=32
)

# Setup model server
server = ModelServer(serving_config)
server.load_model(model)
# server.start()  # Uncomment to actually start server

# Deploy with canary strategy
deployment = CanaryDeployment()
deployment_info = DeploymentInfo(
    deployment_id="canary_deploy_1",
    model_id="sentiment_classifier",
    model_version="1.0.0",
    environment="production",
    status=DeploymentStatus.PENDING,
    strategy="canary",
    traffic_percentage=10.0,
    created_at=datetime.now(),
    updated_at=datetime.now(),
    health_check_url="http://localhost:8080/health",
    endpoint_url="http://localhost:8080/predict",
    replicas=1
)

config = DeploymentConfig(
    environment="production",
    resource_requirements={"cpu": "2", "memory": "4Gi"},
    scaling_config={"canary_phases": [5, 25, 50, 100]},
    monitoring_config={"success_threshold": 0.99}
)

success = deployment.deploy(deployment_info, config)
```

### Model Evaluation

```python
import numpy as np
from model_tea import ModelEvaluator

# Create mock model and test data
class MockModel:
    def predict(self, data):
        return np.random.randint(0, 2, len(data))

model = MockModel()
test_data = np.random.rand(100, 10)

# Evaluate model performance
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, test_data, ['accuracy', 'precision', 'recall'])

print(f"Model performance: {metrics}")
# Output: accuracy: 0.5600, precision: 0.5208, recall: 0.5435
```

### ML Pipeline

```python
import numpy as np
from model_tea import MLPipeline
from model_tea.core.pipeline import PipelineConfig, DataPreprocessingStage, ModelTrainingStage, ModelEvaluationStage

# Create training data
training_data = np.random.rand(100, 10)

# Configure pipeline
config = PipelineConfig(
    pipeline_name="sentiment_analysis_pipeline",
    version="1.0.0"
)

# Create pipeline
pipeline = MLPipeline(config)

# Add stages
pipeline.add_stage(DataPreprocessingStage({"normalize": True}))
pipeline.add_stage(ModelTrainingStage({"epochs": 10, "lr": 0.001}))
pipeline.add_stage(ModelEvaluationStage({"metrics": ["accuracy"]}))

# Execute pipeline
results = pipeline.execute({"raw_data": training_data})
print("Pipeline results:", list(results.keys()))
```

## Architecture

### Current Modules

- **core**: Model management, training coordination, evaluation, ML pipelines
- **deployment**: Deployment strategies, model serving, health monitoring

### Model Lifecycle

```
Training → Evaluation → Registration → Deployment → Monitoring
    ↓         ↓           ↓            ↓           ↓
Pipeline   Metrics     Versioning   Serving    Health Checks
```

## Features

### Model Management
- **Versioning**: Complete model version control with metadata
- **Registry**: Centralized model storage and discovery
- **Lifecycle**: Track model status through development stages

### Training & Evaluation
- **Training Coordination**: Structured training with configuration management
- **Evaluation Framework**: Comprehensive model assessment with multiple metrics
- **ML Pipelines**: End-to-end workflow orchestration

### Deployment
- **Multiple Strategies**: Rolling, Blue-Green, and Canary deployments
- **Model Serving**: HTTP server for model inference
- **Health Monitoring**: Deployment health checks and performance tracking

## API Reference

### Core Classes

#### ModelManager
```python
class ModelManager:
    def save_model(self, model, model_id, version, metadata) -> ModelVersion
    def load_model(self, model_id, version=None) -> Any
    def promote_model(self, model_id, version, target_status) -> bool
```

#### TrainingCoordinator
```python
class TrainingCoordinator:
    def __init__(self, config: TrainingConfig)
    def train(self, model, train_data, val_data=None) -> Dict[str, Any]
    def add_callback(self, callback: Callable)
```

#### ModelEvaluator
```python
class ModelEvaluator:
    def evaluate(self, model, test_data, metrics=None) -> EvaluationMetrics
    def cross_validate(self, model, data, cv_folds=5) -> Dict[str, List[float]]
```

#### DeploymentStrategy
```python
class DeploymentStrategy:
    def deploy(self, deployment_info, config) -> bool
    def rollback(self, deployment_info) -> bool
    def health_check(self, deployment_info) -> bool
```

## Development

### Running Tests
```bash
# Install in development mode
pip install -e .

# Run tests (when available)
python -m pytest tests/

# Test imports
python -c "import model_tea; print('Import successful')"
```

### Project Structure
```
model_tea/
├── core/
│   ├── models.py      # ModelManager, ModelRegistry
│   ├── training.py    # TrainingCoordinator, TrainingConfig
│   ├── evaluation.py  # ModelEvaluator, EvaluationMetrics
│   └── pipeline.py    # MLPipeline, PipelineStage
└── deployment/
    ├── strategies.py  # Deployment strategies
    ├── serving.py     # ModelServer, ServingConfig
    └── monitoring.py  # DeploymentMonitor
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Test your changes (`python -c "import model_tea"`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.