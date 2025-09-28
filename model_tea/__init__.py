"""
Model Tea: Production ML Training and Deployment Framework

A comprehensive framework for machine learning model training, optimization,
deployment, and monitoring in production environments.
"""

# Core functionality
from .core.models import ModelManager, ModelRegistry
from .core.training import TrainingCoordinator, TrainingConfig
from .core.evaluation import ModelEvaluator, EvaluationMetrics
from .core.pipeline import MLPipeline, PipelineStage

# Deployment
from .deployment.strategies import (
    DeploymentStrategy,
    RollingDeployment,
    BlueGreenDeployment,
    CanaryDeployment
)
from .deployment.serving import ModelServer, ServingConfig
from .deployment.monitoring import DeploymentMonitor

__version__ = "2.0.0"

__all__ = [
    # Core
    "ModelManager",
    "ModelRegistry",
    "TrainingCoordinator",
    "TrainingConfig",
    "ModelEvaluator",
    "EvaluationMetrics",
    "MLPipeline",
    "PipelineStage",

    # Deployment
    "DeploymentStrategy",
    "RollingDeployment",
    "BlueGreenDeployment",
    "CanaryDeployment",
    "ModelServer",
    "ServingConfig",
    "DeploymentMonitor",
]