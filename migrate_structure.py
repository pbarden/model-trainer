#!/usr/bin/env python3
"""
Migration script to transition from old model_tea structure to new structure.

This script:
1. Backs up the current model_tea directory
2. Creates the new structure
3. Migrates code with updated imports
4. Updates documentation
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def backup_current_structure():
    """Backup current model_tea directory."""
    print("Backing up current structure...")

    if Path("model_tea").exists():
        if Path("model_tea_backup").exists():
            shutil.rmtree("model_tea_backup")
        shutil.copytree("model_tea", "model_tea_backup")
        print("Backup created: model_tea_backup/")
    else:
        print("WARNING: No model_tea directory found")


def create_new_structure():
    """Create new directory structure."""
    print("Creating new directory structure...")

    directories = [
        "model_tea_new/core",
        "model_tea_new/deployment",
        "model_tea_new/optimization",
        "model_tea_new/testing",
        "model_tea_new/monitoring",
        "model_tea_new/cloud",
        "model_tea_new/utils"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"Created {len(directories)} directories")


def migrate_core_modules():
    """Migrate core functionality."""
    print("Migrating core modules...")

    # Migration mapping: old_file -> new_file
    migrations = {
        "model_tea/mlops/model_manager.py": "model_tea_new/core/models.py",
        "model_tea/mlops/pipeline.py": "model_tea_new/core/pipeline.py",
        "model_tea/core/training/training_coordinator.py": "model_tea_new/core/training.py",
        "model_tea/analysis/metrics/evaluation_framework.py": "model_tea_new/core/evaluation.py",
    }

    migrated_count = 0
    for old_path, new_path in migrations.items():
        if Path(old_path).exists():
            # Copy and update imports
            with open(old_path, 'r') as f:
                content = f.read()

            # Update imports (basic replacements)
            content = update_imports(content)

            with open(new_path, 'w') as f:
                f.write(content)

            migrated_count += 1
            print(f"  {old_path} -> {new_path}")

    print(f"Migrated {migrated_count} core modules")


def migrate_deployment_modules():
    """Migrate deployment functionality."""
    print("Migrating deployment modules...")

    migrations = {
        "model_tea/mlops/deployment.py": "model_tea_new/deployment/strategies.py",
        "model_tea/monitoring/performance_monitor.py": "model_tea_new/deployment/monitoring.py",
    }

    migrated_count = 0
    for old_path, new_path in migrations.items():
        if Path(old_path).exists():
            with open(old_path, 'r') as f:
                content = f.read()

            content = update_imports(content)

            with open(new_path, 'w') as f:
                f.write(content)

            migrated_count += 1
            print(f"  {old_path} -> {new_path}")

    print(f"Migrated {migrated_count} deployment modules")


def migrate_optimization_modules():
    """Migrate optimization functionality."""
    print("Migrating optimization modules...")

    migrations = {
        "model_tea/distributed/hyperparameter_search.py": "model_tea_new/optimization/hyperparameters.py",
        "model_tea/architectures/neural_architecture_search.py": "model_tea_new/optimization/architecture.py",
        "model_tea/architectures/ensemble_learning.py": "model_tea_new/optimization/ensembles.py",
        "model_tea/distributed/multi_gpu.py": "model_tea_new/optimization/distributed.py",
    }

    migrated_count = 0
    for old_path, new_path in migrations.items():
        if Path(old_path).exists():
            with open(old_path, 'r') as f:
                content = f.read()

            content = update_imports(content)

            with open(new_path, 'w') as f:
                f.write(content)

            migrated_count += 1
            print(f"  {old_path} -> {new_path}")

    print(f"Migrated {migrated_count} optimization modules")


def migrate_testing_modules():
    """Migrate testing functionality."""
    print("Migrating testing modules...")

    migrations = {
        "model_tea/mlops/ab_testing.py": "model_tea_new/testing/ab_testing.py",
        "model_tea/analysis/causal_inference.py": "model_tea_new/testing/statistics.py",
    }

    migrated_count = 0
    for old_path, new_path in migrations.items():
        if Path(old_path).exists():
            with open(old_path, 'r') as f:
                content = f.read()

            content = update_imports(content)

            with open(new_path, 'w') as f:
                f.write(content)

            migrated_count += 1
            print(f"  {old_path} -> {new_path}")

    print(f"Migrated {migrated_count} testing modules")


def migrate_cloud_modules():
    """Migrate cloud functionality."""
    print("Migrating cloud modules...")

    migrations = {
        "model_tea/distributed/cloud_native.py": "model_tea_new/cloud/kubernetes.py",
        "model_tea/distributed/federated.py": "model_tea_new/cloud/federated.py",
    }

    migrated_count = 0
    for old_path, new_path in migrations.items():
        if Path(old_path).exists():
            with open(old_path, 'r') as f:
                content = f.read()

            content = update_imports(content)

            with open(new_path, 'w') as f:
                f.write(content)

            migrated_count += 1
            print(f"  {old_path} -> {new_path}")

    print(f"Migrated {migrated_count} cloud modules")


def update_imports(content: str) -> str:
    """Update import statements for new structure."""
    import_mappings = {
        "from model_tea.mlops.": "from model_tea.core.",
        "from model_tea.distributed.": "from model_tea.optimization.",
        "from model_tea.architectures.": "from model_tea.optimization.",
        "from model_tea.monitoring.": "from model_tea.monitoring.",
        "from model_tea.analysis.": "from model_tea.testing.",
        "from model_tea.integration.": "from model_tea.testing.",
    }

    for old_import, new_import in import_mappings.items():
        content = content.replace(old_import, new_import)

    return content


def create_init_files():
    """Create __init__.py files for all modules."""
    print("Creating __init__.py files...")

    init_files = [
        ("model_tea_new/core/__init__.py", get_core_init()),
        ("model_tea_new/deployment/__init__.py", get_deployment_init()),
        ("model_tea_new/optimization/__init__.py", get_optimization_init()),
        ("model_tea_new/testing/__init__.py", get_testing_init()),
        ("model_tea_new/monitoring/__init__.py", get_monitoring_init()),
        ("model_tea_new/cloud/__init__.py", get_cloud_init()),
        ("model_tea_new/utils/__init__.py", get_utils_init()),
    ]

    for file_path, content in init_files:
        with open(file_path, 'w') as f:
            f.write(content)

    print(f"Created {len(init_files)} __init__.py files")


def get_core_init() -> str:
    return '''"""
Core functionality for model management, training, and evaluation.
"""

from .models import ModelManager, ModelRegistry, ModelMetadata
from .training import TrainingCoordinator, TrainingConfig
from .evaluation import ModelEvaluator, EvaluationMetrics
from .pipeline import MLPipeline, PipelineStage

__all__ = [
    "ModelManager", "ModelRegistry", "ModelMetadata",
    "TrainingCoordinator", "TrainingConfig",
    "ModelEvaluator", "EvaluationMetrics",
    "MLPipeline", "PipelineStage"
]
'''


def get_deployment_init() -> str:
    return '''"""
Model deployment and serving functionality.
"""

from .strategies import DeploymentStrategy, RollingDeployment, BlueGreenDeployment, CanaryDeployment
from .monitoring import DeploymentMonitor

__all__ = [
    "DeploymentStrategy", "RollingDeployment", "BlueGreenDeployment",
    "CanaryDeployment", "DeploymentMonitor"
]
'''


def get_optimization_init() -> str:
    return '''"""
Model and architecture optimization.
"""

from .hyperparameters import HyperparameterOptimizer, SearchSpace
from .architecture import ArchitectureSearch, NASConfig
from .ensembles import EnsembleBuilder, EnsembleConfig
from .distributed import DistributedTrainer, GPUCluster

__all__ = [
    "HyperparameterOptimizer", "SearchSpace",
    "ArchitectureSearch", "NASConfig",
    "EnsembleBuilder", "EnsembleConfig",
    "DistributedTrainer", "GPUCluster"
]
'''


def get_testing_init() -> str:
    return '''"""
A/B testing and statistical analysis.
"""

from .ab_testing import ABTest, ABTestConfig, TrafficSplitter
from .statistics import StatisticalAnalyzer

__all__ = [
    "ABTest", "ABTestConfig", "TrafficSplitter",
    "StatisticalAnalyzer"
]
'''


def get_monitoring_init() -> str:
    return '''"""
Performance monitoring and alerting.
"""

from .metrics import MetricsCollector, PerformanceMonitor
from .alerts import AlertManager, AlertRule

__all__ = [
    "MetricsCollector", "PerformanceMonitor",
    "AlertManager", "AlertRule"
]
'''


def get_cloud_init() -> str:
    return '''"""
Cloud infrastructure and deployment.
"""

from .kubernetes import KubernetesDeployer
from .scaling import AutoScaler
from .resources import ResourceOptimizer

__all__ = [
    "KubernetesDeployer", "AutoScaler", "ResourceOptimizer"
]
'''


def get_utils_init() -> str:
    return '''"""
Utility functions and helpers.
"""

from .config import ConfigManager
from .logging import setup_logging
from .helpers import validate_config

__all__ = [
    "ConfigManager", "setup_logging", "validate_config"
]
'''


def finalize_migration():
    """Finalize the migration."""
    print("Finalizing migration...")

    # Replace old structure with new
    if Path("model_tea_new").exists():
        if Path("model_tea").exists():
            shutil.rmtree("model_tea")
        shutil.move("model_tea_new", "model_tea")

        # Replace README
        if Path("README_NEW.md").exists():
            shutil.move("README_NEW.md", "README.md")

        print("Migration completed successfully!")
        print("Old structure backed up to: model_tea_backup/")
        print("Updated documentation in: README.md")
    else:
        print("Migration failed - model_tea_new directory not found")


def main():
    """Run the complete migration."""
    print("Starting Model Tea Structure Migration")
    print("=" * 50)

    try:
        backup_current_structure()
        create_new_structure()

        # Only migrate if we have existing structure
        if Path("model_tea").exists():
            migrate_core_modules()
            migrate_deployment_modules()
            migrate_optimization_modules()
            migrate_testing_modules()
            migrate_cloud_modules()
        else:
            print("No existing model_tea structure to migrate")

        create_init_files()
        finalize_migration()

        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("\nNext steps:")
        print("1. Review migrated code in model_tea/")
        print("2. Update any remaining import statements")
        print("3. Run tests: pytest tests/")
        print("4. Update documentation as needed")

    except Exception as e:
        print(f"Migration failed: {e}")
        print("Please check the error and try again")


if __name__ == "__main__":
    main()