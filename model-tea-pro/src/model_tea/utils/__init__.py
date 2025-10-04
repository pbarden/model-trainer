import logging
from pathlib import Path
from typing import Dict

from .file_utils import FileSystemUtils
from .text_processing import TextProcessingUtils, QualityMetrics
from .training_utils import TrainingUtils
from .memory_utils import MemorySystemUtils
from .errors import ErrorHandling, ModelTeaError, TrainingError, ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


def validate_system_setup() -> Dict[str, bool]:
    validation = {
        "directories": True,
        "dependencies": True,
        "memory_system": True,
        "models": True
    }

    try:
        required_dirs = ["novels", "iterative_models", "episodic_memories"]
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)

        import torch
        import transformers
        import datasets

        validation["memory_system"] = MemorySystemUtils.check_memory_system_available()

        logger.info("Model Tea system validation passed")

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        validation["dependencies"] = False

    return validation


__all__ = [
    "FileSystemUtils",
    "TextProcessingUtils",
    "QualityMetrics",
    "TrainingUtils",
    "MemorySystemUtils",
    "ErrorHandling",
    "ModelTeaError",
    "TrainingError",
    "ValidationError",
    "ConfigurationError",
    "validate_system_setup",
]
