import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelTeaError(Exception):
    pass


class TrainingError(ModelTeaError):
    pass


class ValidationError(ModelTeaError):
    pass


class ConfigurationError(ModelTeaError):
    pass


class ErrorHandling:
    @staticmethod
    def handle_training_error(error: Exception, context: str) -> Dict[str, Any]:
        logger.error(f"Training error in {context}: {error}")
        return {
            "error": str(error),
            "context": context,
            "status": "failed",
            "recovery_suggestion": "Check dependencies and try again"
        }
