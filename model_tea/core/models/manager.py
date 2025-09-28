"""
Model lifecycle management system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

from .metadata import ModelMetadata, ModelVersion
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelManager:
    """Advanced model lifecycle management system."""

    def __init__(self, registry: ModelRegistry, storage_path: str = "model_storage"):
        self.registry = registry
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def save_model(self,
                   model: Any,
                   model_id: str,
                   version: str = None,
                   **metadata) -> ModelVersion:
        """Save model with simplified metadata and smart defaults."""

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        version_dir = self.storage_path / model_id / version
        version_dir.mkdir(exist_ok=True, parents=True)

        model_path = str(version_dir / "model.pkl")
        with open(model_path, 'wb') as f:
            import pickle
            pickle.dump(model, f)

        checksum = self.registry._calculate_checksum(model_path)
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

        metrics = {}
        if 'accuracy' in metadata:
            metrics['accuracy'] = metadata.pop('accuracy')
        if 'loss' in metadata:
            metrics['loss'] = metadata.pop('loss')

        extra = {k: v for k, v in metadata.items() if k not in ['status', 'metrics']}
        extra['model_size_mb'] = model_size_mb

        model_metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            status=metadata.get("status", "training"),
            metrics=metrics,
            extra=extra
        )

        model_version = ModelVersion(
            metadata=model_metadata,
            model_path=model_path,
            checksum=checksum
        )

        self.registry.register_model(model_version)

        logger.info(f"Saved model {model_id} version {version}")
        return model_version

    def load_model(self, model_id: str, version: Optional[str] = None) -> Optional[Any]:
        """Load model from registry."""
        model_version = self.registry.get_model(model_id, version)
        if not model_version:
            return None

        try:
            with open(model_version.model_path, 'rb') as f:
                import pickle
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            return None

    def promote_model(self, model_id: str, version: str, target_status: str) -> bool:
        """Promote model to different lifecycle stage."""
        model_version = self.registry.get_model(model_id, version)
        if not model_version:
            return False

        model_version.metadata.status = target_status
        self.registry.register_model(model_version)

        logger.info(f"Promoted model {model_id}:{version} to {target_status}")
        return True

    def compare_models(self,
                      model1_id: str,
                      model2_id: str,
                      model1_version: Optional[str] = None,
                      model2_version: Optional[str] = None) -> Dict[str, Any]:
        """Compare two model versions."""
        model1 = self.registry.get_model(model1_id, model1_version)
        model2 = self.registry.get_model(model2_id, model2_version)

        if not model1 or not model2:
            return {"error": "One or both models not found"}

        comparison = {
            "model1": {
                "id": f"{model1_id}:{model1.metadata.version}",
                "metrics": model1.metadata.metrics,
                "created_at": model1.metadata.created_at,
                "status": model1.metadata.status,
                "model_size_mb": model1.metadata.extra.get("model_size_mb", 0)
            },
            "model2": {
                "id": f"{model2_id}:{model2.metadata.version}",
                "metrics": model2.metadata.metrics,
                "created_at": model2.metadata.created_at,
                "status": model2.metadata.status,
                "model_size_mb": model2.metadata.extra.get("model_size_mb", 0)
            },
            "differences": {}
        }

        all_metrics = set(model1.metadata.metrics.keys()) | set(model2.metadata.metrics.keys())
        for metric in all_metrics:
            val1 = model1.metadata.metrics.get(metric)
            val2 = model2.metadata.metrics.get(metric)

            if val1 is not None and val2 is not None:
                comparison["differences"][metric] = {
                    "model1": val1,
                    "model2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1) * 100 if val1 != 0 else float('inf')
                }

        return comparison