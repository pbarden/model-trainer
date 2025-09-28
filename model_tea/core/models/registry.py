"""
Model registry for version control and metadata management.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
from functools import lru_cache

from .metadata import ModelMetadata, ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model registry for version control and metadata management."""

    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True, parents=True)
        self.models_db_path = self.registry_path / "models.json"
        self.models_db = self._load_models_db()
        self._cache = {}  # Simple in-memory cache

    def _load_models_db(self) -> Dict[str, Dict]:
        """Load models database from JSON file."""
        if self.models_db_path.exists():
            with open(self.models_db_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_models_db(self):
        """Save models database to JSON file."""
        with open(self.models_db_path, 'w') as f:
            json.dump(self.models_db, f, indent=2, default=str)

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def register_model(self, model_version: ModelVersion) -> str:
        """Register a new model version."""
        model_id = model_version.metadata.model_id
        version = model_version.metadata.version

        if model_id not in self.models_db:
            self.models_db[model_id] = {"versions": {}, "latest_version": version}

        version_data = {
            "metadata": asdict(model_version.metadata),
            "model_path": model_version.model_path,
            "checksum": model_version.checksum
        }

        self.models_db[model_id]["versions"][version] = version_data
        self.models_db[model_id]["latest_version"] = version

        self._save_models_db()
        logger.info(f"Registered model {model_id} version {version}")

        return f"{model_id}:{version}"

    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Retrieve model version from registry with caching."""
        if version is None and model_id in self.models_db:
            version = self.models_db[model_id]["latest_version"]

        cache_key = f"{model_id}:{version}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        if model_id not in self.models_db:
            return None

        if version not in self.models_db[model_id]["versions"]:
            return None

        version_data = self.models_db[model_id]["versions"][version]

        metadata_dict = version_data["metadata"]
        metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])

        metadata = ModelMetadata(**metadata_dict)

        model_version = ModelVersion(
            metadata=metadata,
            model_path=version_data["model_path"],
            checksum=version_data["checksum"]
        )

        # Cache the result
        self._cache[cache_key] = model_version

        return model_version

    def list_models(self, status_filter: Optional[str] = None) -> List[str]:
        """List all registered models, optionally filtered by status."""
        models = []
        for model_id, model_data in self.models_db.items():
            latest_version = model_data["latest_version"]
            if latest_version in model_data["versions"]:
                version_metadata = model_data["versions"][latest_version]["metadata"]
                model_status = version_metadata["status"]

                if status_filter is None or model_status == status_filter:
                    models.append(model_id)

        return models

    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete model version from registry."""
        if model_id not in self.models_db:
            return False

        if version is None:
            del self.models_db[model_id]
            logger.info(f"Deleted model {model_id}")
        else:
            if version in self.models_db[model_id]["versions"]:
                del self.models_db[model_id]["versions"][version]
                logger.info(f"Deleted model {model_id} version {version}")

                if self.models_db[model_id]["latest_version"] == version:
                    remaining_versions = list(self.models_db[model_id]["versions"].keys())
                    if remaining_versions:
                        self.models_db[model_id]["latest_version"] = max(remaining_versions)
                    else:
                        del self.models_db[model_id]
            else:
                return False

        self._save_models_db()
        return True