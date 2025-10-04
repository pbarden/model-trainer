import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Lightweight service for reading metadata without loading ML libraries.
    Fast operations for CLI listing and info commands.
    """

    def __init__(self):
        self.novels_file = Path("novels.json")
        self.models_file = Path("models.json")
        self.novels_dir = Path("novels")
        self.models_dir = Path("iterative_models")

    def load_novels(self) -> Dict[str, Any]:
        """Load novels.json"""
        if not self.novels_file.exists():
            return {}

        try:
            with open(self.novels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load novels.json: {e}")
            return {}

    def load_models(self) -> Dict[str, Any]:
        """Load models.json"""
        if not self.models_file.exists():
            return {}

        try:
            with open(self.models_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models.json: {e}")
            return {}

    def list_novels(self) -> List[Dict[str, Any]]:
        """List all novels from novels.json"""
        novels_data = self.load_novels()
        novels = []

        for novel_key, novel_info in novels_data.items():
            novels.append({
                "key": novel_key,
                "name": novel_info.get("original_name", novel_key),
                "directory": novel_info.get("directory_name", novel_key),
                "word_count": novel_info.get("word_count", 0)
            })

        return sorted(novels, key=lambda x: x["key"])

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models from models.json"""
        models_data = self.load_models()
        novels_data = self.load_novels()
        models = []

        for model_key, model_info in models_data.items():
            novel_keys = model_info.get("novels", [])
            total_words = sum(
                novels_data.get(nk, {}).get("word_count", 0)
                for nk in novel_keys
            )

            models.append({
                "key": model_key,
                "novel_count": len(novel_keys),
                "novel_keys": novel_keys,
                "total_words": total_words
            })

        return sorted(models, key=lambda x: x["key"])

    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List trained models from filesystem"""
        if not self.models_dir.exists():
            return []

        trained = []

        for item in self.models_dir.iterdir():
            if not item.is_dir():
                continue

            final_path = item / "final"
            if final_path.exists() and self._is_model_dir(final_path):
                size_mb = self._get_dir_size_mb(final_path)
                trained.append({
                    "key": item.name,
                    "path": str(final_path),
                    "size_mb": size_mb
                })

        return sorted(trained, key=lambda x: x["key"])

    def get_novel_info(self, novel_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific novel"""
        novels_data = self.load_novels()

        if novel_key not in novels_data:
            return None

        novel_info = novels_data[novel_key]
        return {
            "key": novel_key,
            "name": novel_info.get("original_name", novel_key),
            "directory": novel_info.get("directory_name", novel_key),
            "word_count": novel_info.get("word_count", 0),
            "exists": (self.novels_dir / novel_info.get("directory_name", novel_key)).exists()
        }

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        models_data = self.load_models()
        novels_data = self.load_novels()

        if model_key not in models_data:
            return None

        model_info = models_data[model_key]
        novel_keys = model_info.get("novels", [])

        novels = []
        total_words = 0

        for nk in novel_keys:
            if nk in novels_data:
                novel = novels_data[nk]
                novels.append({
                    "key": nk,
                    "name": novel.get("original_name", nk),
                    "word_count": novel.get("word_count", 0)
                })
                total_words += novel.get("word_count", 0)

        trained_path = self.models_dir / model_key / "final"
        is_trained = trained_path.exists() and self._is_model_dir(trained_path)

        return {
            "key": model_key,
            "novels": novels,
            "novel_count": len(novels),
            "total_words": total_words,
            "is_trained": is_trained,
            "trained_path": str(trained_path) if is_trained else None,
            "size_mb": self._get_dir_size_mb(trained_path) if is_trained else None
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        novels_data = self.load_novels()
        models_data = self.load_models()
        trained_models = self.list_trained_models()

        return {
            "novels_count": len(novels_data),
            "models_count": len(models_data),
            "trained_models_count": len(trained_models),
            "novels_file_exists": self.novels_file.exists(),
            "models_file_exists": self.models_file.exists(),
            "novels_dir_exists": self.novels_dir.exists(),
            "models_dir_exists": self.models_dir.exists()
        }

    def _is_model_dir(self, path: Path) -> bool:
        """Check if directory contains model files"""
        return (
            (path / "config.json").exists() or
            (path / "pytorch_model.bin").exists() or
            (path / "model.safetensors").exists()
        )

    def _get_dir_size_mb(self, path: Path) -> float:
        """Get directory size in MB"""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0
