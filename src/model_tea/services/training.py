import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from model_tea.trainers.iterative import IterativeTrainer, IterativeConfig
from model_tea.trainers.model import ModelTrainer, ModelConfig

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self):
        self.iterative_trainer = None
        self.model_trainer = None

    def train_model(
        self,
        model_key: str,
        config: Optional[ModelConfig] = None
    ) -> Dict[str, Any]:
        """Train a model from models.json (supports 1+ novels)"""
        if config is None:
            config = ModelConfig()

        self.model_trainer = ModelTrainer(config)

        results = self.model_trainer.train_model(model_key)

        return results

    def train_direct(
        self,
        novel_name: str,
        config: Optional[IterativeConfig] = None
    ) -> Dict[str, Any]:
        """Train a single novel directly from novels/ directory (for dev/testing)"""
        if config is None:
            config = IterativeConfig()

        self.iterative_trainer = IterativeTrainer(config)

        results = self.iterative_trainer.train_novel(novel_name)

        return results

    def list_available_novels(self) -> List[str]:
        novels_dir = Path("novels")
        if not novels_dir.exists():
            return []

        novels = []
        for novel_dir in novels_dir.iterdir():
            if novel_dir.is_dir() and list(novel_dir.glob("*.txt")):
                novels.append(novel_dir.name)

        return sorted(novels)

    def list_available_models(self) -> List[str]:
        """List all models defined in models.json"""
        config = ModelConfig()
        trainer = ModelTrainer(config)
        return trainer.get_available_models()

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "running",
            "progress": 0.0,
            "message": "Training in progress"
        }
