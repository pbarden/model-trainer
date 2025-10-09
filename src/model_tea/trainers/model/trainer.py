import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from .config import ModelConfig
from model_tea.core.validator import QualityValidator, ValidationConfig
from model_tea.trainers.iterative import IterativeTrainer, IterativeConfig
from model_tea.utils import FileSystemUtils, validate_system_setup

warnings.filterwarnings("ignore", message=".*Using the model-agnostic default.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download is deprecated.*", category=FutureWarning)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains models using one or more novels from novels.json and models.json
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.novels_data, self.models_data = self._load_mapping_files()
        self.quality_validator = QualityValidator(ValidationConfig())

        self.novels_dir = Path("novels")
        self.models_dir = Path("iterative_models")

        validate_system_setup()

    def _load_mapping_files(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load novels.json and models.json"""
        novels_file = Path("novels.json")
        models_file = Path("models.json")

        if not novels_file.exists():
            raise FileNotFoundError("novels.json not found. Required for model training.")
        if not models_file.exists():
            raise FileNotFoundError("models.json not found. Required for model training.")

        try:
            with open(novels_file, 'r', encoding='utf-8') as f:
                novels = json.load(f)
            with open(models_file, 'r', encoding='utf-8') as f:
                models = json.load(f)
            return novels, models
        except Exception as e:
            raise Exception(f"Failed to load mapping files: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available models from models.json"""
        return list(self.models_data.keys())

    def get_novels_for_model(self, model_key: str) -> List[Dict[str, Any]]:
        """Get novel list for specified model"""
        if model_key not in self.models_data:
            available = ", ".join(self.get_available_models())
            raise ValueError(f"Model '{model_key}' not found. Available models: {available}")

        model_info = self.models_data[model_key]
        novel_keys = model_info.get("novels", [])

        novels = []
        for novel_key in novel_keys:
            if novel_key not in self.novels_data:
                raise ValueError(f"Novel '{novel_key}' referenced in model '{model_key}' not found in novels.json")
            novel_data = self.novels_data[novel_key].copy()
            novels.append(novel_data)

        return novels

    def check_individual_novels_trained(self, novels: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Check which individual novels are already trained"""
        training_status = {}

        for novel_info in novels:
            directory_name = novel_info["directory_name"]
            model_path = self.models_dir / directory_name / "final"
            training_status[directory_name] = model_path.exists()

        return training_status

    def load_novel_content(self, directory_name: str) -> str:
        """Load content from a novel directory"""
        novel_path = self.novels_dir / directory_name

        if not novel_path.exists():
            raise FileNotFoundError(f"Novel directory not found: {novel_path}")

        content_files = ["content.txt", f"{directory_name}.txt", "novel.txt"]
        content = ""

        for filename in content_files:
            file_path = novel_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        logger.info(f"Loaded content from {file_path}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        if not content:
            raise ValueError(f"No readable content found in {novel_path}")

        return content

    def combine_novel_contents(self, novels: List[Dict[str, Any]], method: str = "concatenate") -> str:
        """Combine multiple novel contents into single corpus"""
        combined_content = []
        total_words = 0

        logger.info(f"Combining {len(novels)} novels using '{method}' method...")

        for i, novel_info in enumerate(novels):
            directory_name = novel_info["directory_name"]
            original_name = novel_info["original_name"]

            try:
                content = self.load_novel_content(directory_name)
                word_count = len(content.split())
                total_words += word_count

                logger.info(f"  {i+1}/{len(novels)}: {original_name} ({word_count:,} words)")

                if method == "concatenate":
                    if combined_content:
                        combined_content.append(self.config.novel_separator)
                    combined_content.append(f"=== {original_name} ===\n\n")
                    combined_content.append(content)

                elif method == "interleave":
                    combined_content.append(content)

                if total_words > self.config.max_combined_size:
                    logger.warning(f"Combined content ({total_words:,} words) exceeds limit ({self.config.max_combined_size:,})")
                    break

            except Exception as e:
                logger.error(f"Failed to load novel '{directory_name}': {e}")
                continue

        final_content = "\n".join(combined_content)
        logger.info(f"Combined corpus: {total_words:,} words, {len(final_content):,} characters")

        return final_content

    def train_model(self, model_key: str) -> Dict[str, Any]:
        """Train a model using one or more novels from model_mapping.json"""
        logger.info(f"Starting model training for: {model_key}")

        # Get novel list for this model
        novels = self.get_novels_for_model(model_key)
        if len(novels) < self.config.min_novels_required:
            raise ValueError(f"Model '{model_key}' has only {len(novels)} novels. Minimum required: {self.config.min_novels_required}")

        start_time = time.time()
        combined_content = self.combine_novel_contents(novels, self.config.combine_novels_method)

        model_output_dir = self.models_dir / model_key
        FileSystemUtils.ensure_directory(model_output_dir)

        total_words = sum(novel.get('word_count', 0) for novel in novels)
        adaptive_max_iterations = 50
        if total_words < 80000:  # tiny
            adaptive_steps_per_iteration = 125  # Reverted from 350 - was too slow
            adaptive_min_iterations = 5
            adaptive_lr_start = 3e-5
            adaptive_lr_end = 8e-6
            size_category = "tiny"
        elif total_words < 120000:  # small
            adaptive_steps_per_iteration = 150  # Reverted from 450
            adaptive_min_iterations = 8
            adaptive_lr_start = 2.5e-5
            adaptive_lr_end = 6e-6
            size_category = "small"
        elif total_words < 260000:  # medium/large
            adaptive_steps_per_iteration = 200  # Reverted from 550
            adaptive_min_iterations = 10
            adaptive_lr_start = 2e-5
            adaptive_lr_end = 5e-6
            size_category = "medium/large"
        else:  # xlarge
            adaptive_steps_per_iteration = 300  # Reverted from 700
            adaptive_min_iterations = 12
            adaptive_lr_start = 1.5e-5
            adaptive_lr_end = 3e-6
            size_category = "xlarge"

        logger.info(f"Using ADAPTIVE training parameters for {model_key}:")
        logger.info(f"  Word count: {total_words:,}")
        logger.info(f"  Size category: {size_category}")
        logger.info(f"  Iterations: {adaptive_min_iterations}-{adaptive_max_iterations}")
        logger.info(f"  Steps/iter: {adaptive_steps_per_iteration}")
        logger.info(f"  LR: {adaptive_lr_start:.2e} -> {adaptive_lr_end:.2e}")

        training_config = IterativeConfig(
            max_iterations=adaptive_max_iterations,
            min_iterations=adaptive_min_iterations,
            max_steps_per_iteration=adaptive_steps_per_iteration,
            learning_rate_start=adaptive_lr_start,
            learning_rate_end=adaptive_lr_end
        )

        trainer = IterativeTrainer(training_config)

        logger.info(f"Training model '{model_key}' with {len(novels)} novel(s)...")
        training_results = trainer._train_with_content(combined_content, model_key)

        training_results.update({
            "model_type": "combined",
            "model_key": model_key,
            "novels_included": [novel["original_name"] for novel in novels],
            "novel_count": len(novels),
            "total_training_time": time.time() - start_time,
            "combination_method": self.config.combine_novels_method
        })

        # Memory system disabled for performance
        # Skipping memory creation to improve training speed

        results_file = model_output_dir / "training_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, default=str)
            logger.info(f"Training results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")

        logger.info(f"Model training completed: {model_key}")
        return training_results

    def list_trained_models(self) -> List[str]:
        """List already trained models"""
        trained_models = []

        for model_key in self.get_available_models():
            model_path = self.models_dir / model_key / "final"
            if model_path.exists():
                trained_models.append(model_key)

        return trained_models

    def train_all_models(self) -> Dict[str, Any]:
        """Train all models in sequence"""
        results = {}
        available_models = self.get_available_models()

        logger.info(f"Training all {len(available_models)} models...")

        for i, model_key in enumerate(available_models):
            logger.info(f"\n=== Training Model {i+1}/{len(available_models)}: {model_key} ===")

            try:
                model_path = self.models_dir / model_key / "final"
                if model_path.exists():
                    logger.info(f"Model '{model_key}' already trained. Skipping.")
                    results[model_key] = {"status": "already_trained", "path": str(model_path)}
                    continue

                training_result = self.train_model(model_key)
                results[model_key] = {"status": "success", "results": training_result}

            except Exception as e:
                logger.error(f"Failed to train model '{model_key}': {e}")
                results[model_key] = {"status": "failed", "error": str(e)}

        return results


