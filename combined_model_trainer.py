#!/usr/bin/env python3
"""
Model Tea - Combined Model Training System
Copyright © ChaiQ LLC

Trains combined models using multiple novels from model_mapping.json.
Applies the same 12-iteration progressive learning approach to combined corpus.
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from datasets import Dataset

# Targeted warning suppression for known issues only
warnings.filterwarnings("ignore", message=".*Using the model-agnostic default.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resume_download is deprecated.*", category=FutureWarning)

from model_tea_utils import (
    ModelTeaConfig, FileSystemUtils, TextProcessingUtils, TrainingUtils,
    QualityMetrics, MemorySystemUtils, ErrorHandling, validate_system_setup
)
from quality_validator import QualityValidator, ValidationConfig

try:
    from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    EpisodicMemorySystem = None
    MemoryConfig = None
    MEMORY_SYSTEM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CombinedModelConfig:
    """Configuration for combined model training"""
    base_config: ModelTeaConfig = None

    combine_novels_method: str = "concatenate"
    combine_novels_method: str = "concatenate"
    novel_separator: str = "\n\n=== NEW NOVEL ===\n\n"
    min_novels_required: int = 2
    max_combined_size: int = 2000000  # 2M words max

    max_iterations: int = 12
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 5e-6

    combined_memories_count: int = 350
    cross_novel_memories: bool = True

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = ModelTeaConfig()


class CombinedModelTrainer:
    """
    Trains combined models using multiple novels from model_mapping.json
    """

    def __init__(self, config: CombinedModelConfig = None):
        self.config = config or CombinedModelConfig()
        self.model_mapping = self._load_model_mapping()
        self.quality_validator = QualityValidator(ValidationConfig())

        self.novels_dir = Path("novels")
        self.models_dir = Path("iterative_models")

        validate_system_setup()

    def _load_model_mapping(self) -> Dict[str, Any]:
        """Load model mapping configuration"""
        mapping_file = Path("model_mapping.json")
        if not mapping_file.exists():
            raise FileNotFoundError("model_mapping.json not found. Required for combined model training.")

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model_mapping.json: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available combined models"""
        if "models" not in self.model_mapping:
            return []
        return list(self.model_mapping["models"].keys())

    def get_novels_for_model(self, model_key: str) -> List[Dict[str, Any]]:
        """Get novel list for specified combined model"""
        if "models" not in self.model_mapping:
            raise ValueError("Invalid model_mapping.json format")

        if model_key not in self.model_mapping["models"]:
            available = ", ".join(self.get_available_models())
            raise ValueError(f"Model '{model_key}' not found. Available models: {available}")

        model_info = self.model_mapping["models"][model_key]
        return model_info.get("novels", [])

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

    def train_combined_model(self, model_key: str) -> Dict[str, Any]:
        """Train a combined model using multiple novels"""
        logger.info(f"Starting combined model training for: {model_key}")

        # Get novel list for this model
        novels = self.get_novels_for_model(model_key)
        if len(novels) < self.config.min_novels_required:
            raise ValueError(f"Model '{model_key}' has only {len(novels)} novels. Minimum required: {self.config.min_novels_required}")

        training_status = self.check_individual_novels_trained(novels)
        untrained_count = sum(1 for trained in training_status.values() if not trained)
        if untrained_count > 0:
            logger.warning(f"{untrained_count} individual novels are not yet trained. This is optional but recommended.")

        start_time = time.time()
        combined_content = self.combine_novel_contents(novels, self.config.combine_novels_method)

        model_output_dir = self.models_dir / model_key
        FileSystemUtils.ensure_directory(model_output_dir)

        from iterative_novel_trainer import IterativeTrainer, IterativeConfig

        training_config = IterativeConfig()
        training_config.max_iterations = self.config.max_iterations
        training_config.learning_rate_start = self.config.learning_rate_start
        training_config.learning_rate_end = self.config.learning_rate_end

        trainer = IterativeTrainer(training_config)

        logger.info(f"Training combined model '{model_key}' with {len(novels)} novels...")
        training_results = trainer._train_with_content(combined_content, model_key)

        training_results.update({
            "model_type": "combined",
            "model_key": model_key,
            "novels_included": [novel["original_name"] for novel in novels],
            "novel_count": len(novels),
            "total_training_time": time.time() - start_time,
            "combination_method": self.config.combine_novels_method
        })

        if MEMORY_SYSTEM_AVAILABLE:
            self._create_combined_memories(model_key, combined_content, novels, training_results)

        results_file = model_output_dir / "training_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, default=str)
            logger.info(f"Training results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")

        logger.info(f"Combined model training completed: {model_key}")
        return training_results

    def _create_combined_memories(self, model_key: str, content: str, novels: List[Dict], training_results: Dict):
        """Create enhanced memory system for combined model"""
        if not MEMORY_SYSTEM_AVAILABLE:
            logger.warning("Episodic memory system not available - skipping combined memory creation")
            return

        try:
            logger.info(f"Creating enhanced memory system for combined model: {model_key}")

            memory_config = MemoryConfig(
                max_memories_per_novel=self.config.combined_memories_count,
                memory_chunk_size=35,
                extract_characters=True,
                extract_locations=True,
                extract_emotions=True,
                extract_themes=True,
                extract_dialogue=True,
                extract_descriptions=True
            )

            memory_system = EpisodicMemorySystem(memory_config)

            temp_novel_path = self.models_dir / model_key / "temp_combined.txt"
            temp_novel_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_novel_path, 'w', encoding='utf-8') as f:
                f.write(content)

            memory_analysis = memory_system.build_memory_for_model(model_key, temp_novel_path.parent)

            if temp_novel_path.exists():
                temp_novel_path.unlink()

            training_results["combined_memory_analysis"] = memory_analysis

            logger.info(f"Enhanced memory system created with {memory_analysis.get('total_memories', 0)} memories")

        except Exception as e:
            logger.error(f"Failed to create enhanced memory system: {e}")

    def _create_cross_novel_memories(self, novels: List[Dict], content: str) -> List[Dict]:
        """Create cross-novel relationship memories"""
        cross_memories = []

        for i, novel1 in enumerate(novels):
            for novel2 in novels[i+1:]:
                cross_memory = {
                    "type": "cross_novel_reference",
                    "novel1": novel1["original_name"],
                    "novel2": novel2["original_name"],
                    "relationship": "appears_in_same_model"
                }
                cross_memories.append(cross_memory)

        return cross_memories

    def list_trained_combined_models(self) -> List[str]:
        """List already trained combined models"""
        trained_models = []

        for model_key in self.get_available_models():
            model_path = self.models_dir / model_key / "final"
            if model_path.exists():
                trained_models.append(model_key)

        return trained_models

    def train_all_combined_models(self) -> Dict[str, Any]:
        """Train all combined models in sequence"""
        results = {}
        available_models = self.get_available_models()

        logger.info(f"Training all {len(available_models)} combined models...")

        for i, model_key in enumerate(available_models):
            logger.info(f"\n=== Training Model {i+1}/{len(available_models)}: {model_key} ===")

            try:
                model_path = self.models_dir / model_key / "final"
                if model_path.exists():
                    logger.info(f"Model '{model_key}' already trained. Skipping.")
                    results[model_key] = {"status": "already_trained", "path": str(model_path)}
                    continue

                training_result = self.train_combined_model(model_key)
                results[model_key] = {"status": "success", "results": training_result}

            except Exception as e:
                logger.error(f"Failed to train model '{model_key}': {e}")
                results[model_key] = {"status": "failed", "error": str(e)}

        return results


def main():
    """Main entry point for combined model training"""
    parser = argparse.ArgumentParser(description="Model Tea - Combined Model Training")
    parser.add_argument("--model", type=str, help="Specific model to train (e.g., vs_mintchip)")
    parser.add_argument("--all-models", action="store_true", help="Train all combined models")
    parser.add_argument("--list-models", action="store_true", help="List available combined models")
    parser.add_argument("--list-trained", action="store_true", help="List already trained models")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = CombinedModelTrainer()

        if args.list_models:
            models = trainer.get_available_models()
            print(f"\nAvailable Combined Models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
            return

        if args.list_trained:
            trained = trainer.list_trained_combined_models()
            print(f"\nTrained Combined Models ({len(trained)}):")
            for model in trained:
                print(f"  - {model}")
            return

        if args.all_models:
            print("Training all combined models...")
            results = trainer.train_all_combined_models()

            success_count = sum(1 for r in results.values() if r["status"] == "success")
            already_trained_count = sum(1 for r in results.values() if r["status"] == "already_trained")
            failed_count = sum(1 for r in results.values() if r["status"] == "failed")

            print(f"\n=== Training Summary ===")
            print(f"Success: {success_count}")
            print(f"Already trained: {already_trained_count}")
            print(f"Failed: {failed_count}")

        elif args.model:
            model_path = trainer.models_dir / args.model / "final"
            if model_path.exists() and not args.force:
                print(f"Model '{args.model}' is already trained. Use --force to retrain.")
                return

            print(f"Training combined model: {args.model}")
            result = trainer.train_combined_model(args.model)
            print(f"Training completed successfully for {args.model}")

        else:
            parser.print_help()

            models = trainer.get_available_models()
            print(f"\nAvailable models: {', '.join(models)}")
            print(f"\nExample usage:")
            print(f"  python combined_model_trainer.py --model vs_mintchip")
            print(f"  python combined_model_trainer.py --all-models")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()