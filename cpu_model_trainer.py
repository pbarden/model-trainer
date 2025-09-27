#!/usr/bin/env python3
"""
CPU-Optimized Multi-Model Training System

A CPU-friendly version of the multi-model trainer that can work with limited
GPU resources or entirely on CPU. Features intelligent hardware detection,
automatic fallbacks, and optimized memory management.

Key Features:
- Automatic hardware detection and configuration
- CPU-only training with optional GPU acceleration
- Memory-efficient processing for consumer hardware
- Graceful degradation based on available resources
- Support for smaller models (1B-7B parameters)
"""

import os
import sys
import psutil
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from datasets import Dataset
import time
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detects and configures optimal settings based on available hardware"""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        self.gpu_memory_gb = 0
        self.gpu_name = "None"

        if self.gpu_available:
            try:
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.gpu_name = torch.cuda.get_device_name(0)
            except:
                self.gpu_available = False

    def get_optimal_config(self) -> Dict[str, Any]:
        """Determine optimal configuration based on hardware"""
        config = {
            "device": "cpu",
            "use_gpu": False,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 512,
            "model_size": "small",
            "quantization": "none",
            "cpu_threads": min(self.cpu_count, 8),
            "memory_fraction": 0.7
        }

        logger.info(f"Hardware Detection:")
        logger.info(f"  CPU: {self.cpu_count} cores")
        logger.info(f"  RAM: {self.memory_gb:.1f} GB")
        logger.info(f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)" if self.gpu_available else "  GPU: Not available")

        # Configure based on available memory
        if self.memory_gb >= 32:
            config.update({
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_seq_length": 1024,
                "model_size": "medium"
            })
        elif self.memory_gb >= 16:
            config.update({
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 768,
                "model_size": "small"
            })
        else:
            config.update({
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "max_seq_length": 512,
                "model_size": "tiny"
            })

        # Enable GPU if available and has sufficient memory
        if self.gpu_available and self.gpu_memory_gb >= 4:
            config.update({
                "device": "cuda",
                "use_gpu": True,
                "batch_size": min(config["batch_size"] * 2, 4)
            })

            if self.gpu_memory_gb >= 8:
                config["quantization"] = "8bit"
            elif self.gpu_memory_gb >= 6:
                config["quantization"] = "4bit"

        return config

@dataclass
class CPUModelConfig:
    """
    Configuration for CPU-optimized training with automatic hardware adaptation
    """
    # Model selection based on hardware
    tiny_model: str = "microsoft/DialoGPT-small"  # 117M parameters
    small_model: str = "microsoft/DialoGPT-medium"  # 345M parameters
    medium_model: str = "microsoft/DialoGPT-large"  # 762M parameters
    large_model: str = "gpt2-xl"  # 1.5B parameters

    # Training parameters
    learning_rate: float = 5e-5
    max_steps: int = 50  # Reduced for CPU training
    warmup_steps: int = 10
    save_steps: int = 25

    # Memory management
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = True
    optim: str = "adamw_torch"  # CPU-friendly optimizer

    # Infrastructure
    novels_dir: str = "novels"
    experiments_dir: str = "cpu_model_experiments"
    mapping_file: str = "model_mapping.json"

class CPUNovelProcessor:
    """Memory-efficient novel processing for CPU training"""

    def __init__(self, novel_dir: Path, config: Dict[str, Any]):
        self.novel_dir = novel_dir
        self.config = config
        self.title = novel_dir.name.replace('_', ' ').title()

    def load_and_process(self) -> Dict[str, Any]:
        """Load and process novel with memory-efficient chunking"""
        text_files = list(self.novel_dir.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No .txt file found in {self.novel_dir}")

        novel_file = text_files[0]
        analysis_file = self.novel_dir / "analysis.json"

        # Load analysis
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        # Load content with memory management
        with open(novel_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create memory-efficient chunks
        chunks = self._create_memory_efficient_chunks(content)

        return {
            "title": self.title,
            "word_count": analysis["word_count"],
            "analysis": analysis,
            "training_chunks": chunks,
            "chunk_count": len(chunks),
            "chunking_info": {
                "max_seq_length": self.config["max_seq_length"],
                "chunk_count": len(chunks)
            }
        }

    def _create_memory_efficient_chunks(self, text: str) -> List[str]:
        """Create smaller chunks optimized for CPU training"""
        max_length = self.config["max_seq_length"]
        overlap = max_length // 4

        # Split into sentences
        sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', text) if s.strip()]

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > max_length and current_chunk:
                chunks.append(current_chunk.strip())

                # Add overlap
                if overlap > 0:
                    overlap_text = ' '.join(current_chunk.split()[-overlap:])
                    current_chunk = overlap_text + " " + sentence
                    current_words = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_words = sentence_words
            else:
                current_chunk += " " + sentence
                current_words += sentence_words

        if current_chunk and current_words > 20:
            chunks.append(current_chunk.strip())

        # Limit chunks for memory management
        max_chunks = min(len(chunks), 20)  # Reduced for CPU training
        if len(chunks) > max_chunks:
            step = len(chunks) // max_chunks
            chunks = [chunks[i] for i in range(0, len(chunks), step)][:max_chunks]

        return chunks

class CPUTrainer:
    """CPU-optimized trainer with optional GPU acceleration"""

    def __init__(self, config: CPUModelConfig):
        self.config = config
        self.hardware = HardwareDetector()
        self.hw_config = self.hardware.get_optimal_config()

        # Set up directories
        self.novels_dir = Path(config.novels_dir)
        self.experiments_dir = Path(config.experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

        # Configure torch for optimal CPU performance
        torch.set_num_threads(self.hw_config["cpu_threads"])

        # Load model mapping
        with open(config.mapping_file, 'r') as f:
            self.model_mapping = json.load(f)

    def select_model(self) -> str:
        """Select appropriate model based on hardware"""
        size = self.hw_config["model_size"]

        models = {
            "tiny": self.config.tiny_model,
            "small": self.config.small_model,
            "medium": self.config.medium_model,
            "large": self.config.large_model
        }

        selected = models.get(size, self.config.small_model)
        logger.info(f"Selected model: {selected} (size: {size})")
        return selected

    def train_all_models(self) -> Dict[str, Any]:
        """Train models with CPU optimization"""

        # Import training libraries with error handling
        try:
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TrainingArguments, Trainer,
                DataCollatorForLanguageModeling
            )
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            logger.error("Install with: pip install transformers datasets torch")
            return {"error": "Missing dependencies"}

        model_name = self.select_model()
        results = {}

        # Limit to first 3 models for CPU training demonstration
        model_subset = list(self.model_mapping["models"].items())[:3]

        logger.info(f"Training {len(model_subset)} models for CPU demonstration...")

        for model_id, model_info in model_subset:
            logger.info(f"\nTraining Model: {model_id}")
            logger.info(f"Description: {model_info['description']}")
            logger.info(f"Novels: {model_info['novel_count']}")

            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU-friendly
                    device_map=self.hw_config["device"] if self.hw_config["use_gpu"] else None
                )

                # Process novels for this model
                dataset = self._prepare_cpu_dataset(model_info, tokenizer)

                if not dataset:
                    logger.warning(f"No data prepared for {model_id}")
                    continue

                # Configure training arguments for CPU
                training_args = TrainingArguments(
                    output_dir=str(self.experiments_dir / model_id),
                    overwrite_output_dir=True,
                    num_train_epochs=1,
                    max_steps=self.config.max_steps,
                    per_device_train_batch_size=self.hw_config["batch_size"],
                    gradient_accumulation_steps=self.hw_config["gradient_accumulation_steps"],
                    warmup_steps=self.config.warmup_steps,
                    learning_rate=self.config.learning_rate,
                    logging_steps=10,
                    save_steps=self.config.save_steps,
                    save_total_limit=2,
                    prediction_loss_only=True,
                    remove_unused_columns=self.config.remove_unused_columns,
                    dataloader_num_workers=self.config.dataloader_num_workers,
                    optim=self.config.optim,
                    fp16=False,  # Disabled for CPU compatibility
                    report_to="none",
                    use_cpu=not self.hw_config["use_gpu"]
                )

                # Data collator
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )

                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    data_collator=data_collator
                )

                # Train
                logger.info(f"Starting training for {model_id}...")
                start_time = time.time()

                trainer.train()

                training_time = time.time() - start_time

                # Save model
                trainer.save_model()
                tokenizer.save_pretrained(str(self.experiments_dir / model_id))

                # Generate sample
                sample = self._generate_sample(model, tokenizer, model_id)

                results[model_id] = {
                    "training_time": training_time,
                    "dataset_size": len(dataset),
                    "model_name": model_name,
                    "hardware_config": self.hw_config,
                    "sample": sample,
                    "novels": [n["original_name"] for n in model_info["novels"][:3]]
                }

                logger.info(f"✓ {model_id} completed in {training_time:.1f}s")

                # Clean up memory
                del model, trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                logger.error(f"✗ {model_id} failed: {e}")
                results[model_id] = {"error": str(e)}

        # Save results
        final_results = {
            "training_summary": {
                "total_models": len(model_subset),
                "successful": len([r for r in results.values() if "error" not in r]),
                "hardware_config": self.hw_config,
                "base_model": model_name
            },
            "individual_results": results
        }

        with open(self.experiments_dir / "cpu_training_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)

        return final_results

    def _prepare_cpu_dataset(self, model_info: Dict[str, Any], tokenizer) -> Optional[Dataset]:
        """Prepare dataset optimized for CPU training"""
        all_texts = []

        # Process novels (limit to first 3 for CPU demo)
        for novel_info in model_info["novels"][:3]:
            novel_dir = self.novels_dir / novel_info["directory_name"]

            if not novel_dir.exists():
                logger.warning(f"Novel directory {novel_dir} not found")
                continue

            try:
                processor = CPUNovelProcessor(novel_dir, self.hw_config)
                novel_data = processor.load_and_process()

                # Take smaller chunks for CPU training
                chunks = novel_data["training_chunks"][:5]  # Limit chunks
                all_texts.extend(chunks)

                logger.info(f"    {novel_data['title']}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {novel_dir}: {e}")
                continue

        if not all_texts:
            return None

        logger.info(f"  Total texts for training: {len(all_texts)}")

        # Tokenize with truncation
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.hw_config["max_seq_length"]
            )

        # Create dataset
        dataset = Dataset.from_dict({"text": all_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        return tokenized_dataset

    def _generate_sample(self, model, tokenizer, model_id: str) -> str:
        """Generate a sample text to verify training"""
        model.eval()

        prompt = f"Write a short story in the style of {model_id}:"
        inputs = tokenizer(prompt, return_tensors="pt")

        if self.hw_config["use_gpu"]:
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()

def main():
    """Main function with comprehensive error handling"""
    print("CPU-Optimized Multi-Model Training System")
    print("=" * 50)

    # Check dependencies
    try:
        import transformers
        import datasets
        print(f"[OK] Transformers version: {transformers.__version__}")
        print(f"[OK] Datasets version: {datasets.__version__}")
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("Install with: pip install transformers datasets torch")
        return

    config = CPUModelConfig()

    # Check if mapping file exists
    if not Path(config.mapping_file).exists():
        print(f"[ERROR] Mapping file {config.mapping_file} not found.")
        print("Run: python mapping_generator.py first")
        return

    trainer = CPUTrainer(config)

    try:
        results = trainer.train_all_models()

        print(f"\n{'='*50}")
        print("Training Summary:")
        print(f"Models trained: {results['training_summary']['successful']}/{results['training_summary']['total_models']}")
        print(f"Base model: {results['training_summary']['base_model']}")
        print(f"Hardware: {results['training_summary']['hardware_config']['device'].upper()}")
        print(f"Results saved to: {config.experiments_dir}")

        if results['training_summary']['successful'] > 0:
            print("\n[SUCCESS] CPU training completed successfully!")
            print("Your models are ready for inference and further development.")
        else:
            print("\n[WARNING] No models were successfully trained.")
            print("Check the error messages above for troubleshooting.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n[ERROR] Training failed: {e}")

if __name__ == "__main__":
    main()