#!/usr/bin/env python3
"""
Hybrid CPU/GPU Multi-Model Training System

Intelligent training system that adapts to available hardware, using GPU
when beneficial but falling back to CPU when necessary. Optimized for
consumer hardware with limited GPU memory.

Key Features:
- Smart GPU memory management with fallbacks
- Progressive model loading (offload to CPU when needed)
- Adaptive batch sizing based on available VRAM
- Memory-efficient gradient checkpointing
- Support for mixed precision when available
- Intelligent model selection based on hardware constraints
"""

import os
import sys
import gc
import psutil
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import re
from datasets import Dataset
import time
import warnings
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedHardwareManager:
    """Advanced hardware detection and memory management"""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_memory_gb = 0
        self.gpu_name = "None"
        self.mixed_precision_available = False

        if self.gpu_available:
            try:
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.gpu_name = torch.cuda.get_device_name(0)
                # Check for mixed precision support
                self.mixed_precision_available = torch.cuda.get_device_capability(0)[0] >= 7
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                self.gpu_available = False

        self.optimal_config = self._determine_optimal_config()

    def _determine_optimal_config(self) -> Dict[str, Any]:
        """Determine optimal configuration with intelligent fallbacks"""
        config = {
            "device_strategy": "cpu_only",
            "primary_device": "cpu",
            "offload_device": "cpu",
            "use_mixed_precision": False,
            "gradient_checkpointing": True,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 512,
            "model_max_memory": None,
            "cpu_threads": min(self.cpu_count, 8),
            "memory_fraction": 0.8
        }

        logger.info(f"Hardware Analysis:")
        logger.info(f"  CPU: {self.cpu_count} cores")
        logger.info(f"  System RAM: {self.system_memory_gb:.1f} GB")
        logger.info(f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)" if self.gpu_available else "  GPU: Not available")

        # System memory-based adjustments
        if self.system_memory_gb >= 32:
            config.update({
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_seq_length": 1024
            })
        elif self.system_memory_gb >= 16:
            config.update({
                "batch_size": 1,
                "gradient_accumulation_steps": 12,
                "max_seq_length": 768
            })

        # GPU-specific optimizations
        if self.gpu_available:
            if self.gpu_memory_gb >= 12:  # High-end GPU
                config.update({
                    "device_strategy": "gpu_primary",
                    "primary_device": "cuda",
                    "offload_device": "cpu",
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "use_mixed_precision": self.mixed_precision_available
                })
            elif self.gpu_memory_gb >= 8:  # Mid-range GPU
                config.update({
                    "device_strategy": "hybrid",
                    "primary_device": "cuda",
                    "offload_device": "cpu",
                    "batch_size": 2,
                    "gradient_accumulation_steps": 6,
                    "use_mixed_precision": self.mixed_precision_available,
                    "model_max_memory": {0: "6GB", "cpu": "16GB"}
                })
            elif self.gpu_memory_gb >= 4:  # Entry-level GPU
                config.update({
                    "device_strategy": "cpu_with_gpu_assist",
                    "primary_device": "cpu",
                    "offload_device": "cuda",
                    "batch_size": 1,
                    "gradient_accumulation_steps": 12,
                    "model_max_memory": {0: "3GB", "cpu": "12GB"}
                })

        return config

    @contextmanager
    def memory_management(self):
        """Context manager for automatic memory cleanup"""
        try:
            yield
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_available_memory(self) -> Tuple[float, float]:
        """Get available system and GPU memory in GB"""
        system_available = psutil.virtual_memory().available / (1024**3)
        gpu_available = 0

        if torch.cuda.is_available():
            try:
                gpu_available = (torch.cuda.get_device_properties(0).total_memory -
                               torch.cuda.memory_allocated(0)) / (1024**3)
            except:
                gpu_available = 0

        return system_available, gpu_available

@dataclass
class HybridModelConfig:
    """Configuration for hybrid CPU/GPU training"""

    # Model options by size
    tiny_models = [
        "microsoft/DialoGPT-small",  # 117M
        "distilgpt2",  # 82M
    ]
    small_models = [
        "microsoft/DialoGPT-medium",  # 345M
        "gpt2",  # 124M
    ]
    medium_models = [
        "microsoft/DialoGPT-large",  # 762M
        "gpt2-medium",  # 345M
    ]
    large_models = [
        "gpt2-large",  # 774M
        "gpt2-xl",  # 1.5B
    ]

    # Training parameters
    learning_rate: float = 3e-5
    max_steps: int = 100
    warmup_steps: int = 20
    save_steps: int = 50
    eval_steps: int = 25

    # Infrastructure
    novels_dir: str = "novels"
    experiments_dir: str = "hybrid_model_experiments"
    mapping_file: str = "model_mapping.json"

class MemoryEfficientProcessor:
    """Memory-efficient novel processing with adaptive chunking"""

    def __init__(self, novel_dir: Path, config: Dict[str, Any]):
        self.novel_dir = novel_dir
        self.config = config
        self.title = novel_dir.name.replace('_', ' ').title()

    def load_and_process(self) -> Dict[str, Any]:
        """Load novel with memory-efficient processing"""
        text_files = list(self.novel_dir.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No .txt file found in {self.novel_dir}")

        # Load files
        novel_file = text_files[0]
        analysis_file = self.novel_dir / "analysis.json"

        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

        # Stream-process large files
        chunks = self._stream_process_novel(novel_file)

        return {
            "title": self.title,
            "word_count": analysis["word_count"],
            "analysis": analysis,
            "training_chunks": chunks,
            "chunk_count": len(chunks)
        }

    def _stream_process_novel(self, novel_file: Path) -> List[str]:
        """Stream process novel to avoid loading entire file into memory"""
        max_length = self.config["max_seq_length"]
        chunks = []
        current_chunk = ""

        with open(novel_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check if adding this line would exceed max length
                if len((current_chunk + " " + line).split()) > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = line
                    else:
                        # Line itself is too long, truncate it
                        words = line.split()[:max_length]
                        chunks.append(' '.join(words))
                        current_chunk = ""
                else:
                    current_chunk += " " + line if current_chunk else line

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Limit chunks based on memory constraints
        max_chunks = min(len(chunks), 15)  # Conservative limit
        if len(chunks) > max_chunks:
            # Take evenly distributed chunks
            step = len(chunks) // max_chunks
            chunks = [chunks[i] for i in range(0, len(chunks), step)][:max_chunks]

        return chunks

class HybridTrainer:
    """Hybrid CPU/GPU trainer with intelligent resource management"""

    def __init__(self, config: HybridModelConfig):
        self.config = config
        self.hardware = AdvancedHardwareManager()
        self.hw_config = self.hardware.optimal_config

        # Setup directories
        self.novels_dir = Path(config.novels_dir)
        self.experiments_dir = Path(config.experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)

        # Configure PyTorch
        torch.set_num_threads(self.hw_config["cpu_threads"])

        # Load model mapping
        with open(config.mapping_file, 'r') as f:
            self.model_mapping = json.load(f)

    def select_optimal_model(self) -> str:
        """Select model based on available resources"""
        system_mem, gpu_mem = self.hardware.get_available_memory()

        logger.info(f"Available memory - System: {system_mem:.1f}GB, GPU: {gpu_mem:.1f}GB")

        # Model selection logic
        if gpu_mem >= 8 and system_mem >= 16:
            models = self.config.large_models
            size = "large"
        elif gpu_mem >= 4 or system_mem >= 12:
            models = self.config.medium_models
            size = "medium"
        elif gpu_mem >= 2 or system_mem >= 8:
            models = self.config.small_models
            size = "small"
        else:
            models = self.config.tiny_models
            size = "tiny"

        selected = models[0]  # Take first available
        logger.info(f"Selected {size} model: {selected}")
        return selected

    def train_models_with_hybrid_approach(self) -> Dict[str, Any]:
        """Train models using hybrid CPU/GPU approach"""

        # Import with fallback
        try:
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TrainingArguments, Trainer,
                DataCollatorForLanguageModeling
            )
            from accelerate import Accelerator
        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            return {"error": "Missing dependencies"}

        model_name = self.select_optimal_model()
        results = {}

        # Train subset of models based on resources
        available_mem = self.hardware.get_available_memory()[0]
        max_models = min(5, max(1, int(available_mem // 4)))  # Rough estimate

        model_subset = list(self.model_mapping["models"].items())[:max_models]
        logger.info(f"Training {len(model_subset)} models with hybrid approach...")

        for model_id, model_info in model_subset:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training Model: {model_id}")
            logger.info(f"Novels: {model_info['novel_count']}")

            try:
                with self.hardware.memory_management():
                    result = self._train_single_model_hybrid(
                        model_id, model_info, model_name
                    )
                    results[model_id] = result

            except Exception as e:
                logger.error(f"Failed to train {model_id}: {e}")
                results[model_id] = {"error": str(e)}

        # Compile final results
        successful = len([r for r in results.values() if "error" not in r])

        final_results = {
            "training_summary": {
                "total_models": len(model_subset),
                "successful": successful,
                "hardware_strategy": self.hw_config["device_strategy"],
                "base_model": model_name,
                "hardware_config": self.hw_config
            },
            "individual_results": results
        }

        # Save results
        with open(self.experiments_dir / "hybrid_training_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)

        return final_results

    def _train_single_model_hybrid(self, model_id: str, model_info: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Train single model using hybrid approach"""

        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer,
            DataCollatorForLanguageModeling
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with device map for hybrid training
        device_map = None
        if self.hw_config["device_strategy"] == "hybrid":
            device_map = "auto"
        elif self.hw_config["device_strategy"] == "gpu_primary":
            device_map = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.hw_config["use_mixed_precision"] else torch.float32,
            device_map=device_map,
            max_memory=self.hw_config.get("model_max_memory"),
            low_cpu_mem_usage=True
        )

        # Enable gradient checkpointing for memory efficiency
        if self.hw_config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()

        # Prepare dataset
        dataset = self._prepare_hybrid_dataset(model_info, tokenizer)
        if not dataset:
            raise ValueError("No dataset prepared")

        # Configure training arguments
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
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=True,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            optim="adamw_torch",
            fp16=self.hw_config["use_mixed_precision"],
            dataloader_pin_memory=False,  # Reduce memory pressure
            report_to="none"
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

        # Train with memory monitoring
        logger.info(f"Starting hybrid training for {model_id}...")
        start_time = time.time()

        trainer.train()

        training_time = time.time() - start_time

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(str(self.experiments_dir / model_id))

        # Generate sample
        sample = self._generate_sample_hybrid(model, tokenizer, model_id)

        return {
            "training_time": training_time,
            "dataset_size": len(dataset),
            "model_name": model_name,
            "device_strategy": self.hw_config["device_strategy"],
            "sample": sample,
            "memory_usage": self._get_memory_usage()
        }

    def _prepare_hybrid_dataset(self, model_info: Dict[str, Any], tokenizer) -> Optional[Dataset]:
        """Prepare dataset with memory-efficient processing"""
        all_texts = []

        # Process limited number of novels
        novels_to_process = min(3, len(model_info["novels"]))

        for novel_info in model_info["novels"][:novels_to_process]:
            novel_dir = self.novels_dir / novel_info["directory_name"]

            if not novel_dir.exists():
                continue

            try:
                processor = MemoryEfficientProcessor(novel_dir, self.hw_config)
                novel_data = processor.load_and_process()

                # Limit chunks per novel
                chunks = novel_data["training_chunks"][:8]
                all_texts.extend(chunks)

                logger.info(f"    {novel_data['title']}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {novel_dir}: {e}")
                continue

        if not all_texts:
            return None

        logger.info(f"  Total training texts: {len(all_texts)}")

        # Tokenize efficiently
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.hw_config["max_seq_length"]
            )

        dataset = Dataset.from_dict({"text": all_texts})

        # Process in smaller batches to avoid memory issues
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Small batch size
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        return tokenized_dataset

    def _generate_sample_hybrid(self, model, tokenizer, model_id: str) -> str:
        """Generate sample with hybrid device management"""
        model.eval()

        prompt = f"Write a story in the style of {model_id}:"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to appropriate device
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        elif torch.cuda.is_available() and self.hw_config["primary_device"] == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=80,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        system_mem = psutil.virtual_memory()
        usage = {
            "system_used_gb": (system_mem.total - system_mem.available) / (1024**3),
            "system_total_gb": system_mem.total / (1024**3)
        }

        if torch.cuda.is_available():
            usage.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                "gpu_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })

        return usage

def main():
    """Main function with comprehensive error handling"""
    print("Hybrid CPU/GPU Multi-Model Training System")
    print("=" * 50)

    # Dependency check
    try:
        import transformers
        import accelerate
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ Accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Install with: pip install transformers accelerate datasets torch")
        return

    config = HybridModelConfig()

    # Check mapping file
    if not Path(config.mapping_file).exists():
        print(f"✗ Error: Mapping file {config.mapping_file} not found.")
        print("Run: python mapping_generator.py first")
        return

    trainer = HybridTrainer(config)

    try:
        results = trainer.train_models_with_hybrid_approach()

        print(f"\n{'='*50}")
        print("Hybrid Training Summary:")
        print(f"Strategy: {results['training_summary']['hardware_strategy']}")
        print(f"Models trained: {results['training_summary']['successful']}/{results['training_summary']['total_models']}")
        print(f"Base model: {results['training_summary']['base_model']}")
        print(f"Results saved to: {config.experiments_dir}")

        if results['training_summary']['successful'] > 0:
            print("\n✓ Hybrid training completed successfully!")
            print("Models are optimized for your specific hardware configuration.")
        else:
            print("\n⚠ Training encountered issues. Check logs above.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n✗ Training failed: {e}")

if __name__ == "__main__":
    main()