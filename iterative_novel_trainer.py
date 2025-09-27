#!/usr/bin/env python3
"""
Iterative Novel Training System

Fast CPU-optimized trainer that focuses on one novel at a time with multiple
training iterations to develop fluency rather than memorization. Uses progressive
difficulty and validation to ensure quality learning.
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np
from datasets import Dataset
import random

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IterativeConfig:
    """Configuration for iterative novel training"""

    # Model selection (CPU-optimized)
    base_model: str = "distilgpt2"  # Fast, lightweight model
    max_seq_length: int = 256      # Reduced for speed

    # Training progression
    iterations_per_novel: int = 5   # Multiple training passes
    max_steps_per_iteration: int = 20  # Short iterations
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 1e-5   # Learning rate decay

    # Data management
    chunk_size: int = 200          # Smaller chunks
    chunk_overlap: int = 50        # Overlap for context
    validation_split: float = 0.2  # Hold out for validation

    # Training optimization
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1

    # Quality control
    max_repetition_penalty: float = 1.2
    temperature_range: tuple = (0.7, 1.0)  # For validation generation
    perplexity_threshold: float = 50.0      # Stop if model gets too confused

    # Infrastructure
    novels_dir: str = "novels"
    output_dir: str = "iterative_models"
    save_checkpoints: bool = True

class NovelProcessor:
    """Processes novels for iterative training"""

    def __init__(self, config: IterativeConfig):
        self.config = config

    def load_novel(self, novel_path: Path) -> Dict[str, Any]:
        """Load and analyze a single novel"""
        text_files = list(novel_path.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No text file in {novel_path}")

        # Load novel content
        with open(text_files[0], 'r', encoding='utf-8') as f:
            content = f.read()

        # Load analysis if available
        analysis_file = novel_path / "analysis.json"
        analysis = {}
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

        return {
            "title": novel_path.name.replace('_', ' ').title(),
            "content": content,
            "word_count": analysis.get("word_count", len(content.split())),
            "path": novel_path
        }

    def create_progressive_chunks(self, content: str, iteration: int) -> List[str]:
        """Create chunks with progressive difficulty"""
        # Start with easier (shorter) chunks, progress to longer ones
        base_size = self.config.chunk_size
        progression_factor = 1 + (iteration * 0.2)  # 20% increase per iteration
        current_chunk_size = int(base_size * progression_factor)

        # Split into sentences for better boundaries
        sentences = self._split_sentences(content)

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > current_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Add overlap
                overlap_words = self.config.chunk_overlap
                if overlap_words > 0:
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-overlap_words:])
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

        logger.info(f"Created {len(chunks)} chunks (avg {current_chunk_size} words)")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]

    def create_train_val_split(self, chunks: List[str]) -> tuple:
        """Split chunks into training and validation sets"""
        random.shuffle(chunks)
        split_idx = int(len(chunks) * (1 - self.config.validation_split))

        train_chunks = chunks[:split_idx]
        val_chunks = chunks[split_idx:]

        logger.info(f"Split: {len(train_chunks)} train, {len(val_chunks)} validation")
        return train_chunks, val_chunks

class QualityValidator:
    """Validates training quality and prevents overfitting"""

    def __init__(self, config: IterativeConfig):
        self.config = config
        self.perplexity_history = []
        self.generation_quality_history = []

    def calculate_perplexity(self, model, tokenizer, val_texts: List[str]) -> float:
        """Calculate perplexity on validation set"""
        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in val_texts[:5]:  # Sample for speed
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.config.max_seq_length)

                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                total_loss += loss.item() * inputs.input_ids.size(1)
                total_tokens += inputs.input_ids.size(1)

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        self.perplexity_history.append(perplexity)

        return perplexity

    def test_generation_quality(self, model, tokenizer, prompt: str) -> Dict[str, Any]:
        """Test generation quality with various parameters"""
        model.eval()

        prompts_to_test = [
            f"In the style of this novel: {prompt}",
            "The character walked into the room and",
            "It was a dark and stormy night when"
        ]

        quality_scores = []

        for test_prompt in prompts_to_test:
            inputs = tokenizer(test_prompt, return_tensors="pt")

            with torch.no_grad():
                for temp in self.config.temperature_range:
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        temperature=temp,
                        do_sample=True,
                        repetition_penalty=self.config.max_repetition_penalty,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    quality_score = self._assess_text_quality(generated)
                    quality_scores.append(quality_score)

        avg_quality = np.mean(quality_scores)
        self.generation_quality_history.append(avg_quality)

        return {
            "average_quality": avg_quality,
            "sample_generation": generated,
            "improving": len(self.generation_quality_history) < 2 or
                       avg_quality > self.generation_quality_history[-2]
        }

    def _assess_text_quality(self, text: str) -> float:
        """Simple heuristic for text quality"""
        if not text or len(text) < 10:
            return 0.0

        # Check for repetition
        words = text.lower().split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words) if words else 0

        # Check for reasonable sentence structure
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        sentence_score = min(avg_sentence_length / 15, 1.0)  # Prefer 10-20 word sentences

        # Combine scores
        quality_score = (repetition_ratio * 0.6) + (sentence_score * 0.4)
        return quality_score

    def should_continue_training(self, current_iteration: int) -> bool:
        """Decide if training should continue"""
        if current_iteration == 0:
            return True

        # Check if perplexity is getting too high (overfitting)
        if self.perplexity_history and self.perplexity_history[-1] > self.config.perplexity_threshold:
            logger.warning(f"Stopping: perplexity too high ({self.perplexity_history[-1]:.2f})")
            return False

        # Check if quality stopped improving
        if len(self.generation_quality_history) >= 3:
            recent_scores = self.generation_quality_history[-3:]
            if all(score <= recent_scores[0] for score in recent_scores[1:]):
                logger.info("Stopping: quality stopped improving")
                return False

        return True

class IterativeTrainer:
    """Main iterative training system"""

    def __init__(self, config: IterativeConfig):
        self.config = config
        self.processor = NovelProcessor(config)
        self.validator = QualityValidator(config)

        # Setup directories
        self.novels_dir = Path(config.novels_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configure CPU optimization
        torch.set_num_threads(os.cpu_count())

        logger.info("Iterative Novel Trainer initialized")
        logger.info(f"CPU cores: {os.cpu_count()}")
        logger.info(f"Base model: {config.base_model}")

    def train_novel(self, novel_name: str) -> Dict[str, Any]:
        """Train on a single novel with multiple iterations"""
        novel_path = self.novels_dir / novel_name
        if not novel_path.exists():
            raise FileNotFoundError(f"Novel not found: {novel_path}")

        logger.info(f"Starting iterative training on: {novel_name}")

        # Load novel
        novel_data = self.processor.load_novel(novel_path)
        logger.info(f"Loaded: {novel_data['title']} ({novel_data['word_count']:,} words)")

        # Import training libraries
        try:
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TrainingArguments, Trainer,
                DataCollatorForLanguageModeling
            )
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return {"error": "Missing transformers library"}

        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float32
        )

        # Training results
        results = {
            "novel": novel_data['title'],
            "iterations": [],
            "final_quality": None,
            "training_time": 0
        }

        start_time = time.time()

        # Iterative training loop
        for iteration in range(self.config.iterations_per_novel):
            if not self.validator.should_continue_training(iteration):
                break

            logger.info(f"\n--- Iteration {iteration + 1}/{self.config.iterations_per_novel} ---")

            # Create progressive chunks
            chunks = self.processor.create_progressive_chunks(
                novel_data['content'], iteration
            )

            # Split for training and validation
            train_chunks, val_chunks = self.processor.create_train_val_split(chunks)

            if not train_chunks:
                logger.warning("No training data available")
                break

            # Prepare dataset
            train_dataset = self._prepare_dataset(train_chunks, tokenizer)

            # Calculate learning rate for this iteration
            lr_progress = iteration / (self.config.iterations_per_novel - 1)
            current_lr = self.config.learning_rate_start * (1 - lr_progress) + \
                        self.config.learning_rate_end * lr_progress

            # Training arguments
            iteration_output_dir = self.output_dir / novel_name / f"iteration_{iteration}"
            iteration_output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(iteration_output_dir),
                overwrite_output_dir=True,
                max_steps=self.config.max_steps_per_iteration,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=current_lr,
                warmup_ratio=self.config.warmup_ratio,
                logging_steps=5,
                save_steps=self.config.max_steps_per_iteration,
                save_total_limit=1,
                report_to="none",
                use_cpu=True,
                dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
                prediction_loss_only=True
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
                train_dataset=train_dataset,
                data_collator=data_collator
            )

            # Train this iteration
            logger.info(f"Training iteration {iteration + 1} (LR: {current_lr:.2e})...")
            iter_start = time.time()

            trainer.train()

            iter_time = time.time() - iter_start

            # Validate quality
            perplexity = self.validator.calculate_perplexity(model, tokenizer, val_chunks)
            quality_result = self.validator.test_generation_quality(
                model, tokenizer, f"In the style of {novel_data['title']}"
            )

            iteration_result = {
                "iteration": iteration + 1,
                "learning_rate": current_lr,
                "training_time": iter_time,
                "perplexity": perplexity,
                "quality_score": quality_result["average_quality"],
                "sample_generation": quality_result["sample_generation"],
                "improving": quality_result["improving"]
            }

            results["iterations"].append(iteration_result)

            logger.info(f"Iteration {iteration + 1} completed:")
            logger.info(f"  Time: {iter_time:.1f}s")
            logger.info(f"  Perplexity: {perplexity:.2f}")
            logger.info(f"  Quality: {quality_result['average_quality']:.3f}")
            logger.info(f"  Sample: {quality_result['sample_generation'][:100]}...")

            # Save checkpoint if enabled
            if self.config.save_checkpoints:
                model.save_pretrained(str(iteration_output_dir))
                tokenizer.save_pretrained(str(iteration_output_dir))

        # Final results
        total_time = time.time() - start_time
        results["training_time"] = total_time
        results["final_quality"] = self.validator.generation_quality_history[-1] if self.validator.generation_quality_history else 0

        # Save final model
        final_model_dir = self.output_dir / novel_name / "final"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        # Save training results
        with open(self.output_dir / novel_name / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nTraining completed for {novel_data['title']}:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Iterations: {len(results['iterations'])}")
        logger.info(f"  Final quality: {results['final_quality']:.3f}")

        return results

    def _prepare_dataset(self, chunks: List[str], tokenizer) -> Dataset:
        """Prepare dataset for training"""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length
            )

        dataset = Dataset.from_dict({"text": chunks})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        return tokenized_dataset

    def list_available_novels(self) -> List[str]:
        """List available novels for training"""
        novels = []
        for novel_dir in self.novels_dir.iterdir():
            if novel_dir.is_dir() and list(novel_dir.glob("*.txt")):
                novels.append(novel_dir.name)
        return sorted(novels)

    def generate_sample(self, novel_name: str, prompt: str = None) -> str:
        """Generate a sample from a trained model"""
        model_dir = self.output_dir / novel_name / "final"

        if not model_dir.exists():
            return f"No trained model found for {novel_name}"

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))

            if prompt is None:
                prompt = f"In the style of {novel_name.replace('_', ' ')}: "

            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated[len(prompt):].strip()

        except Exception as e:
            return f"Error generating sample: {e}"

def main():
    """Main function"""
    print("Iterative Novel Training System")
    print("=" * 50)

    config = IterativeConfig()
    trainer = IterativeTrainer(config)

    # List available novels
    novels = trainer.list_available_novels()
    if not novels:
        print("No novels found in the novels directory!")
        return

    print(f"Found {len(novels)} novels available for training")
    print("\nFirst 10 novels:")
    for i, novel in enumerate(novels[:10], 1):
        print(f"  {i}. {novel.replace('_', ' ').title()}")

    # Train the first novel as demonstration
    if novels:
        selected_novel = novels[0]
        print(f"\nStarting iterative training on: {selected_novel}")
        print("This will train with progressive difficulty over multiple iterations")
        print("Each iteration will be evaluated for quality and fluency")

        try:
            results = trainer.train_novel(selected_novel)

            print(f"\n{'='*50}")
            print("TRAINING COMPLETE!")
            print(f"{'='*50}")
            print(f"Novel: {results['novel']}")
            print(f"Total time: {results['training_time']:.1f} seconds")
            print(f"Iterations completed: {len(results['iterations'])}")
            print(f"Final quality score: {results['final_quality']:.3f}")

            # Generate a sample
            print(f"\nGenerating sample from trained model...")
            sample = trainer.generate_sample(selected_novel)
            print(f"Sample output: {sample[:200]}...")

            print(f"\nModel saved to: {config.output_dir}/{selected_novel}/final/")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"Training failed: {e}")

if __name__ == "__main__":
    main()