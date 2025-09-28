#!/usr/bin/env python3
"""
Model Tea - Utilities Module
Copyright © ChaiQ LLC

Common utilities, helper functions, and shared components extracted from
the main training modules to reduce technical debt and improve maintainability.
"""

import os
import sys
import json
import time
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelTeaConfig:
    """Unified configuration for Model Tea system"""
    # Model settings
    base_model: str = "gpt2"
    max_seq_length: int = 1024

    # Training parameters
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 5e-6
    iterations_per_novel: int = 12
    max_steps_per_iteration: int = 20
    warmup_steps: int = 5

    # Data processing
    chunk_size: int = 200
    validation_split: float = 0.2

    # Quality control
    perplexity_threshold: float = 50.0
    quality_threshold: float = 0.8

    # Memory system
    enable_memory_system: bool = True
    max_memories_per_novel: int = 250
    memory_chunk_size: int = 35
    memory_retrieval_limit: int = 5
    memory_randomness: float = 0.15

    # Directories
    novels_dir: str = "novels"
    output_dir: str = "iterative_models"
    memory_dir: str = "episodic_memories"
    results_dir: str = "memory_analysis_results"

class FileSystemUtils:
    """File system utilities for Model Tea"""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists and return Path object"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_json_save(data: Any, file_path: Path) -> bool:
        """Safely save JSON data with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False

    @staticmethod
    def safe_json_load(file_path: Path) -> Optional[Dict]:
        """Safely load JSON data with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    @staticmethod
    def find_novel_files(novels_dir: Path) -> List[Path]:
        """Find all available novel directories"""
        novels = []
        for novel_dir in novels_dir.iterdir():
            if novel_dir.is_dir() and list(novel_dir.glob("*.txt")):
                novels.append(novel_dir)
        return sorted(novels)

class TextProcessingUtils:
    """Text processing utilities"""

    @staticmethod
    def create_progressive_chunks(text: str, iteration: int, base_size: int = 200) -> List[str]:
        """Create chunks with progressive sizing"""
        # Calculate chunk size for this iteration
        progression_factor = 1 + (iteration * 0.2)
        current_size = int(base_size * progression_factor)

        # Split into sentences for better chunk boundaries
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > current_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_words = sentence_words
            else:
                current_chunk += " " + sentence
                current_words += sentence_words

        if current_chunk and current_words > 10:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def calculate_text_stats(text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "character_count": len(text),
            "estimated_reading_time": len(words) / 250  # ~250 words per minute
        }

class TrainingUtils:
    """Training-related utilities"""

    @staticmethod
    def calculate_learning_rate(iteration: int, total_iterations: int,
                              start_lr: float = 5e-5, end_lr: float = 1e-5) -> float:
        """Calculate learning rate for given iteration"""
        if total_iterations <= 1:
            return start_lr

        progress = iteration / (total_iterations - 1)
        return start_lr * (1 - progress) + end_lr * progress

    @staticmethod
    def calculate_chunk_size(iteration: int, base_size: int = 200) -> int:
        """Calculate chunk size for given iteration"""
        return int(base_size + (iteration * base_size * 0.2))

    @staticmethod
    def format_training_time(seconds: float) -> str:
        """Format training time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

class QualityMetrics:
    """Quality assessment utilities"""

    @staticmethod
    def assess_text_quality(text: str) -> Dict[str, float]:
        """Assess text quality using multiple metrics"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Basic metrics
        unique_words = set(word.lower() for word in words)
        repetition_ratio = len(unique_words) / len(words) if words else 0

        # Sentence length variance (good indicator of natural writing)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0

        # Quality score (weighted combination)
        quality_score = (
            repetition_ratio * 0.4 +  # Vocabulary diversity
            min(1.0, avg_sentence_length / 15) * 0.3 +  # Reasonable sentence length
            min(1.0, sentence_variance / 25) * 0.3  # Natural sentence variation
        )

        return {
            "quality_score": quality_score,
            "repetition_ratio": repetition_ratio,
            "avg_sentence_length": avg_sentence_length,
            "sentence_variance": sentence_variance,
            "vocabulary_diversity": len(unique_words),
            "total_words": len(words)
        }

    @staticmethod
    def calculate_perplexity_from_loss(loss: float) -> float:
        """Calculate perplexity from loss value"""
        return np.exp(loss)

    @staticmethod
    def assess_generation_improvement(baseline: str, enhanced: str) -> Dict[str, float]:
        """Compare baseline vs enhanced generation"""
        baseline_metrics = QualityMetrics.assess_text_quality(baseline)
        enhanced_metrics = QualityMetrics.assess_text_quality(enhanced)

        return {
            "quality_improvement": enhanced_metrics["quality_score"] - baseline_metrics["quality_score"],
            "vocabulary_improvement": enhanced_metrics["vocabulary_diversity"] - baseline_metrics["vocabulary_diversity"],
            "length_change": enhanced_metrics["total_words"] - baseline_metrics["total_words"],
            "baseline_quality": baseline_metrics["quality_score"],
            "enhanced_quality": enhanced_metrics["quality_score"]
        }

class MemorySystemUtils:
    """Memory system utilities"""

    @staticmethod
    def check_memory_system_available() -> bool:
        """Check if memory system is available"""
        try:
            from episodic_memory_system import EpisodicMemorySystem
            return True
        except ImportError:
            return False

    @staticmethod
    def safe_memory_operation(operation_func, *args, **kwargs) -> Tuple[bool, Any]:
        """Safely execute memory operation with error handling"""
        try:
            result = operation_func(*args, **kwargs)
            return True, result
        except Exception as e:
            logger.warning(f"Memory operation failed: {e}")
            return False, None

    @staticmethod
    def create_memory_fallback() -> Dict[str, Any]:
        """Create fallback response when memory system fails"""
        return {
            "system_status": "unavailable",
            "total_memories": 0,
            "error": "Memory system not available"
        }

class ErrorHandling:
    """Centralized error handling"""

    @staticmethod
    def handle_training_error(error: Exception, context: str) -> Dict[str, Any]:
        """Handle training errors gracefully"""
        logger.error(f"Training error in {context}: {error}")
        return {
            "error": str(error),
            "context": context,
            "status": "failed",
            "recovery_suggestion": "Check dependencies and try again"
        }

    @staticmethod
    def handle_memory_error(error: Exception, context: str) -> Dict[str, Any]:
        """Handle memory system errors gracefully"""
        logger.warning(f"Memory system error in {context}: {error}")
        return {
            "error": str(error),
            "context": context,
            "status": "memory_disabled",
            "recovery_suggestion": "Training will continue without memory enhancement"
        }

def validate_system_setup() -> Dict[str, bool]:
    """Validate Model Tea system setup"""
    validation = {
        "directories": True,
        "dependencies": True,
        "memory_system": True,
        "models": True
    }

    try:
        # Check required directories
        required_dirs = ["novels", "iterative_models", "episodic_memories"]
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)

        # Check dependencies
        import torch
        import transformers
        import datasets

        # Check memory system
        validation["memory_system"] = MemorySystemUtils.check_memory_system_available()

        logger.info("Model Tea system validation passed")

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        validation["dependencies"] = False

    return validation

if __name__ == "__main__":
    print("Model Tea - Utilities Module")
    print("Copyright © ChaiQ LLC")
    print("=" * 30)

    # Run system validation
    validation = validate_system_setup()
    print("System validation:")
    for component, status in validation.items():
        status_str = "OK" if status else "FAILED"
        print(f"  {component}: {status_str}")