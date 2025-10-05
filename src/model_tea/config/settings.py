"""
Centralized configuration for Model Tea.
All configurable values should be defined here and support environment variables.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTeaSettings(BaseSettings):
    """
    Main configuration class using pydantic-settings.
    Values can be set via environment variables with MODEL_TEA_ prefix.

    Example: MODEL_TEA_BASE_MODEL=gpt2 python -m model_tea.cli.main
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_TEA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    base_model: str = Field(
        default="distilgpt2",
        description="Base model to use for training"
    )

    iterative_base_model: str = Field(
        default="distilgpt2",
        description="Base model for iterative training (smaller, faster)"
    )

    max_seq_length: int = Field(
        default=1024,
        description="Maximum sequence length for models"
    )

    iterative_max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length for iterative training"
    )


    learning_rate_start: float = Field(
        default=5e-5,
        description="Starting learning rate"
    )

    learning_rate_end: float = Field(
        default=5e-6,
        description="Ending learning rate"
    )

    iterative_learning_rate_start: float = Field(
        default=2e-5,
        description="Starting learning rate for iterative training"
    )

    iterative_learning_rate_end: float = Field(
        default=5e-6,
        description="Ending learning rate for iterative training"
    )

    iterations_per_novel: int = Field(
        default=12,
        description="Number of training iterations per novel"
    )

    max_steps_per_iteration: int = Field(
        default=20,
        description="Maximum steps per iteration (basic training)"
    )

    iterative_max_steps: int = Field(
        default=125,
        description="Maximum steps per iteration (iterative training)"
    )

    warmup_steps: int = Field(
        default=5,
        description="Number of warmup steps"
    )

    warmup_ratio: float = Field(
        default=0.15,
        description="Warmup ratio for iterative training"
    )

    batch_size: int = Field(
        default=8,
        description="Training batch size"
    )

    gradient_accumulation_steps: int = Field(
        default=2,
        description="Gradient accumulation steps"
    )

    chunk_size: int = Field(
        default=200,
        description="Text chunk size for basic training"
    )

    iterative_chunk_size: int = Field(
        default=150,
        description="Text chunk size for iterative training"
    )

    chunk_overlap: int = Field(
        default=30,
        description="Overlap between text chunks"
    )

    validation_split: float = Field(
        default=0.2,
        description="Validation split ratio (basic training)"
    )

    iterative_validation_split: float = Field(
        default=0.15,
        description="Validation split ratio (iterative training)"
    )


    use_lora: bool = Field(
        default=True,
        description="Enable LoRA (Low-Rank Adaptation)"
    )

    lora_r: int = Field(
        default=8,
        description="LoRA rank"
    )

    lora_alpha: int = Field(
        default=16,
        description="LoRA alpha parameter"
    )

    lora_dropout: float = Field(
        default=0.05,
        description="LoRA dropout rate"
    )

    lora_target_modules: List[str] = Field(
        default=["c_attn", "c_proj"],
        description="LoRA target modules"
    )


    perplexity_threshold: float = Field(
        default=50.0,
        description="Perplexity threshold (basic training)"
    )

    iterative_perplexity_threshold: float = Field(
        default=20.0,
        description="Perplexity threshold (iterative training)"
    )

    target_perplexity: float = Field(
        default=15.0,
        description="Target perplexity to achieve"
    )

    quality_threshold: float = Field(
        default=0.8,
        description="Quality threshold for validation"
    )

    perplexity_improvement_threshold: float = Field(
        default=2.0,
        description="Minimum perplexity improvement to continue"
    )

    quality_degradation_threshold: float = Field(
        default=0.01,
        description="Maximum quality degradation allowed"
    )


    adaptive_training: bool = Field(
        default=True,
        description="Enable adaptive training"
    )

    early_stopping_patience: int = Field(
        default=3,
        description="Early stopping patience"
    )

    overfitting_detection_window: int = Field(
        default=3,
        description="Window for overfitting detection"
    )

    validation_loss_patience: int = Field(
        default=3,
        description="Validation loss patience"
    )

    min_iterations: int = Field(
        default=5,
        description="Minimum training iterations"
    )

    max_iterations: int = Field(
        default=50,
        description="Maximum training iterations"
    )


    default_max_length: int = Field(
        default=250,
        description="Default maximum generation length"
    )

    default_temperature: float = Field(
        default=0.5,
        description="Default generation temperature"
    )

    validation_temperature: float = Field(
        default=0.8,
        description="Temperature for validation generation"
    )

    max_repetition_penalty: float = Field(
        default=1.1,
        description="Maximum repetition penalty"
    )

    temperature_range_min: float = Field(
        default=0.7,
        description="Minimum temperature range"
    )

    temperature_range_max: float = Field(
        default=0.9,
        description="Maximum temperature range"
    )


    novels_dir: str = Field(
        default="novels",
        description="Directory containing novels"
    )

    models_file: str = Field(
        default="models.json",
        description="Models configuration file"
    )

    novels_file: str = Field(
        default="novels.json",
        description="Novels metadata file"
    )

    output_dir: str = Field(
        default="iterative_models",
        description="Output directory for trained models"
    )

    save_checkpoints: bool = Field(
        default=True,
        description="Save training checkpoints"
    )


    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )

    api_port: int = Field(
        default=8000,
        description="API server port"
    )

    api_cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins (use specific origins in production)"
    )


    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


    @property
    def novels_path(self) -> Path:
        """Get novels directory as Path object"""
        return Path(self.novels_dir)

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object"""
        return Path(self.output_dir)

    @property
    def models_path(self) -> Path:
        """Get models file as Path object"""
        return Path(self.models_file)

    @property
    def novels_metadata_path(self) -> Path:
        """Get novels metadata file as Path object"""
        return Path(self.novels_file)


# Global settings instance
settings = ModelTeaSettings()


def get_settings() -> ModelTeaSettings:
    """Get the global settings instance"""
    return settings


# Legacy compatibility - create old config objects from new settings
from dataclasses import dataclass


@dataclass
class ModelTeaConfig:
    """Legacy config class for backward compatibility"""

    def __init__(self):
        s = settings
        self.base_model = s.base_model
        self.max_seq_length = s.max_seq_length
        self.learning_rate_start = s.learning_rate_start
        self.learning_rate_end = s.learning_rate_end
        self.iterations_per_novel = s.iterations_per_novel
        self.max_steps_per_iteration = s.max_steps_per_iteration
        self.warmup_steps = s.warmup_steps
        self.chunk_size = s.chunk_size
        self.validation_split = s.validation_split
        self.perplexity_threshold = s.perplexity_threshold
        self.quality_threshold = s.quality_threshold
        self.novels_dir = s.novels_dir
        self.output_dir = s.output_dir

    @property
    def novels_path(self) -> Path:
        return settings.novels_path

    @property
    def output_path(self) -> Path:
        return settings.output_path
