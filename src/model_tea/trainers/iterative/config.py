"""
Iterative training configuration.
Now uses centralized settings from config.settings.
"""

from dataclasses import dataclass
from model_tea.config.settings import settings


@dataclass
class IterativeConfig:
    """
    Configuration for iterative training.
    Values loaded from centralized settings, with optional overrides.
    """

    def __init__(self, **kwargs):
        s = settings

        self.base_model = kwargs.get('base_model', s.iterative_base_model)
        self.max_seq_length = kwargs.get('max_seq_length', s.iterative_max_seq_length)

        self.use_lora = kwargs.get('use_lora', s.use_lora)
        self.lora_r = kwargs.get('lora_r', s.lora_r)
        self.lora_alpha = kwargs.get('lora_alpha', s.lora_alpha)
        self.lora_dropout = kwargs.get('lora_dropout', s.lora_dropout)
        self.lora_target_modules = kwargs.get('lora_target_modules', s.lora_target_modules)

        self.iterations_per_novel = kwargs.get('iterations_per_novel', s.iterations_per_novel)
        self.max_steps_per_iteration = kwargs.get('max_steps_per_iteration', s.iterative_max_steps)
        self.learning_rate_start = kwargs.get('learning_rate_start', s.iterative_learning_rate_start)
        self.learning_rate_end = kwargs.get('learning_rate_end', s.iterative_learning_rate_end)

        self.chunk_size = kwargs.get('chunk_size', s.iterative_chunk_size)
        self.chunk_overlap = kwargs.get('chunk_overlap', s.chunk_overlap)
        self.validation_split = kwargs.get('validation_split', s.iterative_validation_split)

        self.batch_size = kwargs.get('batch_size', s.batch_size)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', s.gradient_accumulation_steps)
        self.warmup_ratio = kwargs.get('warmup_ratio', s.warmup_ratio)

        self.max_repetition_penalty = kwargs.get('max_repetition_penalty', s.max_repetition_penalty)
        self.temperature_range = kwargs.get('temperature_range', (s.temperature_range_min, s.temperature_range_max))
        self.perplexity_threshold = kwargs.get('perplexity_threshold', s.iterative_perplexity_threshold)

        self.novels_dir = kwargs.get('novels_dir', s.novels_dir)
        self.output_dir = kwargs.get('output_dir', s.output_dir)
        self.save_checkpoints = kwargs.get('save_checkpoints', s.save_checkpoints)

        self.adaptive_training = kwargs.get('adaptive_training', s.adaptive_training)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', s.early_stopping_patience)
        self.overfitting_detection_window = kwargs.get('overfitting_detection_window', s.overfitting_detection_window)
        self.min_iterations = kwargs.get('min_iterations', s.min_iterations)
        self.max_iterations = kwargs.get('max_iterations', s.max_iterations)
        self.perplexity_improvement_threshold = kwargs.get('perplexity_improvement_threshold', s.perplexity_improvement_threshold)
        self.quality_degradation_threshold = kwargs.get('quality_degradation_threshold', s.quality_degradation_threshold)
        self.validation_loss_patience = kwargs.get('validation_loss_patience', s.validation_loss_patience)
        self.target_perplexity = kwargs.get('target_perplexity', s.target_perplexity)
