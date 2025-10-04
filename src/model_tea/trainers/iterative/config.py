from dataclasses import dataclass


@dataclass
class IterativeConfig:
    base_model: str = "distilgpt2"
    max_seq_length: int = 512

    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["c_attn", "c_proj"]

    iterations_per_novel: int = 12
    max_steps_per_iteration: int = 125
    learning_rate_start: float = 2e-5
    learning_rate_end: float = 5e-6

    chunk_size: int = 150
    chunk_overlap: int = 30
    validation_split: float = 0.15

    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.15

    max_repetition_penalty: float = 1.1
    temperature_range: tuple = (0.7, 0.9)
    perplexity_threshold: float = 20.0

    novels_dir: str = "novels"
    output_dir: str = "iterative_models"
    save_checkpoints: bool = True

    adaptive_training: bool = True
    early_stopping_patience: int = 3
    overfitting_detection_window: int = 3
    min_iterations: int = 5
    max_iterations: int = 50
    perplexity_improvement_threshold: float = 2.0
    quality_degradation_threshold: float = 0.01
    validation_loss_patience: int = 3
    target_perplexity: float = 15.0
