from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelTeaConfig:
    base_model: str = "gpt2"
    max_seq_length: int = 1024

    learning_rate_start: float = 5e-5
    learning_rate_end: float = 5e-6
    iterations_per_novel: int = 12
    max_steps_per_iteration: int = 20
    warmup_steps: int = 5

    chunk_size: int = 200
    validation_split: float = 0.2

    perplexity_threshold: float = 50.0
    quality_threshold: float = 0.8

    enable_memory_system: bool = True
    max_memories_per_novel: int = 250
    memory_chunk_size: int = 35
    memory_retrieval_limit: int = 5
    memory_randomness: float = 0.15

    novels_dir: str = "novels"
    output_dir: str = "iterative_models"
    memory_dir: str = "episodic_memories"
    results_dir: str = "memory_analysis_results"

    @property
    def novels_path(self) -> Path:
        return Path(self.novels_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def memory_path(self) -> Path:
        return Path(self.memory_dir)

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir)
