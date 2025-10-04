from dataclasses import dataclass
from model_tea.config.settings import ModelTeaConfig


@dataclass
class CombinedModelConfig:
    base_config: ModelTeaConfig = None

    combine_novels_method: str = "concatenate"
    novel_separator: str = "\n\n=== NEW NOVEL ===\n\n"
    min_novels_required: int = 1
    max_combined_size: int = 2000000

    max_iterations: int = 14
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 5e-6

    combined_memories_count: int = 350
    cross_novel_memories: bool = True

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = ModelTeaConfig()
