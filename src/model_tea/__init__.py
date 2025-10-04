__version__ = "2.0.0"

__all__ = [
    "IterativeTrainer",
    "IterativeConfig",
    "ModelTrainer",
    "ModelConfig",
    "TrainingService",
    "ChatService",
    "MetadataService",
]


def __getattr__(name):
    if name == "IterativeTrainer":
        from model_tea.trainers.iterative import IterativeTrainer
        return IterativeTrainer
    elif name == "IterativeConfig":
        from model_tea.trainers.iterative import IterativeConfig
        return IterativeConfig
    elif name == "ModelTrainer":
        from model_tea.trainers.model import ModelTrainer
        return ModelTrainer
    elif name == "ModelConfig":
        from model_tea.trainers.model import ModelConfig
        return ModelConfig
    elif name == "TrainingService":
        from model_tea.services.training import TrainingService
        return TrainingService
    elif name == "ChatService":
        from model_tea.services.chat import ChatService
        return ChatService
    elif name == "MetadataService":
        from model_tea.services.metadata import MetadataService
        return MetadataService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
