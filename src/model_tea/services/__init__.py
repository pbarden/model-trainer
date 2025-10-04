from .metadata import MetadataService

__all__ = [
    "MetadataService",
    "TrainingService",
    "ChatService",
]


def get_training_service():
    """Lazy load TrainingService to avoid importing heavy ML libraries"""
    from .training import TrainingService
    return TrainingService


def get_chat_service():
    """Lazy load ChatService to avoid importing heavy ML libraries"""
    from .chat import ChatService
    return ChatService
