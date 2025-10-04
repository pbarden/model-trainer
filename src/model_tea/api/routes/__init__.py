from .training import router as training_router
from .chat import router as chat_router
from .models import router as models_router

__all__ = [
    "training_router",
    "chat_router",
    "models_router",
]
