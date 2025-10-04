import logging
from typing import Tuple, Any, Dict

logger = logging.getLogger(__name__)


class MemorySystemUtils:
    @staticmethod
    def check_memory_system_available() -> bool:
        try:
            from episodic_memory_system import EpisodicMemorySystem
            return True
        except ImportError:
            return False

    @staticmethod
    def safe_memory_operation(operation_func, *args, **kwargs) -> Tuple[bool, Any]:
        try:
            result = operation_func(*args, **kwargs)
            return True, result
        except Exception as e:
            logger.warning(f"Memory operation failed: {e}")
            return False, None

    @staticmethod
    def create_memory_fallback() -> Dict[str, Any]:
        return {
            "system_status": "unavailable",
            "total_memories": 0,
            "error": "Memory system not available"
        }
