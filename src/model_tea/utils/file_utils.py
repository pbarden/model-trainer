import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FileSystemUtils:
    @staticmethod
    def ensure_directory(path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_json_save(data: Any, file_path: Path) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False

    @staticmethod
    def safe_json_load(file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    @staticmethod
    def find_novel_files(novels_dir: Path) -> List[Path]:
        novels = []
        for novel_dir in novels_dir.iterdir():
            if novel_dir.is_dir() and list(novel_dir.glob("*.txt")):
                novels.append(novel_dir)
        return sorted(novels)
