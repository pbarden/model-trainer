#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class PreprocessorConfig:
    novels_dir: str = "novels"
    backup_folder: str = "__old"
    use_huggingface_models: bool = True
    model_name: str = "microsoft/DialoGPT-medium"
    max_chunk_size: int = 1000
    preserve_chapters: bool = True
    convert_scripts_to_prose: bool = True
    standardize_dialogue: bool = True
    remove_metadata: bool = True
    normalize_punctuation: bool = True
    batch_size: int = 10

    content_markers_to_remove: List[str] = None
    chapter_markers: List[str] = None
    dialogue_patterns: Dict[str, str] = None

    def __post_init__(self):
        if self.content_markers_to_remove is None:
            self.content_markers_to_remove = [
                "Title:",
                "Author:",
                "CONTENTS",
                "CHAPTER",
                "[Illustration:",
                "[Footnote",
                "Illustrat",
                "*       *       *",
                "THE SCENES",
                "ACT I",
                "ACT II",
                "ACT III"
            ]

        if self.chapter_markers is None:
            self.chapter_markers = [
                r"CHAPTER\s+[IVXLCDM]+\.?",
                r"Chapter\s+\d+",
                r"^[IVXLCDM]+\.\s*$",
                r"Scene\s+[IVXLCDM]+",
                r"ACT\s+[IVXLCDM]+"
            ]

        if self.dialogue_patterns is None:
            self.dialogue_patterns = {
                "character_name_dialogue": r"^([A-Z][A-Z\s]+)\n\n(.+?)$",
                "stage_direction": r"_(.+?)_",
                "broken_quotes": r'([.!?])\"(\s+[a-z])',
                "missing_quotes": r'([A-Z].*[.!?])\s*$'
            }


@dataclass
class ModelConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    use_gpu: bool = True
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class ProcessingStats:
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_tokens_removed: int = 0
    total_tokens_preserved: int = 0
    processing_time: float = 0.0