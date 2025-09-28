#!/usr/bin/env python3

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..config.settings import PreprocessorConfig, ModelConfig


class TextPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

        if config.use_huggingface_models and HF_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Loaded tokenizer: {self.config.model_name}")
        except Exception as e:
            print(f"Could not load model {self.config.model_name}: {e}")
            self.tokenizer = None

    def preprocess_novel(self, novel_path: Path) -> Tuple[str, Dict]:
        with open(novel_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        stats = {
            'original_length': len(original_text),
            'original_tokens': len(original_text.split())
        }

        text = self.remove_metadata_and_headers(original_text)
        text = self.standardize_chapter_breaks(text)
        text = self.convert_scripts_to_prose(text)
        text = self.normalize_dialogue(text)
        text = self.standardize_punctuation(text)
        text = self.clean_whitespace(text)

        stats.update({
            'processed_length': len(text),
            'processed_tokens': len(text.split()),
            'tokens_removed': stats['original_tokens'] - len(text.split()),
            'compression_ratio': len(text) / len(original_text)
        })

        return text, stats

    def remove_metadata_and_headers(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        skip_mode = False

        for line in lines:
            line_stripped = line.strip()

            if any(marker in line for marker in self.config.content_markers_to_remove):
                skip_mode = True
                continue

            if skip_mode and line_stripped == "":
                continue
            elif skip_mode and len(line_stripped) > 20:
                skip_mode = False

            if not skip_mode:
                if not any(marker in line for marker in self.config.content_markers_to_remove):
                    if not re.match(r'^\s*[IVXLCDM]+\.?\s*$', line_stripped):
                        if not re.match(r'^\s*\d+\s*$', line_stripped):
                            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def standardize_chapter_breaks(self, text: str) -> str:
        for pattern in self.config.chapter_markers:
            try:
                text = re.sub(pattern, '\n\n', text, flags=re.MULTILINE | re.IGNORECASE)
            except re.error as e:
                print(f"Regex error in pattern '{pattern}': {e}")
                continue

        return text

    def convert_scripts_to_prose(self, text: str) -> str:
        if not self.config.convert_scripts_to_prose:
            return text

        try:
            character_dialogue_pattern = self.config.dialogue_patterns["character_name_dialogue"]

            def replace_character_dialogue(match):
                character_name = match.group(1).title()
                dialogue = match.group(2).strip()

                if dialogue.startswith('"') and dialogue.endswith('"'):
                    return f'"{dialogue[1:-1]}," {character_name} said.'
                else:
                    return f'"{dialogue}," {character_name} said.'

            text = re.sub(character_dialogue_pattern, replace_character_dialogue, text, flags=re.MULTILINE)

            stage_direction_pattern = self.config.dialogue_patterns["stage_direction"]
            text = re.sub(stage_direction_pattern, r'\1', text)
        except re.error as e:
            print(f"Regex error in convert_scripts_to_prose: {e}")

        return text

    def normalize_dialogue(self, text: str) -> str:
        if not self.config.standardize_dialogue:
            return text

        try:
            text = re.sub(r'[""]', '"', text)
            text = re.sub(r'['']', "'", text)

            broken_quotes_pattern = self.config.dialogue_patterns["broken_quotes"]
            text = re.sub(broken_quotes_pattern, r'\1" \2', text)

            text = re.sub(r'(\w)"(\w)', r'\1" \2', text)
            text = re.sub(r'(\w)"([.!?])', r'\1"\2', text)
        except re.error as e:
            print(f"Regex error in normalize_dialogue: {e}")

        return text

    def standardize_punctuation(self, text: str) -> str:
        if not self.config.normalize_punctuation:
            return text

        text = re.sub(r'--+', '—', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        return text

    def clean_whitespace(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        lines = []
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line or (lines and lines[-1] != ''):
                lines.append(cleaned_line)

        return '\n'.join(lines).strip()

    def backup_and_replace_file(self, novel_path: Path, processed_text: str):
        novel_dir = novel_path.parent
        backup_dir = novel_dir / self.config.backup_folder
        backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / novel_path.name
        shutil.copy2(novel_path, backup_path)

        with open(novel_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)

        return backup_path