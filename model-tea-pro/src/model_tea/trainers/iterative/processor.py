import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

from .config import IterativeConfig

logger = logging.getLogger(__name__)


class NovelProcessor:
    """Processes novels for iterative training"""

    def __init__(self, config: IterativeConfig):
        self.config = config

    def load_novel(self, novel_path: Path) -> Dict[str, Any]:
        """Load and analyze a single novel"""
        text_files = list(novel_path.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(f"No text file in {novel_path}")

        # Load novel content
        with open(text_files[0], 'r', encoding='utf-8') as f:
            content = f.read()

        # Load analysis if available
        analysis_file = novel_path / "analysis.json"
        analysis = {}
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

        return {
            "title": novel_path.name.replace('_', ' ').title(),
            "content": content,
            "word_count": analysis.get("word_count", len(content.split())),
            "path": novel_path
        }

    def create_progressive_chunks(self, content: str, iteration: int) -> List[str]:
        """Create chunks with progressive difficulty - sentence-based"""
        # Start with easier (shorter) chunks, progress to longer ones
        base_size = self.config.chunk_size
        progression_factor = 1 + (iteration * 0.2)
        current_chunk_size = int(base_size * progression_factor)

        # Split into sentences (ignoring newlines as they don't indicate paragraphs)
        sentences = self._split_sentences(content)

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > current_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Add overlap for context continuity
                overlap_words = self.config.chunk_overlap
                if overlap_words > 0:
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-overlap_words:])
                    current_chunk = overlap_text + " " + sentence
                    current_words = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_words = sentence_words
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_words += sentence_words

        if current_chunk and current_words > 20:
            chunks.append(current_chunk.strip())

        logger.info(f"Created {len(chunks)} sentence-based chunks (target {current_chunk_size} words)")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]

    def create_train_val_split(self, chunks: List[str]) -> tuple:
        """Split chunks into training and validation sets"""
        random.shuffle(chunks)
        split_idx = int(len(chunks) * (1 - self.config.validation_split))

        train_chunks = chunks[:split_idx]
        val_chunks = chunks[split_idx:]

        logger.info(f"Split: {len(train_chunks)} train, {len(val_chunks)} validation")
        return train_chunks, val_chunks

