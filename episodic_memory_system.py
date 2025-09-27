#!/usr/bin/env python3
"""
Model Tea - Episodic Memory System
Copyright © ChaiQ LLC

Creates a human-like memory system that works alongside trained models.
Implements: Prompt > Model > Memory(ies) > Model > Output flow

Key Features:
- CPU-friendly memory storage and retrieval
- Contextual memory activation (like human memory)
- Independent memory networks per model
- Behavioral impact measurement
"""

import os
import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
import time
from collections import defaultdict
import pickle

@dataclass
class MemoryConfig:
    """Configuration for episodic memory system"""
    # Memory extraction settings
    memory_chunk_size: int = 35  # Words per memory chunk (smaller for granularity)
    overlap_ratio: float = 0.3   # Overlap between chunks
    max_memories_per_novel: int = 250  # Increased for richer context

    # Memory types to extract
    extract_characters: bool = True
    extract_locations: bool = True
    extract_emotions: bool = True
    extract_themes: bool = True
    extract_dialogue: bool = True
    extract_descriptions: bool = True

    # Retrieval settings
    max_retrieved_memories: int = 5  # Number of memories to activate (increased)
    relevance_threshold: float = 0.1  # Minimum similarity to activate
    randomness_factor: float = 0.15  # Reduced randomness for consistency

    # CPU optimization
    use_simple_similarity: bool = True  # Use basic keyword matching vs embeddings
    memory_cache_size: int = 100       # Cache frequently accessed memories

class MemoryType:
    """Types of memories that can be extracted"""
    CHARACTER = "character"
    LOCATION = "location"
    EMOTION = "emotion"
    THEME = "theme"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"

@dataclass
class Memory:
    """A single memory entry"""
    content: str
    memory_type: str
    keywords: List[str]
    context: str  # Surrounding context
    emotional_tone: str
    importance_score: float
    chapter_position: float  # Where in novel (0.0 to 1.0)

class MemoryExtractor:
    """Extracts different types of memories from novel text"""

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Simple patterns for memory extraction
        self.character_patterns = [
            r'\b[A-Z][a-z]+\s(?:said|replied|whispered|shouted|asked|thought)',
            r'"[^"]+",?\s+(?:said|replied)\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+(?:walked|ran|sat|stood|looked|felt|seemed)'
        ]

        self.location_patterns = [
            r'\b(?:in|at|near|beside|within)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:castle|house|room|street|city|town|forest|mountain)'
        ]

        self.emotion_patterns = [
            r'\b(happy|sad|angry|afraid|excited|worried|anxious|joyful|terrified|confused|surprised|amazed)\b',
            r'\b(?:felt|was|seemed|appeared)\s+(happy|sad|angry|afraid|excited|worried|anxious|joyful|terrified|confused|surprised|amazed)'
        ]

        self.theme_patterns = [
            r'\b(love|death|friendship|betrayal|loyalty|courage|fear|hope|despair|justice|revenge|sacrifice)\b',
            r'\b(?:about|concerning|regarding)\s+(love|death|friendship|betrayal|loyalty|courage|fear|hope|despair|justice|revenge|sacrifice)'
        ]

    def extract_memories(self, novel_text: str, novel_title: str) -> List[Memory]:
        """Extract all types of memories from novel text"""
        memories = []

        # Split into chunks for processing
        chunks = self._create_chunks(novel_text)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_position = i / total_chunks if total_chunks > 1 else 0.0

            # Extract different memory types
            if self.config.extract_characters:
                memories.extend(self._extract_character_memories(chunk, chunk_position))

            if self.config.extract_locations:
                memories.extend(self._extract_location_memories(chunk, chunk_position))

            if self.config.extract_emotions:
                memories.extend(self._extract_emotion_memories(chunk, chunk_position))

            if self.config.extract_themes:
                memories.extend(self._extract_theme_memories(chunk, chunk_position))

            if self.config.extract_dialogue:
                memories.extend(self._extract_dialogue_memories(chunk, chunk_position))

            if self.config.extract_descriptions:
                memories.extend(self._extract_description_memories(chunk, chunk_position))

        # Limit total memories for CPU efficiency
        memories = self._rank_and_limit_memories(memories)

        return memories

    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks for memory extraction"""
        words = text.split()
        chunks = []
        step_size = int(self.config.memory_chunk_size * (1 - self.config.overlap_ratio))

        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + self.config.memory_chunk_size]
            if len(chunk_words) >= 20:  # Minimum chunk size
                chunks.append(' '.join(chunk_words))

        return chunks

    def _extract_character_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract character-related memories"""
        memories = []

        for pattern in self.character_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 50)
                end = min(len(chunk), match.end() + 50)
                context = chunk[start:end]

                memory = Memory(
                    content=context,
                    memory_type=MemoryType.CHARACTER,
                    keywords=[match.group(1) if match.groups() else match.group(0)],
                    context=chunk,
                    emotional_tone=self._detect_emotional_tone(context),
                    importance_score=self._calculate_importance(context),
                    chapter_position=position
                )
                memories.append(memory)

        return memories

    def _extract_location_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract location-related memories"""
        memories = []

        for pattern in self.location_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(chunk), match.end() + 50)
                context = chunk[start:end]

                memory = Memory(
                    content=context,
                    memory_type=MemoryType.LOCATION,
                    keywords=[match.group(1) if match.groups() else match.group(0)],
                    context=chunk,
                    emotional_tone=self._detect_emotional_tone(context),
                    importance_score=self._calculate_importance(context),
                    chapter_position=position
                )
                memories.append(memory)

        return memories

    def _extract_emotion_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract emotion-related memories"""
        memories = []

        for pattern in self.emotion_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(chunk), match.end() + 50)
                context = chunk[start:end]

                emotion = match.group(1) if match.groups() else match.group(0)

                memory = Memory(
                    content=context,
                    memory_type=MemoryType.EMOTION,
                    keywords=[emotion],
                    context=chunk,
                    emotional_tone=emotion,
                    importance_score=self._calculate_importance(context),
                    chapter_position=position
                )
                memories.append(memory)

        return memories

    def _extract_theme_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract theme-related memories"""
        memories = []

        for pattern in self.theme_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(chunk), match.end() + 50)
                context = chunk[start:end]

                theme = match.group(1) if match.groups() else match.group(0)

                memory = Memory(
                    content=context,
                    memory_type=MemoryType.THEME,
                    keywords=[theme],
                    context=chunk,
                    emotional_tone=self._detect_emotional_tone(context),
                    importance_score=self._calculate_importance(context),
                    chapter_position=position
                )
                memories.append(memory)

        return memories

    def _extract_dialogue_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract dialogue memories"""
        memories = []

        # Find quoted dialogue
        dialogue_pattern = r'"([^"]{20,100})"'
        matches = re.finditer(dialogue_pattern, chunk)

        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(chunk), match.end() + 50)
            context = chunk[start:end]

            dialogue = match.group(1)
            keywords = [word for word in dialogue.split() if len(word) > 3][:5]

            memory = Memory(
                content=context,
                memory_type=MemoryType.DIALOGUE,
                keywords=keywords,
                context=chunk,
                emotional_tone=self._detect_emotional_tone(context),
                importance_score=self._calculate_importance(context),
                chapter_position=position
            )
            memories.append(memory)

        return memories

    def _extract_description_memories(self, chunk: str, position: float) -> List[Memory]:
        """Extract descriptive memories"""
        memories = []

        # Find descriptive sentences (longer sentences without dialogue)
        sentences = re.split(r'[.!?]+', chunk)

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 50 and
                len(sentence) < 200 and
                '"' not in sentence and
                any(word in sentence.lower() for word in ['beautiful', 'dark', 'bright', 'tall', 'small', 'cold', 'warm', 'old', 'new'])):

                keywords = [word for word in sentence.split() if len(word) > 4][:5]

                memory = Memory(
                    content=sentence,
                    memory_type=MemoryType.DESCRIPTION,
                    keywords=keywords,
                    context=chunk,
                    emotional_tone=self._detect_emotional_tone(sentence),
                    importance_score=self._calculate_importance(sentence),
                    chapter_position=position
                )
                memories.append(memory)

        return memories

    def _detect_emotional_tone(self, text: str) -> str:
        """Simple emotional tone detection"""
        positive_words = ['happy', 'joy', 'love', 'beautiful', 'wonderful', 'amazing', 'bright', 'warm']
        negative_words = ['sad', 'angry', 'fear', 'dark', 'cold', 'terrible', 'horrible', 'death', 'pain']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for memory"""
        # Simple heuristics for importance
        score = 0.5  # Base score

        # Longer content tends to be more important
        score += min(0.3, len(text) / 1000)

        # Presence of key words increases importance
        important_words = ['suddenly', 'finally', 'never', 'always', 'forever', 'death', 'love', 'life']
        text_lower = text.lower()
        score += sum(0.1 for word in important_words if word in text_lower)

        return min(1.0, score)

    def _rank_and_limit_memories(self, memories: List[Memory]) -> List[Memory]:
        """Rank memories by importance and limit count with balanced distribution"""
        # Group memories by type
        from collections import defaultdict
        memories_by_type = defaultdict(list)
        for memory in memories:
            memories_by_type[memory.memory_type].append(memory)

        # Sort each type by importance
        for memory_type in memories_by_type:
            memories_by_type[memory_type].sort(key=lambda m: m.importance_score, reverse=True)

        # Define target distribution (more balanced across types)
        max_memories = self.config.max_memories_per_novel
        target_distribution = {
            MemoryType.DESCRIPTION: int(max_memories * 0.60),  # 60% descriptions
            MemoryType.LOCATION: int(max_memories * 0.15),     # 15% locations
            MemoryType.CHARACTER: int(max_memories * 0.10),    # 10% characters
            MemoryType.DIALOGUE: int(max_memories * 0.08),     # 8% dialogue
            MemoryType.EMOTION: int(max_memories * 0.04),      # 4% emotions
            MemoryType.THEME: int(max_memories * 0.03)         # 3% themes
        }

        # Select memories according to target distribution
        selected_memories = []
        for memory_type, target_count in target_distribution.items():
            available_memories = memories_by_type.get(memory_type, [])
            selected_count = min(target_count, len(available_memories))
            selected_memories.extend(available_memories[:selected_count])

        # If we haven't reached max_memories, fill with highest scoring remaining memories
        if len(selected_memories) < max_memories:
            remaining_memories = []
            for memory_type, memory_list in memories_by_type.items():
                used_count = target_distribution.get(memory_type, 0)
                remaining_memories.extend(memory_list[used_count:])

            # Sort remaining by importance and add until we reach max
            remaining_memories.sort(key=lambda m: m.importance_score, reverse=True)
            needed = max_memories - len(selected_memories)
            selected_memories.extend(remaining_memories[:needed])

        return selected_memories

class MemoryRetriever:
    """Retrieves relevant memories based on input prompt"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_cache = {}

    def retrieve_memories(self, prompt: str, memories: List[Memory]) -> List[Memory]:
        """Retrieve relevant memories for a prompt (human-like activation)"""
        if not memories:
            return []

        # Calculate relevance scores
        memory_scores = []
        prompt_words = set(prompt.lower().split())

        for memory in memories:
            relevance = self._calculate_relevance(prompt_words, memory)

            # Add randomness factor (like human memory)
            randomness = random.random() * self.config.randomness_factor
            final_score = relevance + randomness

            if final_score >= self.config.relevance_threshold:
                memory_scores.append((memory, final_score))

        # Sort by score and return top memories
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        selected_memories = [mem for mem, score in memory_scores[:self.config.max_retrieved_memories]]

        return selected_memories

    def _calculate_relevance(self, prompt_words: set, memory: Memory) -> float:
        """Calculate how relevant a memory is to the prompt"""
        relevance = 0.0

        # Keyword matching
        memory_words = set(' '.join(memory.keywords).lower().split())
        common_words = prompt_words.intersection(memory_words)
        if memory_words:
            relevance += len(common_words) / len(memory_words) * 0.4

        # Content similarity (simple word overlap)
        content_words = set(memory.content.lower().split())
        content_overlap = prompt_words.intersection(content_words)
        if content_words:
            relevance += len(content_overlap) / len(content_words) * 0.3

        # Memory type bonuses
        prompt_lower = ' '.join(prompt_words).lower()
        if any(word in prompt_lower for word in ['who', 'character']):
            if memory.memory_type == MemoryType.CHARACTER:
                relevance += 0.2

        if any(word in prompt_lower for word in ['where', 'place', 'location']):
            if memory.memory_type == MemoryType.LOCATION:
                relevance += 0.2

        # Importance score contribution
        relevance += memory.importance_score * 0.1

        return min(1.0, relevance)

class EpisodicMemorySystem:
    """Main episodic memory system for Model Tea"""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.extractor = MemoryExtractor(self.config)
        self.retriever = MemoryRetriever(self.config)
        self.model_memories = {}  # Dict of model_name -> List[Memory]

    def build_memory_for_model(self, model_name: str, novel_path: Path) -> Dict[str, Any]:
        """Build episodic memory for a trained model"""
        print(f"Building episodic memory for {model_name}...")
        start_time = time.time()

        # Load novel text
        novel_files = list(novel_path.glob("*.txt"))
        if not novel_files:
            raise FileNotFoundError(f"No novel text found in {novel_path}")

        with open(novel_files[0], 'r', encoding='utf-8') as f:
            novel_text = f.read()

        # Extract memories
        memories = self.extractor.extract_memories(novel_text, model_name)

        # Store memories
        self.model_memories[model_name] = memories

        # Save memory system
        memory_dir = Path("episodic_memories") / model_name
        memory_dir.mkdir(parents=True, exist_ok=True)

        with open(memory_dir / "memories.pkl", 'wb') as f:
            pickle.dump(memories, f)

        # Create analysis
        analysis = self._analyze_memory_system(memories, novel_text)

        with open(memory_dir / "memory_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        build_time = time.time() - start_time

        print(f"Built {len(memories)} memories in {build_time:.1f}s")
        return analysis

    def load_memory_for_model(self, model_name: str) -> bool:
        """Load pre-built memory for a model"""
        memory_path = Path("episodic_memories") / model_name / "memories.pkl"

        if memory_path.exists():
            with open(memory_path, 'rb') as f:
                self.model_memories[model_name] = pickle.load(f)
            return True
        return False

    def activate_memories(self, model_name: str, prompt: str) -> Tuple[List[Memory], Dict[str, Any]]:
        """Activate relevant memories for a prompt (human-like memory recall)"""
        if model_name not in self.model_memories:
            if not self.load_memory_for_model(model_name):
                return [], {"error": "No memories available"}

        memories = self.model_memories[model_name]
        activated_memories = self.retriever.retrieve_memories(prompt, memories)

        # Create activation analysis
        activation_analysis = {
            "prompt_words": len(prompt.split()),
            "total_memories": len(memories),
            "activated_count": len(activated_memories),
            "memory_types": [m.memory_type for m in activated_memories],
            "emotional_tones": [m.emotional_tone for m in activated_memories],
            "activation_randomness": self.config.randomness_factor
        }

        return activated_memories, activation_analysis

    def _analyze_memory_system(self, memories: List[Memory], novel_text: str) -> Dict[str, Any]:
        """Analyze the built memory system"""
        type_counts = defaultdict(int)
        emotion_counts = defaultdict(int)
        position_distribution = []

        for memory in memories:
            type_counts[memory.memory_type] += 1
            emotion_counts[memory.emotional_tone] += 1
            position_distribution.append(memory.chapter_position)

        return {
            "total_memories": len(memories),
            "novel_word_count": len(novel_text.split()),
            "memory_density": len(memories) / len(novel_text.split()) * 1000,  # memories per 1000 words
            "memory_types": dict(type_counts),
            "emotional_distribution": dict(emotion_counts),
            "average_importance": sum(m.importance_score for m in memories) / len(memories) if memories else 0,
            "position_coverage": {
                "min": min(position_distribution) if position_distribution else 0,
                "max": max(position_distribution) if position_distribution else 0,
                "mean": sum(position_distribution) / len(position_distribution) if position_distribution else 0
            },
            "memory_system_config": {
                "max_memories": self.config.max_memories_per_novel,
                "chunk_size": self.config.memory_chunk_size,
                "retrieval_limit": self.config.max_retrieved_memories,
                "randomness_factor": self.config.randomness_factor
            }
        }

def main():
    """Test the episodic memory system"""
    print("Model Tea - Episodic Memory System Test")
    print("=" * 40)

    config = MemoryConfig()
    memory_system = EpisodicMemorySystem(config)

    # Test with a novel
    novel_path = Path("novels/agony_column")  # Use existing trained novel
    if novel_path.exists():
        analysis = memory_system.build_memory_for_model("agony_column", novel_path)
        print(f"Memory system built with {analysis['total_memories']} memories")

        # Test memory activation
        test_prompts = [
            "Tell me about the characters",
            "What was the setting like?",
            "Describe an emotional scene",
            "What happened at the end?"
        ]

        for prompt in test_prompts:
            memories, activation = memory_system.activate_memories("agony_column", prompt)
            print(f"\nPrompt: '{prompt}'")
            print(f"Activated {len(memories)} memories: {activation['memory_types']}")
    else:
        print("No trained novel found for testing")

if __name__ == "__main__":
    main()