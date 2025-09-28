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
    use_simple_similarity: bool = True
    memory_cache_size: int = 100

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
    context: str
    emotional_tone: str
    importance_score: float
    chapter_position: float

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
            if len(chunk_words) >= 20:
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
        """Rank memories by importance with adaptive distribution and temporal balance"""
        from collections import defaultdict

        # Group memories by type and temporal section
        memories_by_type = defaultdict(list)
        memories_by_section = defaultdict(list)

        for memory in memories:
            memories_by_type[memory.memory_type].append(memory)
            # Categorize by temporal position: beginning (0-0.33), middle (0.33-0.66), end (0.66-1.0)
            if memory.chapter_position <= 0.33:
                section = "beginning"
            elif memory.chapter_position <= 0.66:
                section = "middle"
            else:
                section = "end"
            memories_by_section[section].append(memory)

        # Analyze content to determine adaptive distribution
        adaptive_distribution = self._calculate_adaptive_distribution(memories_by_type)

        # Sort each type by enhanced importance scoring
        for memory_type in memories_by_type:
            memories_by_type[memory_type].sort(key=lambda m: self._enhanced_importance_score(m), reverse=True)

        # Select memories with temporal and type balance
        selected_memories = self._select_temporally_balanced_memories(
            memories_by_type, memories_by_section, adaptive_distribution
        )

        return selected_memories[:self.config.max_memories_per_novel]

    def _calculate_adaptive_distribution(self, memories_by_type: Dict) -> Dict[str, float]:
        """Calculate adaptive distribution based on content richness"""
        total_memories = sum(len(memories) for memories in memories_by_type.values())
        if total_memories == 0:
            return {}

        # Base distribution ratios
        base_ratios = {
            MemoryType.DESCRIPTION: 0.45,  # Reduced from 60%
            MemoryType.CHARACTER: 0.20,    # Increased from 10%
            MemoryType.DIALOGUE: 0.15,     # Increased from 8%
            MemoryType.LOCATION: 0.10,     # Reduced from 15%
            MemoryType.EMOTION: 0.06,      # Increased from 4%
            MemoryType.THEME: 0.04         # Increased from 3%
        }

        # Adapt based on content richness
        adaptive_distribution = {}
        max_memories = self.config.max_memories_per_novel

        for memory_type, base_ratio in base_ratios.items():
            available_count = len(memories_by_type.get(memory_type, []))
            # If type is rare, reduce allocation; if abundant, can maintain or increase
            richness_factor = min(1.5, available_count / (total_memories * base_ratio + 1))
            adaptive_ratio = base_ratio * richness_factor
            adaptive_distribution[memory_type] = int(max_memories * adaptive_ratio)

        # Ensure total doesn't exceed max_memories
        total_allocated = sum(adaptive_distribution.values())
        if total_allocated > max_memories:
            scale_factor = max_memories / total_allocated
            for memory_type in adaptive_distribution:
                adaptive_distribution[memory_type] = int(adaptive_distribution[memory_type] * scale_factor)

        return adaptive_distribution

    def _enhanced_importance_score(self, memory: Memory) -> float:
        """Enhanced importance scoring with context awareness"""
        base_score = memory.importance_score

        # Temporal diversity bonus (favor memories from different narrative sections)
        temporal_bonus = 0.0
        if memory.chapter_position <= 0.2 or memory.chapter_position >= 0.8:
            temporal_bonus = 0.1  # Bonus for beginning/end
        elif 0.4 <= memory.chapter_position <= 0.6:
            temporal_bonus = 0.05  # Small bonus for middle

        # Content richness bonus
        content_bonus = 0.0
        if len(memory.content) > 100:  # Longer, more detailed memories
            content_bonus = 0.1

        # Emotional intensity bonus
        emotion_bonus = 0.0
        if memory.emotional_tone in ["positive", "negative"]:  # vs "neutral"
            emotion_bonus = 0.05

        # Keyword diversity bonus
        keyword_bonus = min(0.1, len(memory.keywords) * 0.02)

        return base_score + temporal_bonus + content_bonus + emotion_bonus + keyword_bonus

    def _select_temporally_balanced_memories(self, memories_by_type: Dict, memories_by_section: Dict, distribution: Dict) -> List[Memory]:
        """Select memories ensuring temporal balance across beginning/middle/end"""
        selected_memories = []

        # Target temporal distribution: 30% beginning, 40% middle, 30% end
        temporal_targets = {"beginning": 0.30, "middle": 0.40, "end": 0.30}

        # For each memory type, select from different temporal sections
        for memory_type, target_count in distribution.items():
            available_memories = memories_by_type.get(memory_type, [])
            if not available_memories:
                continue

            # Group available memories by temporal section
            type_by_section = defaultdict(list)
            for memory in available_memories:
                if memory.chapter_position <= 0.33:
                    section = "beginning"
                elif memory.chapter_position <= 0.66:
                    section = "middle"
                else:
                    section = "end"
                type_by_section[section].append(memory)

            # Select memories maintaining temporal balance
            selected_for_type = []
            remaining_to_select = target_count

            for section, section_ratio in temporal_targets.items():
                section_memories = type_by_section.get(section, [])
                section_target = int(target_count * section_ratio)
                section_selected = min(section_target, len(section_memories), remaining_to_select)

                # Sort by enhanced importance and select top ones
                section_memories.sort(key=lambda m: self._enhanced_importance_score(m), reverse=True)
                selected_for_type.extend(section_memories[:section_selected])
                remaining_to_select -= section_selected

                if remaining_to_select <= 0:
                    break

            # If still need more memories, fill from any remaining
            if remaining_to_select > 0:
                remaining_memories = []
                for section_memories in type_by_section.values():
                    remaining_memories.extend(section_memories)

                # Remove already selected
                remaining_memories = [m for m in remaining_memories if m not in selected_for_type]
                remaining_memories.sort(key=lambda m: self._enhanced_importance_score(m), reverse=True)
                selected_for_type.extend(remaining_memories[:remaining_to_select])

            selected_memories.extend(selected_for_type)

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

        # Save memory system to standardized location
        # Primary location: episodic_memories/{model_name}/
        primary_memory_dir = Path("episodic_memories") / model_name
        primary_memory_dir.mkdir(parents=True, exist_ok=True)

        # Secondary location: iterative_models/{model_name}/memory/
        secondary_memory_dir = Path("iterative_models") / model_name / "memory"
        secondary_memory_dir.mkdir(parents=True, exist_ok=True)

        # Create analysis
        analysis = self._analyze_memory_system(memories, novel_text)

        # Save to both locations for compatibility (using secure JSON serialization)
        for memory_dir in [primary_memory_dir, secondary_memory_dir]:
            # Convert memories to serializable format
            serializable_memories = []
            for memory in memories:
                serializable_memories.append({
                    'content': memory.content,
                    'memory_type': memory.memory_type,
                    'keywords': memory.keywords,
                    'context': memory.context,
                    'emotional_tone': memory.emotional_tone,
                    'importance_score': memory.importance_score,
                    'chapter_position': memory.chapter_position
                })

            with open(memory_dir / "memories.json", 'w', encoding='utf-8') as f:
                json.dump(serializable_memories, f, indent=2, ensure_ascii=False)

            with open(memory_dir / "memory_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

        build_time = time.time() - start_time

        print(f"Built {len(memories)} memories in {build_time:.1f}s")
        return analysis

    def load_memory_for_model(self, model_name: str) -> bool:
        """Load pre-built memory for a model"""
        memory_path = Path("episodic_memories") / model_name / "memories.json"

        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    serialized_memories = json.load(f)

                # Convert back to Memory objects
                memories = []
                for mem_data in serialized_memories:
                    memory = Memory(
                        content=mem_data['content'],
                        memory_type=mem_data['memory_type'],
                        keywords=mem_data['keywords'],
                        context=mem_data['context'],
                        emotional_tone=mem_data['emotional_tone'],
                        importance_score=mem_data['importance_score'],
                        chapter_position=mem_data['chapter_position']
                    )
                    memories.append(memory)

                self.model_memories[model_name] = memories
                return True
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load memories for {model_name}: {e}")
                return False
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