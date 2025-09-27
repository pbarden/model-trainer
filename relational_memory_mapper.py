#!/usr/bin/env python3
"""
Model Tea - Relational Memory Mapping System
Copyright © ChaiQ LLC

Creates cross-novel memory relationships and thematic connections for combined models.
Analyzes relationships between novels in the same combined model and builds
enhanced memory systems with cross-references.
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Model Tea utilities
from model_tea_utils import (
    ModelTeaConfig, FileSystemUtils, TextProcessingUtils,
    ErrorHandling, validate_system_setup
)

# Import episodic memory system
try:
    from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    EpisodicMemorySystem = None
    MemoryConfig = None
    MEMORY_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RelationalConfig:
    """Configuration for relational memory mapping"""
    # Analysis thresholds
    similarity_threshold: float = 0.3
    theme_overlap_threshold: float = 0.2
    character_similarity_threshold: float = 0.4

    # Cross-reference limits
    max_cross_references_per_novel: int = 5
    max_thematic_connections: int = 10
    max_character_mappings: int = 8

    # Analysis depth
    analyze_themes: bool = True
    analyze_characters: bool = True
    analyze_narrative_patterns: bool = True
    analyze_stylistic_elements: bool = True

    # Memory enhancement
    create_enhanced_memories: bool = True
    cross_novel_memory_percentage: float = 0.2  # 20% of memories are cross-novel


class RelationalMemoryMapper:
    """
    Creates cross-novel memory relationships and thematic connections
    """

    def __init__(self, config: RelationalConfig = None):
        self.config = config or RelationalConfig()
        self.model_mapping = self._load_model_mapping()

        # Setup directories
        self.novels_dir = Path("novels")
        self.models_dir = Path("iterative_models")
        self.memory_dir = Path("relational_memories")

        # Create memory directory
        FileSystemUtils.ensure_directory(self.memory_dir)

        # Common word patterns for analysis
        self.theme_keywords = self._load_theme_keywords()
        self.character_indicators = self._load_character_indicators()

        # Validate system setup
        validate_system_setup()

    def _load_model_mapping(self) -> Dict[str, Any]:
        """Load model mapping configuration"""
        mapping_file = Path("model_mapping.json")
        if not mapping_file.exists():
            raise FileNotFoundError("model_mapping.json not found.")

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model_mapping.json: {e}")

    def _load_theme_keywords(self) -> Dict[str, List[str]]:
        """Load theme analysis keywords"""
        return {
            "horror": ["fear", "terror", "monster", "darkness", "nightmare", "evil", "death", "haunted", "ghost", "demon"],
            "mystery": ["detective", "clue", "investigation", "murder", "crime", "suspect", "evidence", "solve", "mystery"],
            "adventure": ["journey", "quest", "explore", "travel", "danger", "rescue", "escape", "treasure", "discover"],
            "romance": ["love", "heart", "romance", "passion", "kiss", "marriage", "wedding", "beloved", "affection"],
            "supernatural": ["magic", "spell", "witch", "wizard", "power", "supernatural", "mystical", "enchanted"],
            "war": ["battle", "war", "soldier", "fight", "army", "weapon", "conflict", "victory", "defeat"],
            "science": ["experiment", "discovery", "scientist", "laboratory", "invention", "research", "technology"],
            "political": ["government", "power", "corruption", "revolution", "politics", "ruler", "empire", "rebellion"]
        }

    def _load_character_indicators(self) -> List[str]:
        """Load character analysis indicators"""
        return [
            "protagonist", "hero", "villain", "detective", "doctor", "professor", "captain", "lord", "lady",
            "king", "queen", "prince", "princess", "priest", "minister", "judge", "lawyer", "soldier",
            "scientist", "inventor", "explorer", "merchant", "servant", "master", "student", "teacher"
        ]

    def get_available_models(self) -> List[str]:
        """Get list of available combined models"""
        if "models" not in self.model_mapping:
            return []
        return list(self.model_mapping["models"].keys())

    def analyze_thematic_similarities(self, novels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze thematic similarities between novels"""
        logger.info("Analyzing thematic similarities...")

        novel_themes = {}

        # Analyze themes for each novel
        for novel_info in novels:
            directory_name = novel_info["directory_name"]
            original_name = novel_info["original_name"]

            try:
                content = self._load_novel_content(directory_name)
                themes = self._extract_themes(content)
                novel_themes[directory_name] = {
                    "original_name": original_name,
                    "themes": themes,
                    "word_count": len(content.split())
                }
                logger.info(f"  {original_name}: {len(themes)} themes identified")
            except Exception as e:
                logger.warning(f"Failed to analyze themes for {directory_name}: {e}")
                continue

        # Find thematic connections
        thematic_connections = self._find_thematic_connections(novel_themes)

        return {
            "novel_themes": novel_themes,
            "thematic_connections": thematic_connections,
            "theme_summary": self._summarize_themes(novel_themes)
        }

    def _extract_themes(self, content: str) -> Dict[str, float]:
        """Extract themes from novel content"""
        content_lower = content.lower()
        word_count = len(content.split())
        themes = {}

        for theme_name, keywords in self.theme_keywords.items():
            keyword_count = 0
            for keyword in keywords:
                keyword_count += len(re.findall(r'\b' + keyword + r'\b', content_lower))

            # Calculate theme strength (keywords per 1000 words)
            theme_strength = (keyword_count / word_count) * 1000 if word_count > 0 else 0

            if theme_strength > self.config.similarity_threshold:
                themes[theme_name] = theme_strength

        return themes

    def _find_thematic_connections(self, novel_themes: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Find connections between novels based on themes"""
        connections = []

        novel_list = list(novel_themes.keys())
        for i, novel1 in enumerate(novel_list):
            for novel2 in novel_list[i+1:]:
                themes1 = set(novel_themes[novel1]["themes"].keys())
                themes2 = set(novel_themes[novel2]["themes"].keys())

                common_themes = themes1.intersection(themes2)

                if len(common_themes) > 0:
                    # Calculate connection strength
                    total_themes = len(themes1.union(themes2))
                    connection_strength = len(common_themes) / total_themes if total_themes > 0 else 0

                    if connection_strength >= self.config.theme_overlap_threshold:
                        connections.append({
                            "novel1": novel_themes[novel1]["original_name"],
                            "novel2": novel_themes[novel2]["original_name"],
                            "novel1_dir": novel1,
                            "novel2_dir": novel2,
                            "common_themes": list(common_themes),
                            "connection_strength": connection_strength,
                            "connection_type": "thematic"
                        })

        # Sort by connection strength
        connections.sort(key=lambda x: x["connection_strength"], reverse=True)
        return connections[:self.config.max_thematic_connections]

    def _summarize_themes(self, novel_themes: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize themes across all novels"""
        all_themes = Counter()

        for novel_data in novel_themes.values():
            for theme, strength in novel_data["themes"].items():
                all_themes[theme] += strength

        return {
            "dominant_themes": dict(all_themes.most_common(5)),
            "total_unique_themes": len(all_themes),
            "average_themes_per_novel": sum(len(novel["themes"]) for novel in novel_themes.values()) / len(novel_themes)
        }

    def analyze_character_relationships(self, novels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze character relationships and archetypes"""
        logger.info("Analyzing character relationships...")

        novel_characters = {}

        # Analyze characters for each novel
        for novel_info in novels:
            directory_name = novel_info["directory_name"]
            original_name = novel_info["original_name"]

            try:
                content = self._load_novel_content(directory_name)
                characters = self._extract_character_archetypes(content)
                novel_characters[directory_name] = {
                    "original_name": original_name,
                    "character_archetypes": characters
                }
                logger.info(f"  {original_name}: {len(characters)} character archetypes")
            except Exception as e:
                logger.warning(f"Failed to analyze characters for {directory_name}: {e}")
                continue

        # Find character archetype connections
        character_connections = self._find_character_connections(novel_characters)

        return {
            "novel_characters": novel_characters,
            "character_connections": character_connections,
            "archetype_summary": self._summarize_character_archetypes(novel_characters)
        }

    def _extract_character_archetypes(self, content: str) -> Dict[str, int]:
        """Extract character archetypes from content"""
        content_lower = content.lower()
        archetypes = {}

        for archetype in self.character_indicators:
            count = len(re.findall(r'\b' + archetype + r'\b', content_lower))
            if count > 0:
                archetypes[archetype] = count

        return archetypes

    def _find_character_connections(self, novel_characters: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Find connections based on character archetypes"""
        connections = []

        novel_list = list(novel_characters.keys())
        for i, novel1 in enumerate(novel_list):
            for novel2 in novel_list[i+1:]:
                archetypes1 = set(novel_characters[novel1]["character_archetypes"].keys())
                archetypes2 = set(novel_characters[novel2]["character_archetypes"].keys())

                common_archetypes = archetypes1.intersection(archetypes2)

                if len(common_archetypes) > 0:
                    total_archetypes = len(archetypes1.union(archetypes2))
                    connection_strength = len(common_archetypes) / total_archetypes if total_archetypes > 0 else 0

                    if connection_strength >= self.config.character_similarity_threshold:
                        connections.append({
                            "novel1": novel_characters[novel1]["original_name"],
                            "novel2": novel_characters[novel2]["original_name"],
                            "novel1_dir": novel1,
                            "novel2_dir": novel2,
                            "common_archetypes": list(common_archetypes),
                            "connection_strength": connection_strength,
                            "connection_type": "character_archetype"
                        })

        connections.sort(key=lambda x: x["connection_strength"], reverse=True)
        return connections[:self.config.max_character_mappings]

    def _summarize_character_archetypes(self, novel_characters: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize character archetypes across novels"""
        all_archetypes = Counter()

        for novel_data in novel_characters.values():
            for archetype, count in novel_data["character_archetypes"].items():
                all_archetypes[archetype] += count

        return {
            "most_common_archetypes": dict(all_archetypes.most_common(10)),
            "total_unique_archetypes": len(all_archetypes),
            "average_archetypes_per_novel": sum(len(novel["character_archetypes"]) for novel in novel_characters.values()) / len(novel_characters)
        }

    def analyze_narrative_patterns(self, novels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze narrative patterns and structures"""
        logger.info("Analyzing narrative patterns...")

        narrative_analysis = {}

        for novel_info in novels:
            directory_name = novel_info["directory_name"]
            original_name = novel_info["original_name"]

            try:
                content = self._load_novel_content(directory_name)
                patterns = self._extract_narrative_patterns(content)
                narrative_analysis[directory_name] = {
                    "original_name": original_name,
                    "patterns": patterns
                }
            except Exception as e:
                logger.warning(f"Failed to analyze narrative patterns for {directory_name}: {e}")
                continue

        # Find pattern connections
        pattern_connections = self._find_pattern_connections(narrative_analysis)

        return {
            "narrative_patterns": narrative_analysis,
            "pattern_connections": pattern_connections,
            "pattern_summary": self._summarize_narrative_patterns(narrative_analysis)
        }

    def _extract_narrative_patterns(self, content: str) -> Dict[str, Any]:
        """Extract narrative patterns from content"""
        sentences = content.split('.')
        word_count = len(content.split())

        return {
            "average_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            "dialogue_percentage": self._calculate_dialogue_percentage(content),
            "description_density": self._calculate_description_density(content),
            "pacing_indicators": self._analyze_pacing(content),
            "narrative_style": self._classify_narrative_style(content)
        }

    def _calculate_dialogue_percentage(self, content: str) -> float:
        """Calculate percentage of content that is dialogue"""
        dialogue_markers = ['"', "'", "said", "asked", "replied", "whispered", "shouted"]
        dialogue_indicators = 0

        for marker in dialogue_markers:
            dialogue_indicators += content.count(marker)

        total_words = len(content.split())
        return (dialogue_indicators / total_words) * 100 if total_words > 0 else 0

    def _calculate_description_density(self, content: str) -> float:
        """Calculate density of descriptive language"""
        descriptive_words = ["beautiful", "dark", "tall", "small", "ancient", "modern", "mysterious", "bright"]
        descriptive_count = 0

        content_lower = content.lower()
        for word in descriptive_words:
            descriptive_count += len(re.findall(r'\b' + word + r'\b', content_lower))

        total_words = len(content.split())
        return (descriptive_count / total_words) * 100 if total_words > 0 else 0

    def _analyze_pacing(self, content: str) -> Dict[str, float]:
        """Analyze pacing indicators"""
        action_words = ["suddenly", "quickly", "rushed", "ran", "jumped", "fought", "screamed"]
        contemplative_words = ["thought", "considered", "pondered", "reflected", "remembered"]

        content_lower = content.lower()
        total_words = len(content.split())

        action_count = sum(len(re.findall(r'\b' + word + r'\b', content_lower)) for word in action_words)
        contemplative_count = sum(len(re.findall(r'\b' + word + r'\b', content_lower)) for word in contemplative_words)

        return {
            "action_density": (action_count / total_words) * 100 if total_words > 0 else 0,
            "contemplative_density": (contemplative_count / total_words) * 100 if total_words > 0 else 0
        }

    def _classify_narrative_style(self, content: str) -> str:
        """Classify the narrative style"""
        first_person_indicators = content.count(" I ") + content.count("I'") + content.count(" me ")
        third_person_indicators = content.count(" he ") + content.count(" she ") + content.count(" they ")

        if first_person_indicators > third_person_indicators * 1.5:
            return "first_person"
        elif third_person_indicators > first_person_indicators * 1.5:
            return "third_person"
        else:
            return "mixed"

    def _find_pattern_connections(self, narrative_analysis: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Find connections based on narrative patterns"""
        connections = []

        novel_list = list(narrative_analysis.keys())
        for i, novel1 in enumerate(novel_list):
            for novel2 in novel_list[i+1:]:
                pattern1 = narrative_analysis[novel1]["patterns"]
                pattern2 = narrative_analysis[novel2]["patterns"]

                # Compare narrative styles
                if pattern1["narrative_style"] == pattern2["narrative_style"]:
                    similarity = self._calculate_pattern_similarity(pattern1, pattern2)

                    if similarity >= self.config.similarity_threshold:
                        connections.append({
                            "novel1": narrative_analysis[novel1]["original_name"],
                            "novel2": narrative_analysis[novel2]["original_name"],
                            "novel1_dir": novel1,
                            "novel2_dir": novel2,
                            "similarity_score": similarity,
                            "shared_style": pattern1["narrative_style"],
                            "connection_type": "narrative_pattern"
                        })

        connections.sort(key=lambda x: x["similarity_score"], reverse=True)
        return connections

    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between narrative patterns"""
        similarities = []

        # Compare quantitative measures
        for key in ["average_sentence_length", "dialogue_percentage", "description_density"]:
            if key in pattern1 and key in pattern2:
                val1, val2 = pattern1[key], pattern2[key]
                max_val = max(val1, val2)
                if max_val > 0:
                    similarity = 1 - abs(val1 - val2) / max_val
                    similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0

    def _summarize_narrative_patterns(self, narrative_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize narrative patterns across novels"""
        styles = Counter()
        total_dialogue = 0
        total_description = 0

        for novel_data in narrative_analysis.values():
            patterns = novel_data["patterns"]
            styles[patterns["narrative_style"]] += 1
            total_dialogue += patterns["dialogue_percentage"]
            total_description += patterns["description_density"]

        novel_count = len(narrative_analysis)

        return {
            "dominant_narrative_style": styles.most_common(1)[0][0] if styles else "unknown",
            "average_dialogue_percentage": total_dialogue / novel_count if novel_count > 0 else 0,
            "average_description_density": total_description / novel_count if novel_count > 0 else 0,
            "style_distribution": dict(styles)
        }

    def _load_novel_content(self, directory_name: str) -> str:
        """Load content from a novel directory"""
        novel_path = self.novels_dir / directory_name

        if not novel_path.exists():
            raise FileNotFoundError(f"Novel directory not found: {novel_path}")

        # Look for content files
        content_files = ["content.txt", f"{directory_name}.txt", "novel.txt"]

        for filename in content_files:
            file_path = novel_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    continue

        raise ValueError(f"No readable content found in {novel_path}")

    def build_relational_memory_table(self, model_key: str) -> Dict[str, Any]:
        """Build comprehensive relational memory table for a model"""
        logger.info(f"Building relational memory table for: {model_key}")

        # Get novel list for this model
        if "models" not in self.model_mapping or model_key not in self.model_mapping["models"]:
            raise ValueError(f"Model '{model_key}' not found in mapping")

        novels = self.model_mapping["models"][model_key]["novels"]

        # Perform all analyses
        relational_data = {
            "model_key": model_key,
            "model_description": self.model_mapping["models"][model_key].get("description", ""),
            "novel_count": len(novels),
            "novels": [novel["original_name"] for novel in novels],
            "analysis_timestamp": time.time()
        }

        if self.config.analyze_themes:
            relational_data["thematic_analysis"] = self.analyze_thematic_similarities(novels)

        if self.config.analyze_characters:
            relational_data["character_analysis"] = self.analyze_character_relationships(novels)

        if self.config.analyze_narrative_patterns:
            relational_data["narrative_analysis"] = self.analyze_narrative_patterns(novels)

        # Create unified cross-references
        relational_data["cross_references"] = self._create_unified_cross_references(relational_data)

        # Generate memory enhancement suggestions
        relational_data["memory_enhancements"] = self._generate_memory_enhancements(relational_data)

        return relational_data

    def _create_unified_cross_references(self, relational_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Create unified cross-reference table from all analyses"""
        cross_refs = defaultdict(list)

        # Add thematic cross-references
        if "thematic_analysis" in relational_data:
            for connection in relational_data["thematic_analysis"].get("thematic_connections", []):
                cross_refs[connection["novel1_dir"]].append({
                    "type": "thematic",
                    "target_novel": connection["novel2"],
                    "target_dir": connection["novel2_dir"],
                    "connection_strength": connection["connection_strength"],
                    "details": connection["common_themes"]
                })
                cross_refs[connection["novel2_dir"]].append({
                    "type": "thematic",
                    "target_novel": connection["novel1"],
                    "target_dir": connection["novel1_dir"],
                    "connection_strength": connection["connection_strength"],
                    "details": connection["common_themes"]
                })

        # Add character cross-references
        if "character_analysis" in relational_data:
            for connection in relational_data["character_analysis"].get("character_connections", []):
                cross_refs[connection["novel1_dir"]].append({
                    "type": "character",
                    "target_novel": connection["novel2"],
                    "target_dir": connection["novel2_dir"],
                    "connection_strength": connection["connection_strength"],
                    "details": connection["common_archetypes"]
                })
                cross_refs[connection["novel2_dir"]].append({
                    "type": "character",
                    "target_novel": connection["novel1"],
                    "target_dir": connection["novel1_dir"],
                    "connection_strength": connection["connection_strength"],
                    "details": connection["common_archetypes"]
                })

        # Limit cross-references per novel
        for novel_dir in cross_refs:
            cross_refs[novel_dir] = sorted(
                cross_refs[novel_dir],
                key=lambda x: x["connection_strength"],
                reverse=True
            )[:self.config.max_cross_references_per_novel]

        return dict(cross_refs)

    def _generate_memory_enhancements(self, relational_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggestions for memory system enhancements"""
        enhancements = {
            "cross_novel_memories": [],
            "thematic_clusters": [],
            "character_archetype_groups": [],
            "narrative_style_connections": []
        }

        # Cross-novel memory suggestions
        if "cross_references" in relational_data:
            for novel_dir, refs in relational_data["cross_references"].items():
                for ref in refs:
                    if ref["connection_strength"] > 0.5:  # High-strength connections
                        enhancements["cross_novel_memories"].append({
                            "source_novel": novel_dir,
                            "target_novel": ref["target_dir"],
                            "memory_type": f"cross_reference_{ref['type']}",
                            "content_suggestion": f"This {ref['type']} relates to {ref['target_novel']}: {ref['details']}"
                        })

        return enhancements

    def save_relational_mappings(self, model_key: str, mappings: Dict[str, Any]) -> str:
        """Save relational mappings to file"""
        output_file = self.memory_dir / f"{model_key}_relational_mappings.json"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, default=str)

            logger.info(f"Relational mappings saved to: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Failed to save relational mappings: {e}")
            raise

    def create_relational_mappings_for_model(self, model_key: str) -> Dict[str, Any]:
        """Create complete relational mappings for a model"""
        logger.info(f"Creating relational mappings for model: {model_key}")

        # Build relational memory table
        relational_data = self.build_relational_memory_table(model_key)

        # Save to file
        output_file = self.save_relational_mappings(model_key, relational_data)

        # Generate summary
        summary = {
            "model_key": model_key,
            "output_file": output_file,
            "novel_count": relational_data["novel_count"],
            "thematic_connections": len(relational_data.get("thematic_analysis", {}).get("thematic_connections", [])),
            "character_connections": len(relational_data.get("character_analysis", {}).get("character_connections", [])),
            "cross_references_created": sum(len(refs) for refs in relational_data.get("cross_references", {}).values()),
            "processing_time": time.time() - relational_data["analysis_timestamp"]
        }

        logger.info(f"Relational mapping completed for {model_key}:")
        logger.info(f"  - {summary['thematic_connections']} thematic connections")
        logger.info(f"  - {summary['character_connections']} character connections")
        logger.info(f"  - {summary['cross_references_created']} total cross-references")

        return summary

    def create_all_relational_mappings(self) -> Dict[str, Any]:
        """Create relational mappings for all models"""
        results = {}
        available_models = self.get_available_models()

        logger.info(f"Creating relational mappings for {len(available_models)} models...")

        for i, model_key in enumerate(available_models):
            logger.info(f"\n=== Processing Model {i+1}/{len(available_models)}: {model_key} ===")

            try:
                result = self.create_relational_mappings_for_model(model_key)
                results[model_key] = {"status": "success", "summary": result}
            except Exception as e:
                logger.error(f"Failed to create mappings for {model_key}: {e}")
                results[model_key] = {"status": "failed", "error": str(e)}

        return results


def main():
    """Main entry point for relational memory mapping"""
    parser = argparse.ArgumentParser(description="Model Tea - Relational Memory Mapping")
    parser.add_argument("--model", type=str, help="Specific model to analyze (e.g., vs_mintchip)")
    parser.add_argument("--all-models", action="store_true", help="Analyze all models")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    args = parser.parse_args()

    try:
        # Initialize mapper
        mapper = RelationalMemoryMapper()

        if args.list_models:
            models = mapper.get_available_models()
            print(f"\nAvailable Models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
            return

        if args.all_models:
            print("Creating relational mappings for all models...")
            results = mapper.create_all_relational_mappings()

            # Summary
            success_count = sum(1 for r in results.values() if r["status"] == "success")
            failed_count = sum(1 for r in results.values() if r["status"] == "failed")

            print(f"\n=== Mapping Summary ===")
            print(f"Success: {success_count}")
            print(f"Failed: {failed_count}")

        elif args.model:
            print(f"Creating relational mappings for: {args.model}")
            result = mapper.create_relational_mappings_for_model(args.model)
            print(f"Mapping completed successfully for {args.model}")

        else:
            # No specific arguments - show help
            parser.print_help()

            # Show available models
            models = mapper.get_available_models()
            print(f"\nAvailable models: {', '.join(models)}")
            print(f"\nExample usage:")
            print(f"  python relational_memory_mapper.py --model vs_mintchip")
            print(f"  python relational_memory_mapper.py --all-models")

    except Exception as e:
        logger.error(f"Relational mapping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()