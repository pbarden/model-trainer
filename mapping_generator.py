#!/usr/bin/env python3
"""
Model Tea - Mapping Generator
Copyright © ChaiQ LLC

Creates optimized model categorization mappings with word count analysis
and chunking strategy recommendations. Validates novel availability and
generates production-ready training configurations.
"""

import json
from pathlib import Path

def load_novel_analysis_data():
    """Load word count and analysis data from processed novels"""
    novels_dir = Path("novels")
    novel_data = {}

    for novel_dir in novels_dir.iterdir():
        if novel_dir.is_dir():
            analysis_file = novel_dir / "analysis.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                    novel_data[novel_dir.name] = {
                        "word_count": analysis.get("word_count", 0),
                        "character_count": analysis.get("character_count", 0),
                        "sentence_count": analysis.get("sentence_count", 0),
                        "original_filename": analysis.get("original_filename", ""),
                        "cleaned_filename": analysis.get("cleaned_filename", "")
                    }

    return novel_data

def normalize_novel_name(name):
    """Convert novel name to our directory naming format"""
    # This matches the logic used in novel_processor.py
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ' -_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')

    safe_name = ''.join(safe_chars)
    safe_name = safe_name.replace(' ', '_')
    safe_name = '_'.join(part for part in safe_name.split('_') if part)

    return safe_name.lower()

def create_cleaned_mapping():
    """Create cleaned mapping with only available novels"""

    # Load existing mapping
    with open("model_categorization_mapping.json", 'r') as f:
        original_mapping = json.load(f)

    # Load novel analysis data
    novel_data = load_novel_analysis_data()
    available_novels = set(novel_data.keys())

    print(f"Available cleaned novels: {len(available_novels)}")

    cleaned_mapping = {
        "metadata": {
            "total_models": 0,
            "total_novels_assigned": 0,
            "total_word_count": 0,
            "created_by": "mapping_generator.py",
            "description": "Cleaned model categorization with only available novels and word counts"
        },
        "models": {}
    }

    total_novels_assigned = 0
    total_word_count = 0

    for model_name, model_info in original_mapping["models"].items():
        cleaned_novels = []
        model_word_count = 0

        for novel_name in model_info["novels"]:
            # Try to find matching cleaned novel
            normalized_name = normalize_novel_name(novel_name)

            # Try exact match first
            if normalized_name in available_novels:
                cleaned_novels.append({
                    "original_name": novel_name,
                    "directory_name": normalized_name,
                    "word_count": novel_data[normalized_name]["word_count"],
                    "sentence_count": novel_data[normalized_name]["sentence_count"]
                })
                model_word_count += novel_data[normalized_name]["word_count"]
            else:
                # Try some common variations
                variations = [
                    novel_name.lower().replace(' ', '_').replace("'", "").replace('"', ''),
                    novel_name.lower().replace(' ', '_').replace("'", "_").replace('"', '_'),
                    novel_name.lower().replace(' ', '_').replace('-', '_'),
                ]

                found = False
                for var in variations:
                    if var in available_novels:
                        cleaned_novels.append({
                            "original_name": novel_name,
                            "directory_name": var,
                            "word_count": novel_data[var]["word_count"],
                            "sentence_count": novel_data[var]["sentence_count"]
                        })
                        model_word_count += novel_data[var]["word_count"]
                        found = True
                        break

                if not found:
                    print(f"  Missing: {novel_name} -> {normalized_name}")

        if cleaned_novels:  # Only include models that have novels
            cleaned_mapping["models"][model_name] = {
                "description": model_info["description"],
                "novel_count": len(cleaned_novels),
                "total_word_count": model_word_count,
                "avg_words_per_novel": model_word_count // len(cleaned_novels) if cleaned_novels else 0,
                "novels": cleaned_novels,
                "chunking_strategy": determine_chunking_strategy(model_word_count, len(cleaned_novels))
            }

            total_novels_assigned += len(cleaned_novels)
            total_word_count += model_word_count

            print(f"{model_name}: {len(cleaned_novels)} novels, {model_word_count:,} total words")

    # Update metadata
    cleaned_mapping["metadata"]["total_models"] = len(cleaned_mapping["models"])
    cleaned_mapping["metadata"]["total_novels_assigned"] = total_novels_assigned
    cleaned_mapping["metadata"]["total_word_count"] = total_word_count

    # Save cleaned mapping
    with open("model_mapping.json", 'w') as f:
        json.dump(cleaned_mapping, f, indent=2)

    print(f"\nCleaned mapping created:")
    print(f"  Models: {len(cleaned_mapping['models'])}")
    print(f"  Total novels: {total_novels_assigned}")
    print(f"  Total words: {total_word_count:,}")
    print(f"  Avg words per model: {total_word_count // len(cleaned_mapping['models']):,}")

    return cleaned_mapping

def determine_chunking_strategy(total_words, novel_count):
    """Determine optimal chunking strategy based on total word count"""
    avg_words = total_words // novel_count if novel_count > 0 else 0

    if total_words < 100000:  # Less than 100k words
        return {
            "chunk_size": 1200,
            "overlap": 100,
            "max_chunks_per_novel": 50,
            "strategy": "small_dataset"
        }
    elif total_words < 500000:  # 100k-500k words
        return {
            "chunk_size": 1000,
            "overlap": 100,
            "max_chunks_per_novel": 40,
            "strategy": "medium_dataset"
        }
    else:  # Large datasets
        return {
            "chunk_size": 800,
            "overlap": 80,
            "max_chunks_per_novel": 30,
            "strategy": "large_dataset"
        }

if __name__ == "__main__":
    print("Generating optimized model mapping...")
    print("=" * 50)

    cleaned_mapping = create_cleaned_mapping()

    print("\nChunking strategies:")
    for model_name, model_data in cleaned_mapping["models"].items():
        strategy = model_data["chunking_strategy"]
        print(f"  {model_name}: {strategy['strategy']} ({strategy['chunk_size']} tokens)")