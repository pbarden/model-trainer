#!/usr/bin/env python3
"""
Merge Lovecraft Mythos Models

Combines lovecraft_1mythos and lovecraft_2mythos back into
a single lovecraft_mythos model with all 8 novels.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil


def merge_lovecraft_models():
    """Merge the two Lovecraft models into one"""

    mapping_file = Path("model_mapping.json")

    if not mapping_file.exists():
        print("Error: model_mapping.json not found!")
        return

    # Load existing mapping
    print("Loading model_mapping.json...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    models = mapping.get('models', {})

    # Find the two Lovecraft models
    lovecraft_1 = models.get('lovecraft_1mythos')
    lovecraft_2 = models.get('lovecraft_2mythos')

    if not lovecraft_1 or not lovecraft_2:
        print("Error: Could not find both lovecraft_1mythos and lovecraft_2mythos")
        return

    print("\nFound Lovecraft models:")
    print(f"  lovecraft_1mythos: {lovecraft_1['novel_count']} novels, {lovecraft_1['total_word_count']:,} words")
    print(f"  lovecraft_2mythos: {lovecraft_2['novel_count']} novels, {lovecraft_2['total_word_count']:,} words")

    # Backup
    backup_file = f"model_mapping_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(mapping_file, backup_file)
    print(f"\nBacked up to {backup_file}")

    # Merge novels
    all_novels = lovecraft_1['novels'] + lovecraft_2['novels']
    total_words = lovecraft_1['total_word_count'] + lovecraft_2['total_word_count']

    print(f"\nMerging into single model:")
    print(f"  Total novels: {len(all_novels)}")
    print(f"  Total words: {total_words:,}")

    # Calculate training parameters for combined model
    from fix_training_parameters import AdaptiveTrainingParameters
    params = AdaptiveTrainingParameters.calculate_parameters(total_words)

    # Merge tier distribution
    tier_dist = {}
    for tier, count in lovecraft_1.get('tier_distribution', {}).items():
        tier_dist[tier] = tier_dist.get(tier, 0) + count
    for tier, count in lovecraft_2.get('tier_distribution', {}).items():
        tier_dist[tier] = tier_dist.get(tier, 0) + count

    # Create merged model
    merged_model = {
        "description": "H.P. Lovecraft Cosmic Horror Collection",
        "base_name": "mythos",
        "original_model": "lovecraft_mythos",
        "novel_count": len(all_novels),
        "total_word_count": total_words,
        "training_parameters": {
            "max_iterations": params['iterations'],
            "max_steps_per_iteration": params['steps_per_iteration'],
            "learning_rate_start": params['learning_rate_start'],
            "learning_rate_end": params['learning_rate_end'],
            "size_category": params['category']
        },
        "tier_distribution": tier_dist,
        "novels": all_novels
    }

    print(f"\nTraining parameters:")
    print(f"  Iterations: {params['iterations']}")
    print(f"  Steps/iteration: {params['steps_per_iteration']}")
    print(f"  Category: {params['category']}")

    # Remove old models and add merged one
    del models['lovecraft_1mythos']
    del models['lovecraft_2mythos']
    models['lovecraft_mythos'] = merged_model

    # Update metadata
    mapping['metadata']['total_models'] = len(models)
    mapping['metadata']['last_updated'] = datetime.now().isoformat()
    mapping['metadata']['update_reason'] = 'Merged lovecraft_1mythos and lovecraft_2mythos into lovecraft_mythos'

    # Save
    print(f"\nSaving updated model_mapping.json...")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

    print("\nCompleted!")
    print(f"Removed: lovecraft_1mythos, lovecraft_2mythos")
    print(f"Added: lovecraft_mythos (8 novels, {total_words:,} words)")
    print(f"\nNovels in merged model:")
    for i, novel in enumerate(all_novels, 1):
        print(f"  {i}. {novel['original_name']} ({novel['word_count']:,} words)")


if __name__ == "__main__":
    print("=" * 80)
    print("MERGE LOVECRAFT MYTHOS MODELS")
    print("=" * 80)
    print()

    merge_lovecraft_models()

    print("\n" + "=" * 80)
    print("Model mapping updated successfully!")
    print("=" * 80)
    print()
