#!/usr/bin/env python3
"""
Fix Training Parameters in Existing model_mapping.json

Updates training parameters with corrected step counts without
redistributing the models.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class AdaptiveTrainingParameters:
    """Calculate corrected adaptive training parameters"""

    @staticmethod
    def calculate_parameters(total_words: int) -> Dict[str, Any]:
        """Calculate training parameters based on combined word count"""
        if total_words < 80000:
            return {
                'iterations': 8,
                'steps_per_iteration': 100,
                'learning_rate_start': 3e-5,
                'learning_rate_end': 8e-6,
                'category': 'tiny',
                'rationale': 'Small corpus, fewer iterations but sufficient steps'
            }
        elif total_words < 120000:
            return {
                'iterations': 10,
                'steps_per_iteration': 150,
                'learning_rate_start': 2.5e-5,
                'learning_rate_end': 6e-6,
                'category': 'small',
                'rationale': 'Small corpus, balanced training'
            }
        elif total_words < 180000:
            return {
                'iterations': 12,
                'steps_per_iteration': 200,
                'learning_rate_start': 2e-5,
                'learning_rate_end': 5e-6,
                'category': 'medium',
                'rationale': 'Optimal corpus size, quality focus'
            }
        elif total_words < 260000:
            return {
                'iterations': 14,
                'steps_per_iteration': 250,
                'learning_rate_start': 1.5e-5,
                'learning_rate_end': 4e-6,
                'category': 'large',
                'rationale': 'Large corpus, extended learning needed'
            }
        else:
            return {
                'iterations': 16,
                'steps_per_iteration': 300,
                'learning_rate_start': 1e-5,
                'learning_rate_end': 3e-6,
                'category': 'xlarge',
                'rationale': 'Very large corpus, maximum iterations and steps'
            }


def fix_training_parameters():
    """Fix training parameters in existing model_mapping.json"""

    mapping_file = Path("model_mapping.json")

    if not mapping_file.exists():
        print("Error: model_mapping.json not found!")
        return

    # Load existing mapping
    print("Loading model_mapping.json...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    models = mapping.get('models', {})
    print(f"Found {len(models)} models")

    # Backup
    backup_file = f"model_mapping_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import shutil
    shutil.copy(mapping_file, backup_file)
    print(f"Backed up to {backup_file}")

    # Update each model's parameters
    print("\nUpdating training parameters...")
    updated_count = 0

    for model_key, model_data in models.items():
        total_words = model_data.get('total_word_count', 0)

        if total_words == 0:
            print(f"  Warning: {model_key} has 0 words, skipping...")
            continue

        # Calculate correct parameters
        new_params = AdaptiveTrainingParameters.calculate_parameters(total_words)

        # Get old parameters for comparison
        old_params = model_data.get('training_parameters', {})
        old_steps = old_params.get('max_steps_per_iteration', 'N/A')
        old_iterations = old_params.get('max_iterations', 'N/A')

        # Update
        model_data['training_parameters'] = {
            'max_iterations': new_params['iterations'],
            'max_steps_per_iteration': new_params['steps_per_iteration'],
            'learning_rate_start': new_params['learning_rate_start'],
            'learning_rate_end': new_params['learning_rate_end'],
            'size_category': new_params['category']
        }

        print(f"  {model_key}:")
        print(f"    Words: {total_words:,}")
        print(f"    Old: {old_iterations} iter × {old_steps} steps")
        print(f"    New: {new_params['iterations']} iter × {new_params['steps_per_iteration']} steps ({new_params['category']})")

        updated_count += 1

    # Update metadata
    mapping['metadata']['last_updated'] = datetime.now().isoformat()
    mapping['metadata']['update_reason'] = 'Fixed training parameters (increased steps_per_iteration)'

    # Save
    print(f"\nSaving updated model_mapping.json...")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nCompleted!")
    print(f"Updated {updated_count} models")
    print(f"Backup saved as: {backup_file}")
    print("\nNew training parameters will be used for future training runs.")


if __name__ == "__main__":
    print("=" * 80)
    print("FIX TRAINING PARAMETERS")
    print("=" * 80)
    print("\nThis script updates training parameters in model_mapping.json")
    print("to use correct step counts for effective training.\n")

    fix_training_parameters()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Retrain models that had low perplexity (40+)")
    print("   python combined_model_trainer.py --model <model_key> --force")
    print("\n2. Or use interactive trainer:")
    print("   python interactive_trainer.py")
    print()
