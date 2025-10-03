#!/usr/bin/env python3
"""
Fix adaptive training by removing hardcoded max_iterations from model_mapping.json
This allows the adaptive training algorithm to determine iterations based on quality metrics.
"""

import json
from pathlib import Path

def fix_model_mapping():
    """Remove max_iterations from training_parameters to enable true adaptive training"""

    mapping_file = Path("model_mapping.json")

    if not mapping_file.exists():
        print("Error: model_mapping.json not found")
        return

    # Load current mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = data.get('models', {})
    modified_count = 0

    print("Removing hardcoded max_iterations to enable adaptive training...")
    print()

    for model_key, model_data in models.items():
        if 'training_parameters' in model_data:
            params = model_data['training_parameters']

            if 'max_iterations' in params:
                old_value = params['max_iterations']
                del params['max_iterations']
                modified_count += 1
                print(f"[OK] {model_key}: removed max_iterations={old_value}")

    # Save updated mapping
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print()
    print(f"Modified {modified_count} models")
    print(f"Adaptive training will now use:")
    print(f"  - min_iterations: 5")
    print(f"  - max_iterations: 15 (upper bound)")
    print(f"  - Actual iterations determined by quality metrics")
    print()
    print("[OK] model_mapping.json updated successfully")

if __name__ == "__main__":
    fix_model_mapping()
