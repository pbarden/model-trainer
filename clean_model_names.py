#!/usr/bin/env python3
"""
Model Name Cleanup Script - Model Tea
Clean up model names by removing version prefixes and keeping only the fun dessert names.

Copyright ChaiQ LLC
"""

import json
import re
from pathlib import Path

def clean_model_names():
    """Clean up model names in the mapping file"""
    mapping_file = Path("model_mapping.json")

    # Load current mapping
    with open(mapping_file, 'r') as f:
        data = json.load(f)

    print("Model Tea - Model Name Cleanup")
    print("=" * 40)
    print("Cleaning up model names to remove version prefixes...")

    changes_made = []

    # Clean up model descriptions
    for model_id, model_data in data["models"].items():
        old_description = model_data["description"]

        # Extract just the dessert name (everything after the last space)
        # Pattern: "0.17-bc Sprinkles" -> "Sprinkles"
        match = re.search(r'^[\d\.]+-[a-z]+ (.+)$', old_description)
        if match:
            new_description = match.group(1)
            model_data["description"] = new_description
            changes_made.append(f"  {model_id}: '{old_description}' -> '{new_description}'")
        else:
            print(f"  Warning: Could not parse '{old_description}' for {model_id}")

    # Update metadata
    data["metadata"]["description"] = "Model Tea - Clean model categorization with dessert-themed names"
    data["metadata"]["created_by"] = "Model Tea Cleanup Script"
    data["metadata"]["copyright"] = "ChaiQ LLC"

    # Save cleaned mapping
    with open(mapping_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nCompleted! Made {len(changes_made)} changes:")
    for change in changes_made:
        print(change)

    print(f"\nUpdated {mapping_file} with clean model names.")
    print("Models are now ready for Model Tea branding!")

if __name__ == "__main__":
    clean_model_names()