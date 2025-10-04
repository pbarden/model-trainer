#!/usr/bin/env python3
"""
Clean model_mapping.json by removing application logic (training parameters)
and keeping only relational data (which novels belong to which models).
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def clean_model_entry(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove training parameters and keep only relational data"""
    cleaned = {}

    # Keep description and metadata
    if "description" in model_data:
        cleaned["description"] = model_data["description"]

    # Keep minimal organizational metadata (optional)
    if "variation" in model_data:
        cleaned["variation"] = model_data["variation"]

    # Keep the novels list (required)
    if "novels" in model_data:
        cleaned["novels"] = []
        for novel in model_data["novels"]:
            cleaned_novel = {
                "original_name": novel["original_name"],
                "directory_name": novel["directory_name"],
                "word_count": novel["word_count"]
            }
            # Optional: keep tier for organization
            if "tier" in novel:
                cleaned_novel["tier"] = novel["tier"]
            cleaned["novels"].append(cleaned_novel)

    # Remove these fields (they're computed by the app):
    # - training_parameters (entire block)
    # - max_steps_per_iteration
    # - learning_rate_start/end
    # - size_category
    # - novel_count (redundant, calculated from len(novels))
    # - total_word_count (redundant, calculated from sum of novel word_counts)
    # - tier_distribution (optional, but not used by training)

    return cleaned


def clean_mapping_file(input_file: Path, output_file: Path = None, dry_run: bool = False):
    """Clean the model mapping file"""
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keep metadata
    cleaned_data = {}
    if "metadata" in data:
        cleaned_data["metadata"] = data["metadata"]

    # Clean each model
    cleaned_data["models"] = {}
    removed_fields_count = 0

    for model_key, model_data in data.get("models", {}).items():
        original_keys = set(model_data.keys())
        cleaned_model = clean_model_entry(model_data)
        cleaned_keys = set(cleaned_model.keys())

        removed_keys = original_keys - cleaned_keys
        if removed_keys:
            removed_fields_count += len(removed_keys)
            print(f"  {model_key}: Removed {removed_keys}")

        cleaned_data["models"][model_key] = cleaned_model

    # Report
    print(f"\nCleaned {len(cleaned_data['models'])} models")
    print(f"Removed {removed_fields_count} erroneous fields")

    if dry_run:
        print("\nDry run - no changes made")
        print("\nExample cleaned entry:")
        first_key = next(iter(cleaned_data["models"]))
        print(json.dumps({first_key: cleaned_data["models"][first_key]}, indent=2))
        return

    # Write output
    if output_file is None:
        output_file = input_file

    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print("Done!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean model_mapping.json")
    parser.add_argument("--input", "-i", type=Path, default=Path("model_mapping.json"),
                       help="Input mapping file (default: model_mapping.json)")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be removed without making changes")

    args = parser.parse_args()
    clean_mapping_file(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
