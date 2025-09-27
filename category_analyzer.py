#!/usr/bin/env python3
"""
Model Category Analysis System

Analyzes novel categorization structure and generates comprehensive mapping
for multi-model training pipeline. Provides statistical analysis and validation
of category assignments.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_data_structure():
    """Analyze all data folders and create comprehensive mapping"""
    data_dir = Path("data")

    # Skip the processed folder
    model_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "processed"]

    model_mapping = {}

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name

        # Read existing model_info.json if it exists
        model_info_file = model_dir / "model_info.json"
        model_info = {}

        if model_info_file.exists():
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)

        # Get all .txt files in the directory
        txt_files = [f.name for f in model_dir.glob("*.txt")]

        # Remove .txt extension for cleaner names
        novel_names = [f.replace('.txt', '') for f in txt_files]

        model_mapping[model_name] = {
            "description": model_info.get("description", f"Model {model_name}"),
            "novel_count": len(novel_names),
            "novels": sorted(novel_names),
            "txt_files": sorted(txt_files)
        }

        print(f"{model_name}: {len(novel_names)} novels")
        if len(novel_names) <= 10:
            print(f"  Novels: {', '.join(novel_names[:5])}")
        else:
            print(f"  Sample: {', '.join(novel_names[:3])}... (+{len(novel_names)-3} more)")

    return model_mapping

def create_mapping_file(model_mapping):
    """Create the definitive model categorization file"""

    # Calculate statistics
    total_novels = sum(data["novel_count"] for data in model_mapping.values())

    mapping_data = {
        "metadata": {
            "total_models": len(model_mapping),
            "total_novels_assigned": total_novels,
            "created_by": "category_analyzer.py",
            "description": "Model categorization mapping for multi-model training approach"
        },
        "models": model_mapping
    }

    # Save mapping file
    with open("model_categorization_mapping.json", 'w') as f:
        json.dump(mapping_data, f, indent=2)

    print(f"\nSummary:")
    print(f"  Total models: {len(model_mapping)}")
    print(f"  Total novels assigned: {total_novels}")
    print(f"  Mapping saved to: model_categorization_mapping.json")

    return mapping_data

def verify_novels_exist(model_mapping):
    """Verify that all referenced novels exist in our cleaned novels directory"""
    novels_dir = Path("novels")
    missing_novels = []
    found_novels = []

    # Get all available cleaned novels
    available_novels = set()
    for novel_dir in novels_dir.iterdir():
        if novel_dir.is_dir():
            available_novels.add(novel_dir.name)

    print(f"\nVerification against cleaned novels directory:")
    print(f"Available cleaned novels: {len(available_novels)}")

    for model_name, data in model_mapping.items():
        model_missing = []
        model_found = []

        for novel in data["novels"]:
            # Convert novel name to directory name format
            novel_dir_name = novel.lower().replace(' ', '_').replace("'", "").replace('"', '').replace(',', '').replace(':', '').replace('(', '').replace(')', '').replace('!', '').replace('?', '').replace('.', '').replace('-', '_').replace('&', 'and')

            # Try exact match first
            if novel_dir_name in available_novels:
                model_found.append(novel)
            else:
                # Try some common variations
                variations = [
                    novel_dir_name.replace('__', '_'),
                    novel_dir_name.replace('_', ''),
                    novel_dir_name.replace(' ', ''),
                    novel.lower().replace(' ', '_')
                ]

                found_variation = False
                for var in variations:
                    if var in available_novels:
                        model_found.append(novel)
                        found_variation = True
                        break

                if not found_variation:
                    model_missing.append(novel)

        if model_missing:
            missing_novels.extend(model_missing)
            print(f"  {model_name}: {len(model_found)} found, {len(model_missing)} missing")
            if len(model_missing) <= 5:
                print(f"    Missing: {model_missing}")
        else:
            print(f"  {model_name}: All {len(model_found)} novels found [OK]")

        found_novels.extend(model_found)

    print(f"\nOverall verification:")
    print(f"  Novels found: {len(found_novels)}")
    print(f"  Novels missing: {len(missing_novels)}")

    if missing_novels:
        print(f"  Missing novels: {missing_novels[:10]}...")

    return len(missing_novels) == 0

if __name__ == "__main__":
    print("Analyzing data folder structure...")
    print("=" * 50)

    model_mapping = analyze_data_structure()

    print("\n" + "=" * 50)
    mapping_data = create_mapping_file(model_mapping)

    print("\n" + "=" * 50)
    all_found = verify_novels_exist(model_mapping)

    if all_found:
        print("\n[SUCCESS] All novels successfully mapped and verified!")
    else:
        print("\n[WARNING] Some novels could not be matched. Check the verification output above.")