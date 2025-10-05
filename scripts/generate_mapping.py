#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Dict, List


def scan_novels(novels_dir: Path) -> Dict:
    novels = {}

    for novel_dir in sorted(novels_dir.iterdir()):
        if not novel_dir.is_dir():
            continue

        txt_files = list(novel_dir.glob("*.txt"))
        if not txt_files:
            continue

        try:
            content = txt_files[0].read_text(encoding='utf-8')
            word_count = len(content.split())

            novel_key = novel_dir.name
            novels[novel_key] = {
                "original_name": novel_dir.name.replace('_', ' ').title(),
                "directory_name": novel_dir.name,
                "word_count": word_count
            }
        except Exception as e:
            print(f"Warning: Failed to process {novel_dir.name}: {e}", file=sys.stderr)

    return novels


def generate_single_novel_models(novels: Dict) -> Dict:
    models = {}

    for novel_key in novels.keys():
        models[novel_key] = {
            "novels": [novel_key]
        }

    return models


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate novels.json and models.json from novels directory")
    parser.add_argument("--novels-dir", type=Path, default=Path("novels"),
                       help="Novels directory (default: novels)")
    parser.add_argument("--novels-output", type=Path, default=Path("novels.json"),
                       help="Novels output file (default: novels.json)")
    parser.add_argument("--models-output", type=Path, default=Path("models.json"),
                       help="Models output file (default: models.json)")
    parser.add_argument("--mode", choices=["single"], default="single",
                       help="Generation mode: single = one model per novel")

    args = parser.parse_args()

    if not args.novels_dir.exists():
        print(f"Error: {args.novels_dir} not found")
        sys.exit(1)

    print(f"Scanning {args.novels_dir}...")
    novels = scan_novels(args.novels_dir)
    print(f"Found {len(novels)} novels")

    print(f"Generating models in '{args.mode}' mode...")
    if args.mode == "single":
        models = generate_single_novel_models(novels)

    print(f"Writing to {args.novels_output}...")
    with open(args.novels_output, 'w', encoding='utf-8') as f:
        json.dump(novels, f, indent=2, ensure_ascii=False)

    print(f"Writing to {args.models_output}...")
    with open(args.models_output, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)

    print(f"Done! Created {len(novels)} novels and {len(models)} models")


if __name__ == "__main__":
    main()
