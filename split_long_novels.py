#!/usr/bin/env python3
"""
Split very long novels into multiple parts based on line count.
Handles novels that are already split (e.g., part1, part2) by renumbering.
Creates separate directories for each part.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re


def load_tier_file(tier_name: str) -> List[Dict[str, Any]]:
    """Load novels from tier JSON file"""
    filename = f"tier_{tier_name}.json"

    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        print("Run analyze_novel_lengths.py first to generate tier files.")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_novel_into_parts(file_path: Path, num_parts: int) -> List[str]:
    """
    Split a novel into N parts by dividing line count.
    Returns list of content strings for each part.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        lines_per_part = total_lines // num_parts

        parts = []

        for i in range(num_parts):
            start_idx = i * lines_per_part

            # Last part gets all remaining lines
            if i == num_parts - 1:
                end_idx = total_lines
            else:
                end_idx = (i + 1) * lines_per_part

            part_lines = lines[start_idx:end_idx]
            part_content = ''.join(part_lines)
            parts.append(part_content)

        return parts

    except Exception as e:
        print(f"Error splitting {file_path}: {e}")
        return []


def extract_part_number(directory_name: str) -> int:
    """
    Extract existing part number from directory name.
    Returns 0 if no part number found.

    Examples:
    - "novel_name_part1" → 1
    - "novel_name_part2" → 2
    - "novel_name" → 0
    """
    match = re.search(r'_part(\d+)$', directory_name)
    if match:
        return int(match.group(1))
    return 0


def get_base_name(directory_name: str) -> str:
    """
    Get base name without part suffix.

    Examples:
    - "novel_name_part1" → "novel_name"
    - "novel_name" → "novel_name"
    """
    return re.sub(r'_part\d+$', '', directory_name)


def calculate_new_part_numbers(existing_part: int, num_splits: int) -> List[int]:
    """
    Calculate new part numbers for split files.

    If existing_part = 0 (no existing part):
        Split into part1, part2, etc.

    If existing_part = N (already a part):
        Split into partN, partN+1, etc.

    Examples:
    - existing_part=0, num_splits=2 → [1, 2]
    - existing_part=1, num_splits=2 → [1, 2]
    - existing_part=2, num_splits=3 → [3, 4, 5]
    """
    if existing_part == 0:
        # First time splitting
        return list(range(1, num_splits + 1))
    else:
        # Already a part, need to renumber sequentially
        # part1 → part1, part2
        # part2 → part3, part4
        start_num = existing_part
        return list(range(start_num, start_num + num_splits))


def split_very_long_novels(max_words: int = 94000, novels_dir: str = "novels"):
    """
    Split novels over max_words into multiple parts.

    Args:
        max_words: Maximum words per novel (default: 94,000)
        novels_dir: Directory containing novels
    """

    print(f"Loading very long novels (>{max_words:,} words)...")

    # Load very long tier
    very_long_novels = load_tier_file("very_long")

    if not very_long_novels:
        print("No very long novels found!")
        return

    print(f"Found {len(very_long_novels)} novels to split\n")
    print("=" * 80)

    novels_path = Path(novels_dir)
    split_count = 0

    for novel_info in very_long_novels:
        directory_name = novel_info['directory_name']
        word_count = novel_info['words']
        file_name = novel_info['file_name']

        novel_dir = novels_path / directory_name
        txt_file = novel_dir / file_name

        if not txt_file.exists():
            print(f"Warning: {txt_file} not found, skipping...")
            continue

        # Determine how many parts needed
        if word_count < 140000:
            num_parts = 2
        elif word_count < 190000:
            num_parts = 3
        else:
            num_parts = 4

        print(f"Splitting: {directory_name}")
        print(f"  Words: {word_count:,}")
        print(f"  Parts: {num_parts}")

        # Check if already a part
        existing_part = extract_part_number(directory_name)
        base_name = get_base_name(directory_name)

        # Calculate new part numbers
        new_part_numbers = calculate_new_part_numbers(existing_part, num_parts)

        print(f"  Base name: {base_name}")
        if existing_part > 0:
            print(f"  Existing part: {existing_part}")
        print(f"  New parts: {new_part_numbers}")

        # Split the file
        parts = split_novel_into_parts(txt_file, num_parts)

        if len(parts) != num_parts:
            print(f"  ERROR: Failed to split into {num_parts} parts")
            continue

        # Create new directories and save parts
        for part_num, part_content in zip(new_part_numbers, parts):
            part_dir_name = f"{base_name}_part{part_num}"
            part_dir = novels_path / part_dir_name

            # Create directory
            part_dir.mkdir(exist_ok=True)

            # Save part file
            part_file = part_dir / f"{part_dir_name}.txt"

            with open(part_file, 'w', encoding='utf-8') as f:
                f.write(part_content)

            part_words = len(part_content.split())
            print(f"  Created: {part_dir_name} ({part_words:,} words)")

        # Delete original file and directory
        try:
            txt_file.unlink()
            # Only delete directory if it's empty or only contains the deleted file
            remaining_files = list(novel_dir.glob("*"))
            if len(remaining_files) == 0:
                novel_dir.rmdir()
                print(f"  Removed: {directory_name}/")
        except Exception as e:
            print(f"  Warning: Could not remove original: {e}")

        split_count += 1
        print()

    print("=" * 80)
    print(f"Split {split_count} novels into parts")
    print("\nNext steps:")
    print("1. Run analyze_novel_lengths.py to verify all novels are under 94K words")
    print("2. Check the tier_very_long.json to ensure it's empty or has acceptable lengths")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split very long novels into multiple parts"
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=94000,
        help='Maximum words per novel (default: 94000)'
    )
    parser.add_argument(
        '--novels-dir',
        type=str,
        default='novels',
        help='Directory containing novels (default: novels)'
    )

    args = parser.parse_args()

    split_very_long_novels(max_words=args.max_words, novels_dir=args.novels_dir)


if __name__ == "__main__":
    main()
