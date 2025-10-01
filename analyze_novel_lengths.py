#!/usr/bin/env python3
"""
Analyze novel lengths and categorize them into tiers for optimal training.
Outputs JSON files with novels grouped by word count ranges.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics


def count_words_in_file(file_path: Path) -> int:
    """Count words in a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return len(content.split())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def analyze_novels(novels_dir: str = "novels") -> List[Dict[str, Any]]:
    """Analyze all novels and return their metadata"""
    novels_path = Path(novels_dir)

    if not novels_path.exists():
        print(f"Error: Directory '{novels_dir}' not found!")
        return []

    novel_data = []

    for novel_dir in novels_path.iterdir():
        if not novel_dir.is_dir():
            continue

        # Look for text files in the directory
        txt_files = list(novel_dir.glob("*.txt"))

        if not txt_files:
            continue

        # Use first text file found
        txt_file = txt_files[0]
        word_count = count_words_in_file(txt_file)

        if word_count == 0:
            continue

        novel_data.append({
            "directory_name": novel_dir.name,
            "file_name": txt_file.name,
            "words": word_count,
            "file_path": str(txt_file.relative_to(novels_path.parent))
        })

    return novel_data


def categorize_into_tiers(novel_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize novels into fixed tiers based on word count.

    Tiers:
    - TINY: < 20,000 words
    - VERY SHORT: 20,000 - 43,999 words
    - SHORT: 44,000 - 66,999 words
    - LONG: 67,000 - 93,999 words
    - VERY LONG: 94,000+ words (needs splitting)
    """

    # Sort by word count
    novel_data.sort(key=lambda x: x['words'])

    # Split into fixed tiers
    tiny = []
    very_short = []
    short = []
    long_novels = []
    very_long = []

    for novel in novel_data:
        word_count = novel['words']

        if word_count < 20000:
            tiny.append(novel)
        elif word_count < 44000:
            very_short.append(novel)
        elif word_count < 67000:
            short.append(novel)
        elif word_count < 94000:
            long_novels.append(novel)
        else:
            very_long.append(novel)

    tiers = {
        'tiny': tiny,
        'very_short': very_short,
        'short': short,
        'long': long_novels,
        'very_long': very_long
    }

    return tiers


def print_tier_analysis(tiers: Dict[str, List[Dict[str, Any]]]):
    """Print detailed tier analysis"""
    print("\n" + "=" * 80)
    print("NOVEL LENGTH ANALYSIS")
    print("=" * 80)

    tier_info = [
        ('tiny', 'TINY', '< 20K words'),
        ('very_short', 'VERY SHORT', '20K - 44K words'),
        ('short', 'SHORT', '44K - 67K words'),
        ('long', 'LONG', '67K - 94K words'),
        ('very_long', 'VERY LONG', '94K+ words (NEEDS SPLITTING)')
    ]

    total_novels = sum(len(tiers[key]) for key, _, _ in tier_info)
    total_words = sum(novel['words'] for tier in tiers.values() for novel in tier)

    print(f"\nTotal Novels: {total_novels}")
    print(f"Total Words: {total_words:,}")
    print(f"Average: {total_words // total_novels:,} words per novel\n")

    for key, name, description in tier_info:
        tier_novels = tiers[key]
        count = len(tier_novels)

        if count == 0:
            print(f"{name} ({description}): 0 novels")
            continue

        words_in_tier = sum(n['words'] for n in tier_novels)
        avg_words = words_in_tier // count if count > 0 else 0
        min_words = min(n['words'] for n in tier_novels)
        max_words = max(n['words'] for n in tier_novels)

        print(f"{name} ({description}):")
        print(f"  Novels: {count}")
        print(f"  Total Words: {words_in_tier:,}")
        print(f"  Range: {min_words:,} - {max_words:,} words")
        print(f"  Average: {avg_words:,} words")
        print()


def save_tier_files(tiers: Dict[str, List[Dict[str, Any]]]):
    """Save each tier to a separate JSON file"""
    for tier_name, novels in tiers.items():
        filename = f"tier_{tier_name}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(novels, f, indent=2)

        print(f"Saved {len(novels)} novels to {filename}")


def show_very_long_novels(very_long: List[Dict[str, Any]]):
    """Show details of very long novels that need splitting"""
    if not very_long:
        print("\n" + "=" * 80)
        print("NO VERY LONG NOVELS - All novels are under 94K words!")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("VERY LONG NOVELS (NEED SPLITTING)")
    print("=" * 80)
    print(f"\nThese {len(very_long)} novels should be split into parts:\n")

    for novel in very_long:
        words = novel['words']

        # Determine how many parts needed
        if words < 140000:
            parts_needed = 2
            approx_part_size = words // 2
        elif words < 190000:
            parts_needed = 3
            approx_part_size = words // 3
        else:
            parts_needed = 4
            approx_part_size = words // 4

        print(f"  {novel['directory_name']}")
        print(f"    Words: {words:,}")
        print(f"    Recommended: Split into {parts_needed} parts (~{approx_part_size:,} words each)")
        print()


def calculate_training_recommendations(tiers: Dict[str, List[Dict[str, Any]]]):
    """Calculate training recommendations based on tier sizes"""
    print("\n" + "=" * 80)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 80)

    # Optimal training: 4 novels per combined model, 12-18 iterations
    recommendations = []

    for tier_name in ['tiny', 'very_short', 'short', 'long']:
        novels = tiers[tier_name]
        count = len(novels)

        if count == 0:
            continue

        # Calculate how many 4-novel models we can make
        complete_models = count // 4
        remainder = count % 4

        avg_words = sum(n['words'] for n in novels) // count if count > 0 else 0
        estimated_combined_words = avg_words * 4

        # Estimate iterations based on combined word count
        if estimated_combined_words < 80000:
            est_iterations = "8-12"
        elif estimated_combined_words < 160000:
            est_iterations = "12-15"
        else:
            est_iterations = "15-18"

        recommendations.append({
            'tier': tier_name.upper().replace('_', ' '),
            'novels': count,
            'complete_models': complete_models,
            'remainder': remainder,
            'avg_words': avg_words,
            'combined_words': estimated_combined_words,
            'iterations': est_iterations
        })

    print(f"\nCombining 4 novels per model:\n")

    total_complete_models = 0
    for rec in recommendations:
        print(f"{rec['tier']}:")
        print(f"  {rec['novels']} novels → {rec['complete_models']} complete models " +
              f"+ {rec['remainder']} remaining")
        print(f"  Avg per novel: {rec['avg_words']:,} words")
        print(f"  Combined model size: ~{rec['combined_words']:,} words")
        print(f"  Estimated iterations: {rec['iterations']}")
        print()
        total_complete_models += rec['complete_models']

    print(f"Total complete 4-novel models: {total_complete_models}")
    print()

    # Show very long tier separately
    if tiers['very_long']:
        print("VERY LONG tier: Must be split first before combining!")


def main():
    """Main analysis function"""
    print("Analyzing novels directory...")

    # Analyze all novels
    novel_data = analyze_novels("novels")

    if not novel_data:
        print("No novels found!")
        return

    print(f"Found {len(novel_data)} novels")

    # Categorize into tiers
    tiers = categorize_into_tiers(novel_data)

    # Print analysis
    print_tier_analysis(tiers)

    # Show very long novels
    show_very_long_novels(tiers['very_long'])

    # Calculate training recommendations
    calculate_training_recommendations(tiers)

    # Save tier files
    print("\n" + "=" * 80)
    print("Saving tier files...")
    print("=" * 80 + "\n")
    save_tier_files(tiers)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if tiers['very_long']:
        print(f"\n1. Run split script on {len(tiers['very_long'])} very long novels")
        print("2. Re-run this analysis to verify all novels are under 94K words")
        print("3. Create combined models using 4 novels per model")
    else:
        print("\n1. All novels are under 94K words - ready for combining!")
        print("2. Create combined models using 4 novels per model")
        print("3. Use tier_*.json files to group novels by similar length")

    print()


if __name__ == "__main__":
    main()
