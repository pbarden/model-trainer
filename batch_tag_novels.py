#!/usr/bin/env python3

import sys
from corpus_tagger import tag_all_novels, SimpleNovelTagger
from pathlib import Path
import json
import time

def batch_tag_novels(max_novels=None, start_from=0):

    tagger = SimpleNovelTagger()
    novels_dir = Path("novels")

    if not novels_dir.exists():
        print("Error: novels directory not found")
        return

    novel_dirs = [d for d in novels_dir.iterdir() if d.is_dir()]

    if max_novels:
        novel_dirs = novel_dirs[start_from:start_from + max_novels]
    else:
        novel_dirs = novel_dirs[start_from:]

    total = len(novel_dirs)
    processed = 0
    start_time = time.time()

    print(f"Batch tagging {total} novels starting from index {start_from}")
    print("=" * 60)

    for i, novel_dir in enumerate(novel_dirs):
        print(f"\n[{i+1}/{total}] Processing: {novel_dir.name}")

        # Check if already tagged
        tag_file = novel_dir / "tags.json"
        if tag_file.exists():
            print(f"  [SKIP] Already tagged")
            continue

        # Find text file
        txt_files = list(novel_dir.glob("*.txt"))
        if not txt_files:
            print(f"  [ERROR] No .txt file found")
            continue

        try:
            # Tag the novel
            tags = tagger.tag_novel(txt_files[0])

            # Create tag data
            tag_data = {
                'title': tags.title,
                'basic_stats': tags.basic_stats,
                'themes': tags.themes,
                'genre_hints': tags.genre_hints,
                'characters': tags.characters,
                'locations': tags.locations,
                'sentiment_profile': tags.sentiment_profile,
                'writing_style': tags.writing_style,
                'memory_compatibility': tags.memory_compatibility,
                'scene_types': tags.scene_types,
                'behaviors': tags.behaviors,
                'narrative_elements': tags.narrative_elements,
                'social_dynamics': tags.social_dynamics,
                'tagged_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_file': txt_files[0].name
            }

            # Save tags
            with open(tag_file, 'w', encoding='utf-8') as f:
                json.dump(tag_data, f, indent=2)

            processed += 1

            # Progress info
            elapsed = time.time() - start_time
            rate = processed / elapsed * 60  # novels per minute

            print(f"  [OK] Tagged: {len(tags.themes)} themes, {len(tags.characters)} chars, {len(tags.scene_types)} scenes, {len(tags.behaviors)} behaviors")
            print(f"  Rate: {rate:.1f} novels/min, Elapsed: {elapsed/60:.1f}min")

        except Exception as e:
            print(f"  [ERROR] {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"BATCH TAGGING COMPLETE")
    print(f"Processed: {processed}/{total} novels")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {processed/(total_time/60):.1f} novels/minute")

def tag_specific_novels(novel_names):
    tagger = SimpleNovelTagger()

    for novel_name in novel_names:
        novel_dir = Path("novels") / novel_name

        if not novel_dir.exists():
            print(f"Novel directory not found: {novel_name}")
            continue

        print(f"Tagging: {novel_name}")

        txt_files = list(novel_dir.glob("*.txt"))
        if not txt_files:
            print(f"  No .txt file found in {novel_name}")
            continue

        try:
            tags = tagger.tag_novel(txt_files[0])

            tag_data = {
                'title': tags.title,
                'basic_stats': tags.basic_stats,
                'themes': tags.themes,
                'genre_hints': tags.genre_hints,
                'characters': tags.characters,
                'locations': tags.locations,
                'sentiment_profile': tags.sentiment_profile,
                'writing_style': tags.writing_style,
                'memory_compatibility': tags.memory_compatibility,
                'scene_types': tags.scene_types,
                'behaviors': tags.behaviors,
                'narrative_elements': tags.narrative_elements,
                'social_dynamics': tags.social_dynamics,
                'tagged_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source_file': txt_files[0].name
            }

            tag_file = novel_dir / "tags.json"
            with open(tag_file, 'w', encoding='utf-8') as f:
                json.dump(tag_data, f, indent=2)

            print(f"  [OK] Saved to {tag_file}")
            print(f"  Themes: {', '.join(tags.themes)}")

        except Exception as e:
            print(f"  [ERROR] {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python batch_tag_novels.py all                    # Tag all novels")
        print("  python batch_tag_novels.py 50                     # Tag first 50 novels")
        print("  python batch_tag_novels.py 25 100                 # Tag 25 novels starting from index 100")
        print("  python batch_tag_novels.py specific novel1 novel2 # Tag specific novels")

    elif sys.argv[1] == "all":
        batch_tag_novels()

    elif sys.argv[1] == "specific":
        if len(sys.argv) < 3:
            print("Error: Please specify novel names after 'specific'")
        else:
            tag_specific_novels(sys.argv[2:])

    elif sys.argv[1].isdigit():
        max_novels = int(sys.argv[1])
        start_from = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
        batch_tag_novels(max_novels, start_from)

    else:
        print(f"Invalid argument: {sys.argv[1]}")
        print("Use 'all', a number, or 'specific' followed by novel names")