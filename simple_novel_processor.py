#!/usr/bin/env python3
"""
Simple Novel Processor
Robust processing for novels from novels__uncleaned folder
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

class SimpleNovelProcessor:
    """Simple, robust novel processor"""

    def __init__(self, source_dir: str = "novels__uncleaned", target_dir: str = "novels"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)

        # Simple status tracking
        self.status_file = Path("simple_processing_status.json")
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load processing status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {"processed": [], "failed": []}

    def _save_status(self):
        """Save processing status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def clean_text_simple(self, text: str) -> str:
        """Simple, robust text cleaning"""
        # Remove BOM if present
        if text.startswith('\ufeff'):
            text = text[1:]

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Clean up excessive whitespace (without regex)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line
                cleaned_lines.append('')

        text = '\n'.join(cleaned_lines)

        # Simple character filtering - keep only printable ASCII + common punctuation
        clean_chars = []
        for char in text:
            if 32 <= ord(char) <= 126 or char in '\n\t':
                clean_chars.append(char)
            else:
                clean_chars.append(' ')  # Replace with space

        return ''.join(clean_chars).strip()

    def analyze_text(self, text: str) -> Dict:
        """Simple text analysis"""
        words = text.split()
        sentences = []

        # Simple sentence splitting
        for delimiter in ['. ', '! ', '? ']:
            text = text.replace(delimiter, delimiter + '|SENT|')

        sentences = [s.strip() for s in text.split('|SENT|') if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "dialogue_ratio": text.count('"') / len(text) * 100 if text else 0
        }

    def make_safe_filename(self, original: str) -> str:
        """Create safe filename"""
        name = Path(original).stem
        # Keep only alphanumeric, spaces, and hyphens
        safe_chars = []
        for char in name:
            if char.isalnum() or char in ' -_':
                safe_chars.append(char)
            else:
                safe_chars.append('_')

        safe_name = ''.join(safe_chars)
        # Replace spaces with underscores and clean up
        safe_name = safe_name.replace(' ', '_')
        safe_name = '_'.join(part for part in safe_name.split('_') if part)

        return f"{safe_name.lower()}.txt"

    def process_novel(self, source_file: Path) -> bool:
        """Process a single novel with robust error handling"""
        filename = source_file.name

        if filename in self.status['processed']:
            print(f"SKIP: {filename} already processed")
            return True

        if filename in self.status['failed']:
            print(f"SKIP: {filename} previously failed")
            return False

        print(f"Processing: {filename}")

        try:
            # Read file with multiple encoding attempts
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(source_file, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    break
                except:
                    continue

            if not content:
                print(f"  ERROR: Could not read file")
                self.status['failed'].append(filename)
                return False

            if len(content) < 1000:
                print(f"  ERROR: File too short ({len(content)} chars)")
                self.status['failed'].append(filename)
                return False

            # Clean text
            cleaned_text = self.clean_text_simple(content)

            if len(cleaned_text) < 500:
                print(f"  ERROR: Cleaned text too short ({len(cleaned_text)} chars)")
                self.status['failed'].append(filename)
                return False

            # Analyze
            analysis = self.analyze_text(cleaned_text)

            if analysis['word_count'] < 1000:
                print(f"  ERROR: Too few words ({analysis['word_count']})")
                self.status['failed'].append(filename)
                return False

            # Create individual directory for this novel
            safe_filename = self.make_safe_filename(filename)
            novel_name = Path(safe_filename).stem
            novel_dir = self.target_dir / novel_name
            novel_dir.mkdir(exist_ok=True)

            # Save cleaned version in its own directory
            target_file = novel_dir / safe_filename

            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # Save analysis in the same directory
            analysis_file = novel_dir / "analysis.json"
            analysis['original_filename'] = filename
            analysis['cleaned_filename'] = safe_filename
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)

            # Create a readme file for the novel
            readme_file = novel_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(f"# {filename}\n\n")
                f.write(f"**Word Count:** {analysis['word_count']:,}\n")
                f.write(f"**Characters:** {analysis['character_count']:,}\n")
                f.write(f"**Sentences:** {analysis['sentence_count']:,}\n")
                f.write(f"**Avg Sentence Length:** {analysis['avg_sentence_length']:.1f} words\n")
                f.write(f"**Dialogue Ratio:** {analysis['dialogue_ratio']:.1f}%\n\n")
                f.write(f"## Files\n")
                f.write(f"- `{safe_filename}` - Cleaned novel text\n")
                f.write(f"- `analysis.json` - Text analysis data\n")
                f.write(f"- `README.md` - This file\n")

            print(f"  SUCCESS: {analysis['word_count']:,} words -> {safe_filename}")
            self.status['processed'].append(filename)
            return True

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}")
            self.status['failed'].append(filename)
            return False

    def process_batch(self, batch_size: int = 5) -> Dict[str, int]:
        """Process novels in batches"""
        if not self.source_dir.exists():
            print(f"ERROR: Source directory {self.source_dir} not found")
            return {"processed": 0, "failed": 0, "remaining": 0}

        # Get all txt files
        all_files = list(self.source_dir.glob("*.txt"))
        remaining_files = [f for f in all_files if f.name not in self.status['processed']]

        if not remaining_files:
            print("All files already processed!")
            return {"processed": 0, "failed": 0, "remaining": 0}

        print(f"Found {len(all_files)} total files")
        print(f"Processing {min(batch_size, len(remaining_files))} files...")

        success_count = 0
        failed_count = 0

        for i, file_path in enumerate(remaining_files[:batch_size]):
            print(f"\n[{i+1}/{min(batch_size, len(remaining_files))}]")

            if self.process_novel(file_path):
                success_count += 1
            else:
                failed_count += 1

            # Save status after each file
            self._save_status()

        remaining = len(remaining_files) - batch_size
        print(f"\nBatch Results:")
        print(f"  Processed: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Remaining: {max(0, remaining)}")

        return {
            "processed": success_count,
            "failed": failed_count,
            "remaining": max(0, remaining)
        }

    def list_processed(self):
        """List processed novels"""
        novel_dirs = [d for d in self.target_dir.iterdir() if d.is_dir()]

        if not novel_dirs:
            print("No novels processed yet.")
            return

        print(f"\nProcessed Novels ({len(novel_dirs)}):")
        print("=" * 70)

        for novel_dir in sorted(novel_dirs):
            analysis_file = novel_dir / "analysis.json"
            if analysis_file.exists():
                try:
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)

                    novel_name = novel_dir.name
                    word_count = analysis.get('word_count', 0)
                    sentence_count = analysis.get('sentence_count', 0)
                    print(f"{novel_name:<35} {word_count:>6,} words, {sentence_count:>4,} sentences")

                except:
                    print(f"{novel_dir.name:<35} ERROR reading analysis")
            else:
                print(f"{novel_dir.name:<35} Missing analysis file")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Novel Processor")
    parser.add_argument("--batch", type=int, default=5, help="Batch size")
    parser.add_argument("--list", action="store_true", help="List processed novels")

    args = parser.parse_args()

    processor = SimpleNovelProcessor()

    if args.list:
        processor.list_processed()
    else:
        results = processor.process_batch(args.batch)

        if results['processed'] > 0:
            print(f"\nSUCCESS: {results['processed']} novels ready for training!")
            print("Run: python single_novel_trainer.py")

if __name__ == "__main__":
    main()