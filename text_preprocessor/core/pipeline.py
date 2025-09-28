#!/usr/bin/env python3

import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from .preprocessor import TextPreprocessor
from ..config.settings import PreprocessorConfig, ProcessingStats


class PreprocessingPipeline:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        self.stats = ProcessingStats()

    def process_single_novel(self, novel_path: Path) -> Dict:
        try:
            print(f"Processing: {novel_path.name}")

            txt_files = list(novel_path.glob("*.txt"))
            if not txt_files:
                print(f"  No .txt file found in {novel_path.name}")
                self.stats.files_skipped += 1
                return {"status": "skipped", "reason": "No txt file"}

            main_txt = txt_files[0]
            backup_dir = novel_path / self.config.backup_folder

            if backup_dir.exists():
                print(f"  Already processed (backup exists)")
                self.stats.files_skipped += 1
                return {"status": "skipped", "reason": "Already processed"}

            processed_text, processing_stats = self.preprocessor.preprocess_novel(main_txt)

            backup_path = self.preprocessor.backup_and_replace_file(main_txt, processed_text)

            self.stats.files_processed += 1
            self.stats.total_tokens_removed += processing_stats['tokens_removed']
            self.stats.total_tokens_preserved += processing_stats['processed_tokens']

            print(f"  Processed: {processing_stats['tokens_removed']} tokens removed, "
                  f"compression: {processing_stats['compression_ratio']:.2f}")

            return {
                "status": "success",
                "backup_path": str(backup_path),
                "stats": processing_stats
            }

        except Exception as e:
            print(f"  Error processing {novel_path.name}: {e}")
            self.stats.files_failed += 1
            return {"status": "error", "error": str(e)}

    def process_all_novels(self, max_novels: Optional[int] = None) -> Dict:
        start_time = time.time()

        novels_dir = Path(self.config.novels_dir)
        if not novels_dir.exists():
            raise FileNotFoundError(f"Novels directory not found: {novels_dir}")

        novel_dirs = [d for d in novels_dir.iterdir() if d.is_dir()]

        if max_novels:
            novel_dirs = novel_dirs[:max_novels]

        print(f"Found {len(novel_dirs)} novels to process")
        print("=" * 60)

        results = {}

        if self.config.batch_size > 1:
            with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
                futures = {executor.submit(self.process_single_novel, novel_dir): novel_dir
                          for novel_dir in novel_dirs}

                for future in futures:
                    novel_dir = futures[future]
                    results[novel_dir.name] = future.result()
        else:
            for novel_dir in novel_dirs:
                results[novel_dir.name] = self.process_single_novel(novel_dir)

        self.stats.processing_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print(f"Processed: {self.stats.files_processed} novels")
        print(f"Skipped: {self.stats.files_skipped} novels")
        print(f"Failed: {self.stats.files_failed} novels")
        print(f"Total tokens removed: {self.stats.total_tokens_removed:,}")
        print(f"Total tokens preserved: {self.stats.total_tokens_preserved:,}")
        print(f"Processing time: {self.stats.processing_time/60:.1f} minutes")

        return {
            "stats": self.stats,
            "results": results
        }

    def process_specific_novels(self, novel_names: List[str]) -> Dict:
        start_time = time.time()

        novels_dir = Path(self.config.novels_dir)
        results = {}

        print(f"Processing {len(novel_names)} specific novels")
        print("=" * 60)

        for novel_name in novel_names:
            novel_path = novels_dir / novel_name
            if novel_path.exists():
                results[novel_name] = self.process_single_novel(novel_path)
            else:
                print(f"Novel not found: {novel_name}")
                results[novel_name] = {"status": "error", "error": "Novel directory not found"}
                self.stats.files_failed += 1

        self.stats.processing_time = time.time() - start_time

        print(f"\nCompleted in {self.stats.processing_time:.1f} seconds")

        return {
            "stats": self.stats,
            "results": results
        }

    def restore_novel_from_backup(self, novel_name: str) -> bool:
        novel_dir = Path(self.config.novels_dir) / novel_name
        backup_dir = novel_dir / self.config.backup_folder

        if not backup_dir.exists():
            print(f"No backup found for {novel_name}")
            return False

        txt_files = list(novel_dir.glob("*.txt"))
        backup_files = list(backup_dir.glob("*.txt"))

        if not backup_files:
            print(f"No backup txt file found for {novel_name}")
            return False

        if txt_files:
            txt_files[0].unlink()

        backup_file = backup_files[0]
        restored_file = novel_dir / backup_file.name

        import shutil
        shutil.copy2(backup_file, restored_file)

        print(f"Restored {novel_name} from backup")
        return True

    def get_processing_summary(self) -> Dict:
        return {
            "files_processed": self.stats.files_processed,
            "files_skipped": self.stats.files_skipped,
            "files_failed": self.stats.files_failed,
            "total_tokens_removed": self.stats.total_tokens_removed,
            "total_tokens_preserved": self.stats.total_tokens_preserved,
            "compression_ratio": (self.stats.total_tokens_preserved /
                                (self.stats.total_tokens_preserved + self.stats.total_tokens_removed))
                               if self.stats.total_tokens_preserved > 0 else 0,
            "processing_time_minutes": self.stats.processing_time / 60,
            "novels_per_minute": self.stats.files_processed / (self.stats.processing_time / 60)
                               if self.stats.processing_time > 0 else 0
        }