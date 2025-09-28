#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

from text_preprocessor import PreprocessingPipeline, PreprocessorConfig


def main():
    parser = argparse.ArgumentParser(description="Preprocess novel text files for training")
    parser.add_argument("command", choices=["all", "specific", "restore", "status"],
                       help="Processing command")
    parser.add_argument("novels", nargs="*", help="Novel names for specific processing")
    parser.add_argument("--max", type=int, help="Maximum number of novels to process")
    parser.add_argument("--batch-size", type=int, default=1, help="Parallel processing batch size")
    parser.add_argument("--no-hf", action="store_true", help="Disable Hugging Face model usage")
    parser.add_argument("--preserve-scripts", action="store_true",
                       help="Don't convert play scripts to prose")

    args = parser.parse_args()

    config = PreprocessorConfig(
        batch_size=args.batch_size,
        use_huggingface_models=not args.no_hf,
        convert_scripts_to_prose=not args.preserve_scripts
    )

    pipeline = PreprocessingPipeline(config)

    if args.command == "all":
        results = pipeline.process_all_novels(max_novels=args.max)

    elif args.command == "specific":
        if not args.novels:
            print("Error: Please specify novel names for specific processing")
            return 1
        results = pipeline.process_specific_novels(args.novels)

    elif args.command == "restore":
        if not args.novels:
            print("Error: Please specify novel names to restore")
            return 1
        for novel_name in args.novels:
            success = pipeline.restore_novel_from_backup(novel_name)
            if not success:
                print(f"Failed to restore {novel_name}")
        return 0

    elif args.command == "status":
        summary = pipeline.get_processing_summary()
        print("Processing Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0

    return 0


if __name__ == "__main__":
    exit(main())