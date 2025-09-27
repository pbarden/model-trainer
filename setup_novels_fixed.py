#!/usr/bin/env python3
"""
Setup Novels (Fixed)
Interactive setup using the working simple processor
"""

import sys
from pathlib import Path
from simple_novel_processor import SimpleNovelProcessor

def main():
    """Interactive setup for novels"""
    print("=" * 60)
    print("Novel Processing Setup (Fixed Version)")
    print("=" * 60)

    # Check if source directory exists
    source_dir = Path("novels__uncleaned")
    if not source_dir.exists():
        print(f"ERROR: {source_dir} folder not found!")
        print("Please ensure the novels__uncleaned folder exists with .txt files.")
        return

    processor = SimpleNovelProcessor()

    # Show current status
    all_files = list(source_dir.glob("*.txt"))
    processed_count = len(processor.status['processed'])
    failed_count = len(processor.status['failed'])
    remaining = len(all_files) - processed_count - failed_count

    print(f"Source folder: {source_dir}")
    print(f"Total novels: {len(all_files)}")
    print(f"Processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Remaining: {remaining}")

    if remaining == 0:
        print("\nAll novels already processed!")
        processor.list_processed()
        print(f"\nReady for training! Run: python single_novel_trainer.py")
        return

    print(f"\nProcessing Options:")
    print("1. Quick start (5 novels)")
    print("2. Small batch (10 novels)")
    print("3. Medium batch (25 novels)")
    print("4. Large batch (50 novels)")
    print("5. Process all remaining")
    print("6. Show processed novels")
    print("7. Exit")

    while True:
        try:
            choice = input(f"\nSelect option (1-7): ").strip()

            if choice == "1":
                print("\nProcessing 5 novels for quick start...")
                results = processor.process_batch(5)
                print_results(results)
                break

            elif choice == "2":
                print("\nProcessing 10 novels...")
                results = processor.process_batch(10)
                print_results(results)
                break

            elif choice == "3":
                print("\nProcessing 25 novels...")
                results = processor.process_batch(25)
                print_results(results)
                break

            elif choice == "4":
                print("\nProcessing 50 novels...")
                results = processor.process_batch(50)
                print_results(results)
                break

            elif choice == "5":
                print(f"\nProcessing all {remaining} remaining novels...")
                # Process in chunks to avoid issues
                total_processed = 0
                total_failed = 0
                batch_size = 20

                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    print(f"\nProcessing batch of {current_batch} novels...")
                    results = processor.process_batch(current_batch)
                    total_processed += results['processed']
                    total_failed += results['failed']
                    remaining = results['remaining']

                    if remaining > 0:
                        print(f"Continuing with {remaining} remaining...")

                print(f"\nFinal Results:")
                print(f"  Total processed: {total_processed}")
                print(f"  Total failed: {total_failed}")
                break

            elif choice == "6":
                processor.list_processed()
                continue

            elif choice == "7":
                print("Exiting...")
                break

            else:
                print("Please select 1-7")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

def print_results(results):
    """Print processing results"""
    print(f"\nResults:")
    print(f"  SUCCESS: {results['processed']} novels")
    if results['failed'] > 0:
        print(f"  FAILED: {results['failed']} novels")
    if results['remaining'] > 0:
        print(f"  REMAINING: {results['remaining']} novels")
        print(f"  Run again to continue processing")

    if results['processed'] > 0:
        print(f"\nReady for training!")
        print(f"  python single_novel_trainer.py")
        print(f"  python simple_novel_processor.py --list")

if __name__ == "__main__":
    main()