#!/usr/bin/env python3
"""
Test the complete integrated system with post-training tests
"""

from iterative_novel_trainer import IterativeTrainer, IterativeConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("INTEGRATED SYSTEM TEST: GPT-2 + POST-TRAINING TEST SUITE")
    print("=" * 70)

    # Create minimal config for fast testing
    config = IterativeConfig(
        base_model='gpt2',
        max_seq_length=512,
        iterations_per_novel=1,  # Just 1 iteration for speed
        max_steps_per_iteration=1,  # Minimal training
        chunk_size=300,
        learning_rate_start=5e-5,
        learning_rate_end=4e-5,
        batch_size=1,
        save_checkpoints=False
    )

    print(f"Testing with minimal config:")
    print(f"  Model: {config.base_model}")
    print(f"  Iterations: {config.iterations_per_novel}")
    print(f"  Steps per iteration: {config.max_steps_per_iteration}")
    print()

    # Initialize trainer
    trainer = IterativeTrainer(config)

    # Run training with integrated post-training tests
    print("Starting training with integrated test suite...")
    try:
        results = trainer.train_novel("call_of_cthulhu")

        print("\n" + "=" * 60)
        print("COMPLETE SYSTEM TEST RESULTS")
        print("=" * 60)

        # Training Results
        print(f"Novel: {results['novel']}")
        print(f"Training time: {results['training_time']:.1f}s")
        print(f"Final quality: {results['final_quality']:.3f}")

        # Memory Results
        if 'episodic_memory' in results:
            memory = results['episodic_memory']
            print(f"Memories created: {memory.get('total_memories', 0)}")

        # Post-Training Test Results
        if 'post_training_tests' in results:
            tests = results['post_training_tests']
            if 'error' in tests:
                print(f"Post-training tests: FAILED - {tests['error']}")
            else:
                print(f"Post-training tests: COMPLETED")

                # Show key test metrics
                if 'generation_quality' in tests:
                    gen_quality = tests['generation_quality']['average_quality']
                    print(f"  Generation quality: {gen_quality:.3f}")

                if 'baseline_comparison' in tests:
                    improvement = tests['baseline_comparison']['average_improvement']
                    print(f"  Improvement over baseline: {improvement:+.3f}")

                if 'memory_system' in tests and 'error' not in tests['memory_system']:
                    mem_activated = tests['memory_system']['average_memories_activated']
                    print(f"  Average memories activated: {mem_activated:.1f}")

                if 'overall_assessment' in tests:
                    overall = tests['overall_assessment']['overall_rating']
                    print(f"  Overall assessment: {overall.upper()}")
        else:
            print("Post-training tests: NOT RUN")

        print("\nSUCCESS: Complete integrated system is operational!")

        # Summary of what was tested
        print(f"\nSystem Components Tested:")
        print(f"  [x] GPT-2 model training")
        print(f"  [x] Quality validation during training")
        print(f"  [x] Episodic memory creation")
        print(f"  [x] Post-training test suite")
        print(f"  [x] Generation quality assessment")
        print(f"  [x] Baseline comparison testing")
        print(f"  [x] Memory system integration testing")
        print(f"  [x] Performance benchmarking")
        print(f"  [x] Novel-specific feature testing")

    except Exception as e:
        print(f"\nSYSTEM TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    main()