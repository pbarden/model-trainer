#!/usr/bin/env python3
"""
Quick test script to validate GPT-2 training works
"""

from iterative_novel_trainer import IterativeTrainer, IterativeConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("GPT-2 TRAINING VALIDATION TEST")
    print("=" * 60)

    # Create minimal test config - just 2 iterations, 2 steps each
    config = IterativeConfig(
        base_model='gpt2',
        max_seq_length=512,
        iterations_per_novel=2,  # Minimal test
        max_steps_per_iteration=2,  # Very fast
        chunk_size=300,  # Reasonable size
        learning_rate_start=5e-5,
        learning_rate_end=4e-5,
        batch_size=1,  # Small batch for speed
        save_checkpoints=False  # Skip saves for speed
    )

    print(f"Testing with config:")
    print(f"  Model: {config.base_model}")
    print(f"  Iterations: {config.iterations_per_novel}")
    print(f"  Steps per iteration: {config.max_steps_per_iteration}")
    print(f"  Sequence length: {config.max_seq_length}")
    print()

    # Initialize trainer
    trainer = IterativeTrainer(config)

    # Run minimal training on Call of Cthulhu
    print("Starting minimal training test...")
    try:
        results = trainer.train_novel("call_of_cthulhu")

        print("\n" + "=" * 50)
        print("TRAINING TEST RESULTS:")
        print("=" * 50)

        print(f"Novel: {results['novel']}")
        print(f"Total time: {results['training_time']:.1f}s")
        print(f"Iterations: {len(results['iterations'])}")
        print(f"Final quality: {results['final_quality']:.3f}")

        if 'episodic_memory' in results:
            memory = results['episodic_memory']
            print(f"Memories created: {memory.get('total_memories', 0)}")
            print(f"Memory density: {memory.get('memory_density', 0):.2f}")

        # Check if perplexity improved or stayed stable
        iterations = results['iterations']
        if len(iterations) >= 2:
            initial_perp = iterations[0]['perplexity']
            final_perp = iterations[-1]['perplexity']
            change = final_perp - initial_perp

            print(f"Perplexity: {initial_perp:.2f} -> {final_perp:.2f} (change: {change:+.2f})")

            if change < 5:  # Allow small increase, but not massive degradation
                print("SUCCESS: Perplexity stable (good sign!)")
            else:
                print("WARNING: Perplexity increased significantly")

        print("\nOVERALL: GPT-2 training system is operational!")

    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        raise

if __name__ == "__main__":
    main()