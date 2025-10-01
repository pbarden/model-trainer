#!/usr/bin/env python3
"""
Extended Training for Lovecraft Mythos Model
Trains until target perplexity is reached or max iterations hit

Usage:
    python train_lovecraft_extended.py [--iterations N] [--target-perplexity P]

Examples:
    python train_lovecraft_extended.py --iterations 30
    python train_lovecraft_extended.py --iterations 50 --target-perplexity 8.0
"""

import sys
import argparse
from pathlib import Path
from combined_model_trainer import CombinedModelTrainer, CombinedModelConfig
from iterative_novel_trainer import IterativeConfig

def main():
    parser = argparse.ArgumentParser(
        description="Extended training for Lovecraft Mythos model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=30,
        help='Maximum number of training iterations (default: 30)'
    )
    parser.add_argument(
        '--target-perplexity',
        type=float,
        default=10.0,
        help='Target perplexity to achieve (default: 10.0)'
    )
    parser.add_argument(
        '--min-iterations',
        type=int,
        default=12,
        help='Minimum iterations before adaptive stopping (default: 12)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience (default: 3)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retrain even if model exists'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LOVECRAFT MYTHOS - EXTENDED TRAINING")
    print("=" * 60)

    print(f"\nTraining Configuration:")
    print(f"  Max Iterations: {args.iterations}")
    print(f"  Min Iterations: {args.min_iterations}")
    print(f"  Target Perplexity: {args.target_perplexity}")
    print(f"  Early Stopping Patience: {args.patience}")
    print(f"  Learning Rate: 3e-5 -> 1e-6")

    # Create combined model config
    combined_config = CombinedModelConfig()
    combined_config.max_iterations = args.iterations
    combined_config.learning_rate_start = 3e-5
    combined_config.learning_rate_end = 1e-6

    # Create trainer with custom config
    # Note: We need to monkey-patch the trainer to use our extended config
    trainer = CombinedModelTrainer(combined_config)

    # Patch the train_combined_model method to use our custom settings
    original_train_combined_model = trainer.train_combined_model

    def patched_train_combined_model(model_key: str):
        """Patched train_model that applies our extended configuration"""
        from iterative_novel_trainer import IterativeTrainer, IterativeConfig

        # Load model data
        if model_key not in trainer.model_mapping["models"]:
            raise ValueError(f"Model '{model_key}' not found in model_mapping.json")

        model_data = trainer.model_mapping["models"][model_key]
        novels = model_data.get("novels", [])

        # Combine content
        combined_content = trainer.combine_novel_contents(novels, trainer.config.combine_novels_method)
        model_output_dir = trainer.models_dir / model_key

        # Create EXTENDED training config
        training_config = IterativeConfig()
        training_config.iterations_per_novel = args.iterations  # KEY FIX: Use our iterations
        training_config.max_iterations = args.iterations
        training_config.min_iterations = args.min_iterations
        training_config.target_perplexity = args.target_perplexity
        training_config.perplexity_threshold = args.target_perplexity * 1.5
        training_config.adaptive_training = True
        training_config.early_stopping_patience = args.patience
        training_config.learning_rate_start = combined_config.learning_rate_start
        training_config.learning_rate_end = combined_config.learning_rate_end

        # Create trainer with extended config
        iterative_trainer = IterativeTrainer(training_config)

        # Train with content
        training_results = iterative_trainer._train_with_content(combined_content, model_key)

        # Add metadata
        training_results.update({
            "model_type": "combined",
            "model_key": model_key,
            "novels_included": [novel["original_name"] for novel in novels],
            "novel_count": len(novels),
        })

        return training_results

    trainer.train_combined_model = patched_train_combined_model

    print(f"\nStarting extended training for lovecraft_mythos...")
    print("Training will continue until:")
    print(f"  1. Perplexity reaches {args.target_perplexity} or below")
    print(f"  2. No improvement for {args.patience} iterations")
    print(f"  3. Maximum {args.iterations} iterations reached")
    print("=" * 60 + "\n")

    # Train the model
    try:
        results = trainer.train_combined_model("lovecraft_mythos")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)

        if results and "training_metrics" in results:
            metrics = results["training_metrics"]
            iterations = results.get("iterations", [])

            print(f"\nTotal Iterations: {len(iterations)}")
            if iterations:
                final_iter = iterations[-1]
                print(f"Final Perplexity: {final_iter.get('perplexity', 'N/A')}")
                print(f"Final Quality Score: {final_iter.get('quality_score', 'N/A'):.3f}")

                # Show improvement
                first_iter = iterations[0]
                perp_improvement = first_iter.get('perplexity', 0) - final_iter.get('perplexity', 0)
                print(f"Perplexity Improvement: {perp_improvement:.2f}")

            print(f"\nModel saved to: iterative_models/lovecraft_mythos/final/")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
