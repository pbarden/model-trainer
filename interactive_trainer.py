#!/usr/bin/env python3
"""
Interactive Model Trainer

A robust interactive interface for managing and training combined models.
Provides model browsing, filtering, status tracking, and training control.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time


class InteractiveTrainer:
    """Interactive model training interface"""

    def __init__(self, models_dir: str = "iterative_models"):
        self.models_dir = Path(models_dir)
        self.model_mapping = self._load_model_mapping()
        self.training_history = []

        # Command registry
        self.commands = {
            '/help': self.show_help,
            '/h': self.show_help,
            '/list': self.list_models,
            '/ls': self.list_models,
            '/trained': self.show_trained_models,
            '/untrained': self.show_untrained_models,
            '/pending': self.show_untrained_models,
            '/status': self.show_status,
            '/info': self.show_model_info,
            '/search': self.search_models,
            '/filter': self.filter_models,
            '/train': self.train_model,
            '/trainall': self.train_all_models,
            '/history': self.show_training_history,
            '/stats': self.show_statistics,
            '/quit': None,
            '/exit': None,
            '/q': None
        }

    def _load_model_mapping(self) -> Dict[str, Any]:
        """Load model mapping configuration"""
        mapping_file = Path("model_mapping.json")
        if not mapping_file.exists():
            print("Error: model_mapping.json not found!")
            return {"models": {}}

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model_mapping.json: {e}")
            return {"models": {}}

    def get_model_status(self, model_key: str) -> Dict[str, Any]:
        """Get training status for a model"""
        model_path = self.models_dir / model_key / "final"

        status = {
            'model_key': model_key,
            'trained': model_path.exists(),
            'path': str(model_path) if model_path.exists() else None
        }

        # Check for training results
        results_file = self.models_dir / model_key / "training_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    status['final_quality_score'] = results.get('final_quality_score')
                    status['total_training_time'] = results.get('total_training_time')
                    status['iterations_completed'] = len(results.get('iterations', []))
            except:
                pass

        # Get model info from mapping
        if model_key in self.model_mapping.get('models', {}):
            model_data = self.model_mapping['models'][model_key]
            status['description'] = model_data.get('description', 'N/A')
            status['novel_count'] = model_data.get('novel_count', 0)
            status['total_words'] = model_data.get('total_word_count', 0)
            status['training_params'] = model_data.get('training_parameters', {})

        return status

    def list_models(self, args: List[str] = None):
        """List all available models with status"""
        models = self.model_mapping.get('models', {})

        if not models:
            print("\nNo models found in model_mapping.json\n")
            return

        print("\n" + "=" * 100)
        print("AVAILABLE MODELS")
        print("=" * 100)
        print(f"{'Model Key':<35} {'Status':<12} {'Novels':<8} {'Words':<12} {'Description':<30}")
        print("-" * 100)

        for model_key in sorted(models.keys()):
            status = self.get_model_status(model_key)

            status_str = "[TRAINED]" if status['trained'] else "[PENDING]"
            novel_count = status.get('novel_count', 0)
            words = status.get('total_words', 0)
            desc = status.get('description', 'N/A')[:28]

            print(f"{model_key:<35} {status_str:<12} {novel_count:<8} {words:<12,} {desc:<30}")

        print("=" * 100)
        print(f"Total: {len(models)} models")
        print()

    def show_trained_models(self, args: List[str] = None):
        """Show only trained models"""
        models = self.model_mapping.get('models', {})
        trained = []

        for model_key in models.keys():
            status = self.get_model_status(model_key)
            if status['trained']:
                trained.append((model_key, status))

        if not trained:
            print("\nNo trained models found.\n")
            return

        print("\n" + "=" * 100)
        print(f"TRAINED MODELS ({len(trained)})")
        print("=" * 100)
        print(f"{'Model Key':<35} {'Quality':<10} {'Time':<12} {'Iterations':<12} {'Description':<30}")
        print("-" * 100)

        for model_key, status in sorted(trained):
            quality = status.get('final_quality_score', 'N/A')
            if isinstance(quality, (int, float)):
                quality = f"{quality:.3f}"

            train_time = status.get('total_training_time', 0)
            if train_time:
                hours = int(train_time // 3600)
                minutes = int((train_time % 3600) // 60)
                time_str = f"{hours}h {minutes}m"
            else:
                time_str = "N/A"

            iterations = status.get('iterations_completed', 'N/A')
            desc = status.get('description', 'N/A')[:28]

            print(f"{model_key:<35} {str(quality):<10} {time_str:<12} {str(iterations):<12} {desc:<30}")

        print("=" * 100)
        print()

    def show_untrained_models(self, args: List[str] = None):
        """Show only untrained/pending models"""
        models = self.model_mapping.get('models', {})
        untrained = []

        for model_key in models.keys():
            status = self.get_model_status(model_key)
            if not status['trained']:
                untrained.append((model_key, status))

        if not untrained:
            print("\nAll models are trained!\n")
            return

        print("\n" + "=" * 100)
        print(f"PENDING MODELS ({len(untrained)})")
        print("=" * 100)
        print(f"{'Model Key':<35} {'Novels':<8} {'Words':<12} {'Iterations':<12} {'Description':<30}")
        print("-" * 100)

        for model_key, status in sorted(untrained):
            novel_count = status.get('novel_count', 0)
            words = status.get('total_words', 0)
            params = status.get('training_params', {})
            iterations = params.get('max_iterations', 'N/A')
            desc = status.get('description', 'N/A')[:28]

            print(f"{model_key:<35} {novel_count:<8} {words:<12,} {str(iterations):<12} {desc:<30}")

        print("=" * 100)
        print()

    def show_status(self, args: List[str] = None):
        """Show overall training status"""
        models = self.model_mapping.get('models', {})
        total = len(models)
        trained = sum(1 for k in models.keys() if self.get_model_status(k)['trained'])
        pending = total - trained

        print("\n" + "=" * 80)
        print("TRAINING STATUS")
        print("=" * 80)
        print(f"Total Models: {total}")
        print(f"Trained: {trained} ({trained/total*100:.1f}%)" if total > 0 else "Trained: 0")
        print(f"Pending: {pending} ({pending/total*100:.1f}%)" if total > 0 else "Pending: 0")

        if self.training_history:
            print(f"\nModels Trained This Session: {len(self.training_history)}")
            print("Recent:")
            for entry in self.training_history[-5:]:
                print(f"  - {entry['model_key']}: {entry['status']}")

        print("=" * 80)
        print()

    def show_model_info(self, args: List[str] = None):
        """Show detailed information about a specific model"""
        if not args:
            model_key = input("Enter model key: ").strip()
        else:
            model_key = args[0]

        if model_key not in self.model_mapping.get('models', {}):
            print(f"\nModel '{model_key}' not found!\n")
            return

        status = self.get_model_status(model_key)
        model_data = self.model_mapping['models'][model_key]

        print("\n" + "=" * 80)
        print(f"MODEL INFO: {model_key}")
        print("=" * 80)
        print(f"Description: {status.get('description', 'N/A')}")
        print(f"Status: {'TRAINED' if status['trained'] else 'PENDING'}")

        if status.get('training_params'):
            params = status['training_params']
            print(f"\nTraining Parameters:")
            print(f"  Iterations: {params.get('max_iterations', 'N/A')}")
            print(f"  Steps/Iteration: {params.get('max_steps_per_iteration', 'N/A')}")
            print(f"  Learning Rate: {params.get('learning_rate_start', 'N/A'):.2e} -> {params.get('learning_rate_end', 'N/A'):.2e}")
            print(f"  Size Category: {params.get('size_category', 'N/A')}")

        print(f"\nNovel Information:")
        print(f"  Count: {status.get('novel_count', 0)}")
        print(f"  Total Words: {status.get('total_words', 0):,}")

        if model_data.get('tier_distribution'):
            print(f"  Tier Distribution:")
            for tier, count in model_data['tier_distribution'].items():
                print(f"    {tier}: {count}")

        print(f"\nNovels:")
        for i, novel in enumerate(model_data.get('novels', []), 1):
            print(f"  {i}. {novel.get('original_name', 'N/A')} ({novel.get('word_count', 0):,} words, {novel.get('tier', 'N/A')})")

        if status['trained']:
            print(f"\nTraining Results:")
            if status.get('final_quality_score'):
                print(f"  Quality Score: {status['final_quality_score']:.3f}")
            if status.get('total_training_time'):
                hours = int(status['total_training_time'] // 3600)
                minutes = int((status['total_training_time'] % 3600) // 60)
                print(f"  Training Time: {hours}h {minutes}m")
            if status.get('iterations_completed'):
                print(f"  Iterations: {status['iterations_completed']}")
            print(f"  Path: {status.get('path', 'N/A')}")

        print("=" * 80)
        print()

    def search_models(self, args: List[str] = None):
        """Search models by keyword"""
        if not args:
            keyword = input("Enter search keyword: ").strip().lower()
        else:
            keyword = ' '.join(args).lower()

        models = self.model_mapping.get('models', {})
        matches = []

        for model_key, model_data in models.items():
            # Search in key, description, and novel names
            if (keyword in model_key.lower() or
                keyword in model_data.get('description', '').lower() or
                any(keyword in novel.get('original_name', '').lower()
                    for novel in model_data.get('novels', []))):
                matches.append(model_key)

        if not matches:
            print(f"\nNo models found matching '{keyword}'\n")
            return

        print(f"\n" + "=" * 100)
        print(f"SEARCH RESULTS: '{keyword}' ({len(matches)} matches)")
        print("=" * 100)
        print(f"{'Model Key':<35} {'Status':<12} {'Description':<50}")
        print("-" * 100)

        for model_key in sorted(matches):
            status = self.get_model_status(model_key)
            status_str = "[TRAINED]" if status['trained'] else "[PENDING]"
            desc = status.get('description', 'N/A')[:48]
            print(f"{model_key:<35} {status_str:<12} {desc:<50}")

        print("=" * 100)
        print()

    def filter_models(self, args: List[str] = None):
        """Filter models by criteria"""
        print("\nFilter Options:")
        print("  1. By prefix (e.g., bc, vs, xo)")
        print("  2. By size category (tiny, small, medium, large, xlarge)")
        print("  3. By word count range")
        print("  4. By tier distribution")

        choice = input("\nSelect filter (1-4): ").strip()

        models = self.model_mapping.get('models', {})
        filtered = []

        if choice == '1':
            prefix = input("Enter prefix: ").strip().lower()
            filtered = [k for k in models.keys() if k.startswith(prefix + '_')]

        elif choice == '2':
            category = input("Enter category (tiny/small/medium/large/xlarge): ").strip().lower()
            for model_key, model_data in models.items():
                params = model_data.get('training_parameters', {})
                if params.get('size_category', '').lower() == category:
                    filtered.append(model_key)

        elif choice == '3':
            min_words = int(input("Minimum words: ").strip() or 0)
            max_words = int(input("Maximum words: ").strip() or 999999999)
            for model_key, model_data in models.items():
                words = model_data.get('total_word_count', 0)
                if min_words <= words <= max_words:
                    filtered.append(model_key)

        elif choice == '4':
            tier = input("Enter tier (tiny/very_short/short/long): ").strip().lower()
            for model_key, model_data in models.items():
                tier_dist = model_data.get('tier_distribution', {})
                if tier in tier_dist and tier_dist[tier] > 0:
                    filtered.append(model_key)

        if not filtered:
            print("\nNo models match the filter criteria.\n")
            return

        print(f"\n" + "=" * 100)
        print(f"FILTERED RESULTS ({len(filtered)} matches)")
        print("=" * 100)
        print(f"{'Model Key':<35} {'Status':<12} {'Words':<12} {'Description':<40}")
        print("-" * 100)

        for model_key in sorted(filtered):
            status = self.get_model_status(model_key)
            status_str = "[TRAINED]" if status['trained'] else "[PENDING]"
            words = status.get('total_words', 0)
            desc = status.get('description', 'N/A')[:38]
            print(f"{model_key:<35} {status_str:<12} {words:<12,} {desc:<40}")

        print("=" * 100)
        print()

    def train_model(self, args: List[str] = None):
        """Train a specific model"""
        if not args:
            model_key = input("Enter model key to train: ").strip()
        else:
            model_key = args[0]

        if model_key not in self.model_mapping.get('models', {}):
            print(f"\nModel '{model_key}' not found!\n")
            return

        status = self.get_model_status(model_key)

        if status['trained']:
            confirm = input(f"Model '{model_key}' is already trained. Retrain? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.\n")
                return
            force_flag = "--force"
        else:
            force_flag = ""

        print(f"\nStarting training for: {model_key}")
        print(f"Description: {status.get('description', 'N/A')}")
        print(f"Novels: {status.get('novel_count', 0)}")
        print(f"Total Words: {status.get('total_words', 0):,}")

        params = status.get('training_params', {})
        if params:
            print(f"Iterations: {params.get('max_iterations', 'N/A')}")
            print(f"Category: {params.get('size_category', 'N/A')}")

        confirm = input("\nProceed with training? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.\n")
            return

        # Execute training
        import subprocess
        cmd = f"python combined_model_trainer.py --model {model_key} {force_flag}"

        print(f"\nExecuting: {cmd}\n")
        print("=" * 80)

        start_time = time.time()
        result = subprocess.run(cmd, shell=True)
        end_time = time.time()

        # Log to history
        self.training_history.append({
            'model_key': model_key,
            'timestamp': datetime.now().isoformat(),
            'duration': end_time - start_time,
            'status': 'success' if result.returncode == 0 else 'failed'
        })

        if result.returncode == 0:
            print("\n" + "=" * 80)
            print(f"Training completed successfully for {model_key}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"Training failed for {model_key}")
            print("=" * 80)

        print()

    def train_all_models(self, args: List[str] = None):
        """Train all untrained models"""
        models = self.model_mapping.get('models', {})
        untrained = [k for k in models.keys() if not self.get_model_status(k)['trained']]

        if not untrained:
            print("\nAll models are already trained!\n")
            return

        print(f"\nFound {len(untrained)} untrained models:")
        for i, model_key in enumerate(sorted(untrained)[:10], 1):
            print(f"  {i}. {model_key}")

        if len(untrained) > 10:
            print(f"  ... and {len(untrained) - 10} more")

        confirm = input(f"\nTrain all {len(untrained)} models? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.\n")
            return

        # Execute training
        import subprocess
        cmd = "python combined_model_trainer.py --all-models"

        print(f"\nExecuting: {cmd}\n")
        print("=" * 80)

        start_time = time.time()
        result = subprocess.run(cmd, shell=True)
        end_time = time.time()

        # Log to history
        self.training_history.append({
            'model_key': 'ALL_MODELS',
            'timestamp': datetime.now().isoformat(),
            'duration': end_time - start_time,
            'status': 'completed' if result.returncode == 0 else 'failed'
        })

        print("\n" + "=" * 80)
        print("Batch training completed")
        print("=" * 80)
        print()

    def show_training_history(self, args: List[str] = None):
        """Show training history for this session"""
        if not self.training_history:
            print("\nNo training history for this session.\n")
            return

        print("\n" + "=" * 80)
        print("TRAINING HISTORY (This Session)")
        print("=" * 80)
        print(f"{'Model Key':<35} {'Status':<12} {'Duration':<15} {'Timestamp':<25}")
        print("-" * 80)

        for entry in self.training_history:
            model_key = entry['model_key']
            status = entry['status']
            duration = entry['duration']

            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            duration_str = f"{hours}h {minutes}m"

            timestamp = entry['timestamp'][:19].replace('T', ' ')

            print(f"{model_key:<35} {status:<12} {duration_str:<15} {timestamp:<25}")

        print("=" * 80)
        print()

    def show_statistics(self, args: List[str] = None):
        """Show comprehensive statistics"""
        models = self.model_mapping.get('models', {})

        if not models:
            print("\nNo models available.\n")
            return

        total = len(models)
        trained = sum(1 for k in models.keys() if self.get_model_status(k)['trained'])

        # Categorize by size
        categories = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0, 'xlarge': 0}
        prefixes = {}
        total_words = 0
        total_novels = 0

        for model_key, model_data in models.items():
            prefix = model_key.split('_')[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

            params = model_data.get('training_parameters', {})
            category = params.get('size_category', 'unknown')
            if category in categories:
                categories[category] += 1

            total_words += model_data.get('total_word_count', 0)
            total_novels += model_data.get('novel_count', 0)

        print("\n" + "=" * 80)
        print("MODEL STATISTICS")
        print("=" * 80)
        print(f"\nOverall:")
        print(f"  Total Models: {total}")
        print(f"  Trained: {trained} ({trained/total*100:.1f}%)")
        print(f"  Pending: {total - trained} ({(total-trained)/total*100:.1f}%)")
        print(f"  Total Novels: {total_novels}")
        print(f"  Total Words: {total_words:,}")
        print(f"  Avg Words/Model: {total_words//total:,}")

        print(f"\nBy Size Category:")
        for category, count in sorted(categories.items()):
            if count > 0:
                print(f"  {category.title()}: {count} ({count/total*100:.1f}%)")

        print(f"\nBy Prefix:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count}")

        print("=" * 80)
        print()

    def show_help(self, args: List[str] = None):
        """Show help menu"""
        print("\n" + "=" * 80)
        print("INTERACTIVE MODEL TRAINER - HELP")
        print("=" * 80)
        print("\nAVAILABLE COMMANDS:")
        print("  /help, /h           - Show this help menu")
        print("  /list, /ls          - List all models with status")
        print("  /trained            - Show only trained models")
        print("  /untrained, /pending - Show only untrained models")
        print("  /status             - Show overall training status")
        print("")
        print("MODEL INFORMATION:")
        print("  /info [model_key]   - Show detailed model information")
        print("  /search <keyword>   - Search models by keyword")
        print("  /filter             - Filter models by criteria")
        print("  /stats              - Show comprehensive statistics")
        print("")
        print("TRAINING:")
        print("  /train [model_key]  - Train a specific model")
        print("  /trainall           - Train all untrained models")
        print("  /history            - Show training history (this session)")
        print("")
        print("EXIT:")
        print("  /quit, /exit, /q    - Exit the trainer")
        print("=" * 80)
        print()

    def show_main_menu(self):
        """Show main menu"""
        models = self.model_mapping.get('models', {})
        total = len(models)
        trained = sum(1 for k in models.keys() if self.get_model_status(k)['trained'])
        pending = total - trained

        print("\n" + "=" * 80)
        print(" INTERACTIVE MODEL TRAINER - MAIN MENU")
        print("=" * 80)
        print(f"\n Status: {trained}/{total} models trained ({pending} pending)")
        print()
        print("  BROWSE MODELS:")
        print("    1. List all models")
        print("    2. Show trained models")
        print("    3. Show untrained/pending models")
        print("    4. Search models")
        print("    5. Filter models")
        print()
        print("  MODEL INFORMATION:")
        print("    6. Show model details")
        print("    7. Show statistics")
        print("    8. Show training status")
        print()
        print("  TRAINING:")
        print("    9. Train a single model")
        print("   10. Train all untrained models")
        print("   11. Show training history")
        print()
        print("  OTHER:")
        print("   12. Help")
        print("    0. Exit")
        print("=" * 80)

    def handle_menu_choice(self, choice: str):
        """Handle menu selection"""
        if choice == '1':
            self.list_models()
        elif choice == '2':
            self.show_trained_models()
        elif choice == '3':
            self.show_untrained_models()
        elif choice == '4':
            self.search_models()
        elif choice == '5':
            self.filter_models()
        elif choice == '6':
            self.show_model_info()
        elif choice == '7':
            self.show_statistics()
        elif choice == '8':
            self.show_status()
        elif choice == '9':
            self.train_model()
        elif choice == '10':
            self.train_all_models()
        elif choice == '11':
            self.show_training_history()
        elif choice == '12':
            self.show_help()
        elif choice == '0':
            return False
        else:
            print(f"\nInvalid choice: {choice}\n")

        return True

    def monitor_training(self, model_key: str):
        """Monitor training progress in real-time"""
        import subprocess
        import threading

        results_file = self.models_dir / model_key / "training_results.json"
        last_iteration = -1

        print("\n" + "=" * 80)
        print(f"TRAINING MONITOR: {model_key}")
        print("=" * 80)
        print("\nPress Ctrl+C to stop monitoring (training will continue)\n")

        def check_progress():
            nonlocal last_iteration
            while True:
                try:
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            iterations = results.get('iterations', [])

                            if len(iterations) > last_iteration:
                                # New iteration completed
                                for i in range(last_iteration + 1, len(iterations)):
                                    iter_data = iterations[i]
                                    print(f"[Iteration {iter_data.get('iteration', i)}] " +
                                          f"Perplexity: {iter_data.get('perplexity', 'N/A'):.2f} " +
                                          f"Quality: {iter_data.get('quality_score', 'N/A'):.3f} " +
                                          f"LR: {iter_data.get('learning_rate', 'N/A'):.2e}")

                                last_iteration = len(iterations) - 1

                    time.sleep(5)  # Check every 5 seconds
                except:
                    break

        # Start monitoring thread
        monitor_thread = threading.Thread(target=check_progress, daemon=True)
        monitor_thread.start()

    def run_menu_mode(self):
        """Run in menu mode"""
        print("\n" + "=" * 80)
        print(" INTERACTIVE MODEL TRAINER")
        print("=" * 80)
        print(" Manage and train your combined models")
        print("=" * 80)

        # Show initial status
        models = self.model_mapping.get('models', {})
        total = len(models)
        trained = sum(1 for k in models.keys() if self.get_model_status(k)['trained'])

        print(f"\nLoaded: {total} models ({trained} trained, {total-trained} pending)")

        while True:
            try:
                self.show_main_menu()
                choice = input("\nSelect option (0-12): ").strip()

                if choice == '0':
                    print("\nExiting trainer...")
                    break

                if not self.handle_menu_choice(choice):
                    break

                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\nExiting trainer...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                input("\nPress Enter to continue...")

    def run_command_mode(self):
        """Run in command mode (original interface)"""
        print("\n" + "=" * 80)
        print(" INTERACTIVE MODEL TRAINER - COMMAND MODE")
        print("=" * 80)
        print(" Manage and train your combined models")
        print(" Type '/help' for available commands or '/menu' for menu mode")
        print("=" * 80)

        # Show initial status
        models = self.model_mapping.get('models', {})
        total = len(models)
        trained = sum(1 for k in models.keys() if self.get_model_status(k)['trained'])

        print(f"\nLoaded: {total} models ({trained} trained, {total-trained} pending)")
        print()

        while True:
            try:
                user_input = input("Trainer> ").strip()

                if not user_input:
                    continue

                # Check for menu mode switch
                if user_input.lower() == '/menu':
                    self.run_menu_mode()
                    return

                # Parse command and args
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else None

                # Handle exit
                if command in ['/quit', '/exit', '/q']:
                    print("\nExiting trainer...")
                    break

                # Handle commands
                if command in self.commands:
                    handler = self.commands[command]
                    if handler:
                        handler(args)
                else:
                    print(f"Unknown command: {command}")
                    print("Type '/help' for available commands\n")

            except KeyboardInterrupt:
                print("\n\nExiting trainer...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")

    def run(self, mode: str = 'menu'):
        """Main entry point - choose mode"""
        if mode == 'menu':
            self.run_menu_mode()
        else:
            self.run_command_mode()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive Model Trainer - Manage and train combined models"
    )
    parser.add_argument(
        '--models-dir',
        default='iterative_models',
        help='Directory containing trained models (default: iterative_models)'
    )
    parser.add_argument(
        '--mode',
        choices=['menu', 'command'],
        default='menu',
        help='Interface mode: menu (numbered options) or command (slash commands)'
    )

    args = parser.parse_args()

    trainer = InteractiveTrainer(models_dir=args.models_dir)
    trainer.run(mode=args.mode)


if __name__ == "__main__":
    main()
