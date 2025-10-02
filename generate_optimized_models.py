#!/usr/bin/env python3
"""
Research Study: Optimized Model Generation with Adaptive Training Parameters

This script generates optimized combined models by:
1. Analyzing novel length tiers (from analyze_novel_lengths.py output)
2. Creating strategic mixes across length tiers for diversity
3. Calculating adaptive training parameters based on combined word count
4. Generating model_mapping.json with embedded training parameters
5. Logging all decisions in YAML format for research analysis

Training parameters adapt based on combined word count:
- TINY (< 80K): 8-10 iterations, 12 steps/iter, LR 3e-5 to 8e-6
- SMALL (80K-120K): 10-12 iterations, 14 steps/iter, LR 2.5e-5 to 6e-6
- MEDIUM (120K-180K): 12-14 iterations, 16 steps/iter, LR 2e-5 to 5e-6
- LARGE (180K-260K): 14-16 iterations, 18 steps/iter, LR 1.5e-5 to 4e-6
- XLARGE (260K+): 16-18 iterations, 20 steps/iter, LR 1e-5 to 3e-6
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
import random


class AdaptiveTrainingParameters:
    """Calculate adaptive training parameters based on text length"""

    @staticmethod
    def calculate_parameters(total_words: int) -> Dict[str, Any]:
        """
        Calculate training parameters based on combined word count.

        Returns dict with:
        - iterations: number of training iterations
        - steps_per_iteration: training steps per iteration
        - learning_rate_start: initial learning rate
        - learning_rate_end: final learning rate
        - category: size category for logging
        """

        if total_words < 80000:
            # TINY: Quick training, fewer iterations
            return {
                'iterations': 8,
                'steps_per_iteration': 100,
                'learning_rate_start': 3e-5,
                'learning_rate_end': 8e-6,
                'category': 'tiny',
                'rationale': 'Small corpus, fewer iterations but sufficient steps'
            }
        elif total_words < 120000:
            # SMALL: Standard short training
            return {
                'iterations': 10,
                'steps_per_iteration': 150,
                'learning_rate_start': 2.5e-5,
                'learning_rate_end': 6e-6,
                'category': 'small',
                'rationale': 'Small corpus, balanced training'
            }
        elif total_words < 180000:
            # MEDIUM: Optimal range
            return {
                'iterations': 12,
                'steps_per_iteration': 200,
                'learning_rate_start': 2e-5,
                'learning_rate_end': 5e-6,
                'category': 'medium',
                'rationale': 'Optimal corpus size, quality focus'
            }
        elif total_words < 260000:
            # LARGE: Extended training
            return {
                'iterations': 14,
                'steps_per_iteration': 250,
                'learning_rate_start': 1.5e-5,
                'learning_rate_end': 4e-6,
                'category': 'large',
                'rationale': 'Large corpus, extended learning needed'
            }
        else:
            # XLARGE: Maximum training
            return {
                'iterations': 16,
                'steps_per_iteration': 300,
                'learning_rate_start': 1e-5,
                'learning_rate_end': 3e-6,
                'category': 'xlarge',
                'rationale': 'Very large corpus, maximum iterations and steps'
            }


class ModelGenerationResearch:
    """Research-grade model generation with adaptive parameters"""

    def __init__(self):
        self.tier_data = {}
        self.existing_models = {}
        self.generated_models = {}
        self.research_log = {
            'metadata': {
                'study_name': 'Optimized Combined Model Generation with Adaptive Training',
                'date': datetime.now().isoformat(),
                'version': '2.0',
                'researcher': 'Model Tea Research Team'
            },
            'methodology': {
                'target_novels_per_model': 4,
                'adaptive_parameters': True,
                'tier_definitions': {
                    'tiny': '< 20K words',
                    'very_short': '20K - 44K words',
                    'short': '44K - 67K words',
                    'long': '67K - 94K words'
                },
                'parameter_adaptation': {
                    'tiny': '< 80K combined: 10 iter, 12 steps',
                    'small': '80K-120K: 12 iter, 14 steps',
                    'medium': '120K-180K: 14 iter, 16 steps',
                    'large': '180K-260K: 16 iter, 18 steps',
                    'xlarge': '260K+: 18 iter, 20 steps'
                },
                'mixing_strategy': 'strategic cross-tier for diversity'
            },
            'data_sources': [],
            'model_generation_decisions': [],
            'statistics': {}
        }

    def load_tier_data(self):
        """Load novel tier data from JSON files"""
        print("Loading tier data...")

        tier_files = ['tiny', 'very_short', 'short', 'long', 'very_long']

        for tier_name in tier_files:
            filename = f"tier_{tier_name}.json"

            if not Path(filename).exists():
                print(f"Warning: {filename} not found, skipping...")
                continue

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.tier_data[tier_name] = data

            print(f"  Loaded {tier_name}: {len(data)} novels")

            self.research_log['data_sources'].append({
                'file': filename,
                'tier': tier_name,
                'novel_count': len(data),
                'total_words': sum(n['words'] for n in data),
                'avg_words': sum(n['words'] for n in data) // len(data) if data else 0
            })

    def load_existing_models(self):
        """Load existing model mapping to preserve prefix codes"""
        print("\nLoading existing model mapping...")

        if not Path('model_mapping.json').exists():
            print("Warning: model_mapping.json not found")
            return

        with open('model_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.existing_models = data.get('models', {})

        print(f"  Found {len(self.existing_models)} existing models")

        # Log existing model analysis
        for model_key, model_data in self.existing_models.items():
            prefix = model_key.split('_')[0]
            novel_count = len(model_data.get('novels', []))

            self.research_log['model_generation_decisions'].append({
                'stage': 'existing_model_analysis',
                'model_key': model_key,
                'prefix': prefix,
                'description': model_data.get('description', ''),
                'novel_count': novel_count,
                'status': 'too_large' if novel_count > 4 else 'acceptable',
                'action_required': 'split_into_smaller_models' if novel_count > 4 else 'none'
            })

    def create_novel_pool(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create pools of novels by tier for selection"""
        pools = {}

        for tier_name, novels in self.tier_data.items():
            # Create a copy so we can track assignments
            pools[tier_name] = [
                {
                    **novel,
                    'tier': tier_name,
                    'assigned': False
                }
                for novel in novels
            ]

        return pools

    def calculate_combined_words(self, novels: List[Dict]) -> int:
        """Calculate total words for a group of novels"""
        return sum(n.get('words', 0) for n in novels)

    def generate_model_key(self, prefix: str, model_type: str, variant: int) -> str:
        """Generate model key: prefix_typename_v# if multiple variants"""
        if variant > 0:
            return f"{prefix}_{model_type}{variant}"
        return f"{prefix}_{model_type}"

    def create_model_group(self, pools: Dict, tier_selection: List[Tuple[str, int]],
                           prefix: str, model_type: str, count: int) -> List[Dict]:
        """
        Generic model creation based on tier selection pattern.

        Args:
            pools: Novel pools by tier
            tier_selection: List of (tier_name, count) tuples
            prefix: Model prefix code
            model_type: Type name for model
            count: Number of models to create
        """
        models = []

        for i in range(count):
            selected = []

            # Select novels based on pattern
            for tier_name, tier_count in tier_selection:
                for _ in range(tier_count):
                    available = [n for n in pools.get(tier_name, []) if not n['assigned']]

                    if not available:
                        break

                    novel = random.choice(available)
                    novel['assigned'] = True
                    selected.append(novel)

            # Only create model if we got all required novels
            expected_count = sum(count for _, count in tier_selection)
            if len(selected) == expected_count:
                total_words = self.calculate_combined_words(selected)

                # Calculate adaptive parameters
                params = AdaptiveTrainingParameters.calculate_parameters(total_words)

                model = {
                    'prefix': prefix,
                    'type': model_type,
                    'variant': i + 1,
                    'novels': selected,
                    'total_words': total_words,
                    'training_parameters': params,
                    'tier_distribution': {tier: count for tier, count in tier_selection}
                }

                models.append(model)

                self.research_log['model_generation_decisions'].append({
                    'stage': 'model_creation',
                    'type': model_type,
                    'variant': i + 1,
                    'model_key': self.generate_model_key(prefix, model_type, i + 1),
                    'novels': [n['directory_name'] for n in selected],
                    'total_words': total_words,
                    'training_parameters': {
                        'iterations': params['iterations'],
                        'steps_per_iteration': params['steps_per_iteration'],
                        'learning_rate_start': params['learning_rate_start'],
                        'learning_rate_end': params['learning_rate_end'],
                        'category': params['category']
                    },
                    'rationale': params['rationale']
                })

        return models

    def generate_all_models(self):
        """Generate all model combinations based on research design"""
        print("\n" + "=" * 80)
        print("GENERATING OPTIMIZED MODELS WITH ADAPTIVE PARAMETERS")
        print("=" * 80)

        pools = self.create_novel_pool()

        # Use existing prefix codes or create new ones
        existing_prefixes = list(set(k.split('_')[0] for k in self.existing_models.keys()))

        if not existing_prefixes:
            existing_prefixes = ['rm']  # Research models

        prefix = existing_prefixes[0] if len(existing_prefixes) == 1 else 'rm'

        print(f"\nUsing prefix: {prefix}")
        print(f"Available novels by tier:")
        for tier, novels in pools.items():
            print(f"  {tier}: {len(novels)} novels")

        all_models = []

        # Model Type Definitions
        model_types = [
            {
                'name': 'balanced',
                'tier_selection': [('tiny', 1), ('very_short', 1), ('short', 1), ('long', 1)],
                'count': 15,
                'description': 'Balanced mix - 1 from each tier'
            },
            {
                'name': 'shortfocus',
                'tier_selection': [('tiny', 2), ('very_short', 2)],
                'count': 12,
                'description': 'Short focus - 2 tiny + 2 very short'
            },
            {
                'name': 'mediumfocus',
                'tier_selection': [('very_short', 2), ('short', 2)],
                'count': 15,
                'description': 'Medium focus - 2 very short + 2 short'
            },
            {
                'name': 'longfocus',
                'tier_selection': [('short', 1), ('long', 3)],
                'count': 7,
                'description': 'Long focus - 1 short + 3 long'
            },
            {
                'name': 'escalating',
                'tier_selection': [('tiny', 1), ('very_short', 1), ('short', 1), ('long', 1)],
                'count': 8,
                'description': 'Escalating - ordered by length'
            }
        ]

        print("\nGenerating model types...")

        for model_def in model_types:
            print(f"  {model_def['description']}...")
            models = self.create_model_group(
                pools,
                model_def['tier_selection'],
                prefix,
                model_def['name'],
                model_def['count']
            )

            # Special handling for escalating - sort by word count
            if model_def['name'] == 'escalating':
                for model in models:
                    model['novels'].sort(key=lambda x: x['words'])

            all_models.extend(models)
            print(f"    Created {len(models)} models")

        print(f"\nTotal models generated: {len(all_models)}")

        # Convert to model_mapping.json format
        self.convert_to_mapping_format(all_models)

        # Calculate statistics
        self.calculate_statistics(all_models, pools)

        return all_models

    def convert_to_mapping_format(self, models: List[Dict]):
        """Convert internal format to model_mapping.json format with training parameters"""
        mapping = {
            'metadata': {
                'total_models': len(models),
                'created_by': 'Optimized Model Generation Research Script v2.0',
                'date': datetime.now().isoformat(),
                'description': 'Research-optimized combined models with adaptive training parameters',
                'adaptive_parameters': True
            },
            'models': {}
        }

        for model in models:
            model_key = self.generate_model_key(
                model['prefix'],
                model['type'],
                model['variant']
            )

            params = model['training_parameters']

            novels_list = [
                {
                    'original_name': n.get('directory_name', '').replace('_', ' ').title(),
                    'directory_name': n['directory_name'],
                    'word_count': n['words'],
                    'tier': n['tier']
                }
                for n in model['novels']
            ]

            mapping['models'][model_key] = {
                'description': f"{model['type'].title()} {model['variant']}",
                'type': model['type'],
                'novel_count': len(novels_list),
                'total_word_count': model['total_words'],

                # Adaptive training parameters
                'training_parameters': {
                    'max_iterations': params['iterations'],
                    'max_steps_per_iteration': params['steps_per_iteration'],
                    'learning_rate_start': params['learning_rate_start'],
                    'learning_rate_end': params['learning_rate_end'],
                    'size_category': params['category']
                },

                'tier_distribution': model['tier_distribution'],
                'novels': novels_list
            }

        # Update metadata
        mapping['metadata']['total_novels_assigned'] = sum(
            len(m['novels']) for m in mapping['models'].values()
        )
        mapping['metadata']['total_word_count'] = sum(
            m['total_word_count'] for m in mapping['models'].values()
        )

        self.generated_models = mapping

    def calculate_statistics(self, models: List[Dict], pools: Dict):
        """Calculate comprehensive statistics for research analysis"""
        stats = {
            'model_counts': {
                'total': len(models),
                'by_type': defaultdict(int),
                'by_size_category': defaultdict(int)
            },
            'word_count_distribution': {
                'min': min(m['total_words'] for m in models) if models else 0,
                'max': max(m['total_words'] for m in models) if models else 0,
                'mean': sum(m['total_words'] for m in models) // len(models) if models else 0,
                'by_type': {}
            },
            'training_parameters': {
                'iterations': defaultdict(int),
                'steps_per_iteration': defaultdict(int),
                'size_categories': defaultdict(int)
            },
            'novel_usage': {
                'total_available': sum(len(novels) for novels in pools.values()),
                'total_assigned': sum(sum(1 for n in novels if n['assigned']) for novels in pools.values()),
                'by_tier': {}
            }
        }

        # Count by type and category
        for model in models:
            stats['model_counts']['by_type'][model['type']] += 1
            stats['model_counts']['by_size_category'][model['training_parameters']['category']] += 1
            stats['training_parameters']['iterations'][model['training_parameters']['iterations']] += 1
            stats['training_parameters']['steps_per_iteration'][model['training_parameters']['steps_per_iteration']] += 1
            stats['training_parameters']['size_categories'][model['training_parameters']['category']] += 1

        # Word count by type
        type_words = defaultdict(list)
        for model in models:
            type_words[model['type']].append(model['total_words'])

        for model_type, words_list in type_words.items():
            stats['word_count_distribution']['by_type'][model_type] = {
                'min': min(words_list),
                'max': max(words_list),
                'mean': sum(words_list) // len(words_list)
            }

        # Novel usage by tier
        for tier, novels in pools.items():
            assigned = sum(1 for n in novels if n['assigned'])
            stats['novel_usage']['by_tier'][tier] = {
                'available': len(novels),
                'assigned': assigned,
                'unassigned': len(novels) - assigned,
                'utilization_rate': f"{(assigned / len(novels) * 100):.1f}%" if novels else "0%"
            }

        self.research_log['statistics'] = stats

    def save_outputs(self, backup_existing: bool = True):
        """Save all outputs in research-grade formats"""
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)

        # Backup existing model_mapping.json
        if backup_existing and Path('model_mapping.json').exists():
            backup_file = f'model_mapping_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy('model_mapping.json', backup_file)
            print(f"✓ Backed up existing model_mapping.json to {backup_file}")

        # Save as new model_mapping.json (for combined_model_trainer.py)
        with open('model_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.generated_models, f, indent=2)
        print(f"✓ Saved model_mapping.json (ready for combined_model_trainer.py)")

        # Also save a copy for reference
        output_file = 'research_model_mapping.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.generated_models, f, indent=2)
        print(f"✓ Saved {output_file} (research copy)")

        # Save research log
        log_file = 'model_generation_log.yaml'
        with open(log_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.research_log, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved {log_file}")

        # Save statistics
        stats_file = 'model_statistics.yaml'
        with open(stats_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.research_log['statistics'], f, default_flow_style=False)
        print(f"✓ Saved {stats_file}")

    def print_summary(self):
        """Print research summary"""
        print("\n" + "=" * 80)
        print("RESEARCH SUMMARY")
        print("=" * 80)

        stats = self.research_log['statistics']

        print(f"\nModels Generated: {stats['model_counts']['total']}")
        print("\nBy Type:")
        for model_type, count in stats['model_counts']['by_type'].items():
            print(f"  {model_type}: {count}")

        print(f"\nBy Size Category:")
        for category, count in stats['model_counts']['by_size_category'].items():
            print(f"  {category}: {count}")

        print(f"\nWord Count Distribution:")
        print(f"  Min: {stats['word_count_distribution']['min']:,} words")
        print(f"  Max: {stats['word_count_distribution']['max']:,} words")
        print(f"  Mean: {stats['word_count_distribution']['mean']:,} words")

        print(f"\nTraining Parameter Distribution:")
        print(f"  Iterations:")
        for iters, count in sorted(stats['training_parameters']['iterations'].items()):
            print(f"    {iters} iterations: {count} models")
        print(f"  Steps per iteration:")
        for steps, count in sorted(stats['training_parameters']['steps_per_iteration'].items()):
            print(f"    {steps} steps: {count} models")

        print(f"\nNovel Utilization:")
        print(f"  Total Available: {stats['novel_usage']['total_available']}")
        print(f"  Total Assigned: {stats['novel_usage']['total_assigned']}")
        print(f"  By Tier:")
        for tier, data in stats['novel_usage']['by_tier'].items():
            print(f"    {tier}: {data['assigned']}/{data['available']} ({data['utilization_rate']})")


def main():
    """Main research execution"""
    print("=" * 80)
    print("MODEL TEA RESEARCH STUDY v2.0")
    print("Optimized Model Generation with Adaptive Training Parameters")
    print("=" * 80)

    research = ModelGenerationResearch()

    # Load data
    research.load_tier_data()
    research.load_existing_models()

    # Generate models
    research.generate_all_models()

    # Save outputs
    research.save_outputs()

    # Print summary
    research.print_summary()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review model_generation_log.yaml for decision rationale")
    print("2. Review model_statistics.yaml for distribution analysis")
    print("3. Update combined_model_trainer.py to read training_parameters from model_mapping.json")
    print("4. Train all models:")
    print("   python combined_model_trainer.py --all-models")
    print("\nEach model now has adaptive parameters based on its combined word count.")
    print()


if __name__ == "__main__":
    main()
