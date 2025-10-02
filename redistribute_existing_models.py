#!/usr/bin/env python3
"""
Redistribute Existing Models Based on Novel Length Tiers

This script:
1. Loads existing model_mapping.json (with 21-29 novels per model)
2. Loads tier analysis data (tier_*.json)
3. For each existing model's novels, determines their tiers
4. Redistributes novels into smaller 4-novel groups
5. Preserves 2-letter prefix codes
6. Applies adaptive training parameters based on combined word count
7. Outputs new model_mapping.json ready for training

Goal: Break down unrealistically large models into optimal 4-novel groups
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict


class AdaptiveTrainingParameters:
    """Calculate adaptive training parameters based on text length"""

    @staticmethod
    def calculate_parameters(total_words: int) -> Dict[str, Any]:
        """Calculate training parameters based on combined word count"""
        if total_words < 80000:
            return {
                'iterations': 8,
                'steps_per_iteration': 100,
                'learning_rate_start': 3e-5,
                'learning_rate_end': 8e-6,
                'category': 'tiny',
                'rationale': 'Small corpus, fewer iterations but sufficient steps'
            }
        elif total_words < 120000:
            return {
                'iterations': 10,
                'steps_per_iteration': 150,
                'learning_rate_start': 2.5e-5,
                'learning_rate_end': 6e-6,
                'category': 'small',
                'rationale': 'Small corpus, balanced training'
            }
        elif total_words < 180000:
            return {
                'iterations': 12,
                'steps_per_iteration': 200,
                'learning_rate_start': 2e-5,
                'learning_rate_end': 5e-6,
                'category': 'medium',
                'rationale': 'Optimal corpus size, quality focus'
            }
        elif total_words < 260000:
            return {
                'iterations': 14,
                'steps_per_iteration': 250,
                'learning_rate_start': 1.5e-5,
                'learning_rate_end': 4e-6,
                'category': 'large',
                'rationale': 'Large corpus, extended learning needed'
            }
        else:
            return {
                'iterations': 16,
                'steps_per_iteration': 300,
                'learning_rate_start': 1e-5,
                'learning_rate_end': 3e-6,
                'category': 'xlarge',
                'rationale': 'Very large corpus, maximum iterations and steps'
            }


class ModelRedistributor:
    """Redistribute large existing models into optimal 4-novel groups"""

    def __init__(self):
        self.existing_models = {}
        self.tier_lookup = {}  # directory_name -> tier info
        self.redistributed_models = {}
        self.research_log = {
            'metadata': {
                'study_name': 'Existing Model Redistribution',
                'date': datetime.now().isoformat(),
                'version': '1.0',
                'purpose': 'Break down large models into optimal 4-novel groups'
            },
            'original_models': {},
            'redistribution_decisions': [],
            'statistics': {}
        }

        # Thematic name variations for each dessert family
        self.dessert_variations = {
            'sprinkles': ['rainbow', 'chocolate', 'vanilla', 'strawberry', 'confetti', 'jimmies'],
            'bananasplit': ['classic', 'supreme', 'deluxe', 'royale', 'tropical', 'caramel'],
            'cherryfloat': ['classic', 'cherry', 'vanilla', 'chocolate', 'creamy', 'fizzy'],
            'cupcake': ['vanilla', 'chocolate', 'red_velvet', 'lemon', 'strawberry', 'funfetti'],
            'cheesecake': ['classic', 'strawberry', 'blueberry', 'chocolate', 'caramel', 'new_york'],
            'marshmallow': ['vanilla', 'toasted', 'chocolate', 'strawberry', 'fluffy', 'campfire'],
            'keylimepie': ['classic', 'zesty', 'tangy', 'creamy', 'tropical', 'florida'],
            'tiramisu': ['classic', 'espresso', 'chocolate', 'mocha', 'italiano', 'ladyfinger'],
            'limesoda': ['classic', 'fizzy', 'sparkling', 'zesty', 'citrus', 'refreshing'],
            'sorbet': ['lemon', 'mango', 'raspberry', 'lime', 'orange', 'tropical'],
            'peachcobbler': ['classic', 'southern', 'spiced', 'cinnamon', 'buttery', 'homemade'],
            'parfait': ['berry', 'granola', 'yogurt', 'chocolate', 'fruit', 'layered'],
            'smores': ['classic', 'campfire', 'chocolate', 'toasted', 'graham', 'gooey'],
            'sugarcookie': ['vanilla', 'chocolate_chip', 'snickerdoodle', 'frosted', 'sprinkled', 'cutout'],
            'creamsoda': ['vanilla', 'classic', 'fizzy', 'smooth', 'old_fashioned', 'float'],
            'mintchip': ['peppermint', 'chocolate', 'double', 'swirl', 'fresh', 'cool'],
            'rootbeer': ['classic', 'sassafras', 'float', 'old_fashioned', 'barrel', 'frosty'],
            'chocolateshake': ['classic', 'thick', 'creamy', 'fudge', 'malt', 'double']
        }

    def load_existing_models(self):
        """Load existing model_mapping.json"""
        print("Loading existing model_mapping.json...")

        if not Path('model_mapping.json').exists():
            raise FileNotFoundError("model_mapping.json not found!")

        with open('model_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.existing_models = data.get('models', {})

        print(f"  Found {len(self.existing_models)} existing models")

        # Log original model structure
        for model_key, model_data in self.existing_models.items():
            prefix = model_key.split('_')[0]
            novel_count = len(model_data.get('novels', []))

            self.research_log['original_models'][model_key] = {
                'prefix': prefix,
                'novel_count': novel_count,
                'description': model_data.get('description', '')
            }

            print(f"    {model_key}: {novel_count} novels (prefix: {prefix})")

    def load_tier_data(self):
        """Load tier data and create lookup by directory_name"""
        print("\nLoading tier data...")

        tier_files = ['tiny', 'very_short', 'short', 'long', 'very_long']

        for tier_name in tier_files:
            filename = f"tier_{tier_name}.json"

            if not Path(filename).exists():
                print(f"  Warning: {filename} not found, skipping...")
                continue

            with open(filename, 'r', encoding='utf-8') as f:
                novels = json.load(f)

                for novel in novels:
                    self.tier_lookup[novel['directory_name']] = {
                        **novel,
                        'tier': tier_name
                    }

            print(f"  Loaded {tier_name}: {len(novels)} novels")

        print(f"\nTotal novels in tier lookup: {len(self.tier_lookup)}")

    def enrich_novels_with_tiers(self, novels: List[Dict]) -> List[Dict]:
        """Add tier and word count info to novel list"""
        enriched = []

        for novel in novels:
            directory_name = novel.get('directory_name')

            if directory_name in self.tier_lookup:
                tier_info = self.tier_lookup[directory_name]
                enriched.append({
                    **novel,
                    'tier': tier_info['tier'],
                    'words': tier_info['words']
                })
            else:
                # Novel not in tiers - might be missing or deleted
                print(f"    Warning: {directory_name} not found in tiers, skipping...")

        return enriched

    def get_variation_name(self, base_name: str, group_num: int) -> str:
        """Get thematic variation name for a dessert family"""
        # Check if we have variations for this dessert
        if base_name in self.dessert_variations:
            variations = self.dessert_variations[base_name]
            # Use modulo to cycle through variations if we have more groups than variations
            variation_idx = (group_num - 1) % len(variations)
            return variations[variation_idx]
        else:
            # Fallback to numbered if no variations defined
            return str(group_num)

    def redistribute_model(self, model_key: str, model_data: Dict) -> List[Dict]:
        """Break down one large model into multiple 4-novel groups"""
        prefix = model_key.split('_')[0]
        base_name = model_key.split('_', 1)[1] if '_' in model_key else 'model'
        original_description = model_data.get('description', base_name.title())
        novels = model_data.get('novels', [])

        print(f"\nRedistributing: {model_key} ({original_description})")
        print(f"  Original: {len(novels)} novels")

        # Enrich with tier data
        enriched_novels = self.enrich_novels_with_tiers(novels)

        if len(enriched_novels) < len(novels):
            print(f"  Warning: Only {len(enriched_novels)}/{len(novels)} novels found in tiers")

        if len(enriched_novels) < 4:
            print(f"  Error: Not enough novels to create groups, skipping...")
            return []

        # Sort by tier and word count for strategic grouping
        # Group similar sizes together for more consistent training
        enriched_novels.sort(key=lambda x: (x['tier'], x['words']))

        # Create 4-novel groups
        groups = []
        group_num = 1

        for i in range(0, len(enriched_novels), 4):
            group_novels = enriched_novels[i:i+4]

            if len(group_novels) < 4:
                # Handle remainder: distribute to existing groups or skip
                print(f"  Remainder: {len(group_novels)} novels (will try to distribute)")
                # For now, skip remainders - could be improved
                continue

            total_words = sum(n['words'] for n in group_novels)
            params = AdaptiveTrainingParameters.calculate_parameters(total_words)

            # Get thematic variation name
            variation = self.get_variation_name(base_name, group_num)

            # Create new model key with thematic variation
            new_model_key = f"{prefix}_{variation}{base_name}"

            # Create description with variation theme
            new_description = f"{variation.replace('_', ' ').title()} {original_description}"

            group = {
                'model_key': new_model_key,
                'prefix': prefix,
                'base_name': base_name,
                'variation': variation,
                'description': new_description,
                'original_model': model_key,
                'group_number': group_num,
                'novels': group_novels,
                'total_words': total_words,
                'training_parameters': params,
                'tier_distribution': self._count_tiers(group_novels)
            }

            groups.append(group)

            print(f"  Group {group_num}: {new_model_key} ({new_description})")
            print(f"    Novels: {[n['directory_name'] for n in group_novels]}")
            print(f"    Total: {total_words:,} words")
            print(f"    Tiers: {group['tier_distribution']}")
            print(f"    Training: {params['iterations']} iter, {params['steps_per_iteration']} steps, category={params['category']}")

            self.research_log['redistribution_decisions'].append({
                'original_model': model_key,
                'new_model': new_model_key,
                'description': new_description,
                'variation': variation,
                'group_number': group_num,
                'novels': [n['directory_name'] for n in group_novels],
                'total_words': total_words,
                'tier_distribution': group['tier_distribution'],
                'training_parameters': params
            })

            group_num += 1

        print(f"  Created {len(groups)} groups from {model_key}")
        return groups

    def _count_tiers(self, novels: List[Dict]) -> Dict[str, int]:
        """Count novels per tier in a group"""
        tier_counts = defaultdict(int)
        for novel in novels:
            tier_counts[novel['tier']] += 1
        return dict(tier_counts)

    def redistribute_all_models(self):
        """Redistribute all existing models"""
        print("\n" + "=" * 80)
        print("REDISTRIBUTING ALL MODELS")
        print("=" * 80)

        all_groups = []

        for model_key, model_data in self.existing_models.items():
            groups = self.redistribute_model(model_key, model_data)
            all_groups.extend(groups)

        print(f"\n" + "=" * 80)
        print(f"Total new models created: {len(all_groups)}")
        print("=" * 80)

        return all_groups

    def convert_to_mapping_format(self, groups: List[Dict]):
        """Convert groups to model_mapping.json format"""
        mapping = {
            'metadata': {
                'total_models': len(groups),
                'created_by': 'Model Redistribution Script v1.0',
                'date': datetime.now().isoformat(),
                'description': 'Redistributed existing models into optimal 4-novel groups',
                'adaptive_parameters': True,
                'source': 'Broken down from large existing models'
            },
            'models': {}
        }

        for group in groups:
            model_key = group['model_key']
            params = group['training_parameters']

            novels_list = [
                {
                    'original_name': n.get('original_name', n['directory_name'].replace('_', ' ').title()),
                    'directory_name': n['directory_name'],
                    'word_count': n['words'],
                    'tier': n['tier']
                }
                for n in group['novels']
            ]

            mapping['models'][model_key] = {
                'description': group['description'],
                'variation': group['variation'],
                'base_name': group['base_name'],
                'original_model': group['original_model'],
                'novel_count': len(novels_list),
                'total_word_count': group['total_words'],
                'training_parameters': {
                    'max_iterations': params['iterations'],
                    'max_steps_per_iteration': params['steps_per_iteration'],
                    'learning_rate_start': params['learning_rate_start'],
                    'learning_rate_end': params['learning_rate_end'],
                    'size_category': params['category']
                },
                'tier_distribution': group['tier_distribution'],
                'novels': novels_list
            }

        # Update metadata
        mapping['metadata']['total_novels_assigned'] = sum(
            len(m['novels']) for m in mapping['models'].values()
        )
        mapping['metadata']['total_word_count'] = sum(
            m['total_word_count'] for m in mapping['models'].values()
        )

        self.redistributed_models = mapping

    def calculate_statistics(self):
        """Calculate statistics for research log"""
        if not self.redistributed_models:
            return

        models = self.redistributed_models['models']

        stats = {
            'original_model_count': len(self.existing_models),
            'new_model_count': len(models),
            'models_per_original': {},
            'word_count_distribution': {
                'min': min(m['total_word_count'] for m in models.values()),
                'max': max(m['total_word_count'] for m in models.values()),
                'mean': sum(m['total_word_count'] for m in models.values()) // len(models)
            },
            'training_parameters': {
                'iterations': defaultdict(int),
                'size_categories': defaultdict(int)
            },
            'tier_distributions': defaultdict(int)
        }

        # Count models per original
        for model_key, model_data in models.items():
            original = model_data['original_model']
            stats['models_per_original'][original] = stats['models_per_original'].get(original, 0) + 1

            # Count parameters
            params = model_data['training_parameters']
            stats['training_parameters']['iterations'][params['max_iterations']] += 1
            stats['training_parameters']['size_categories'][params['size_category']] += 1

            # Count tier distributions
            tier_dist = model_data['tier_distribution']
            tier_key = ', '.join(f"{tier}:{count}" for tier, count in sorted(tier_dist.items()))
            stats['tier_distributions'][tier_key] += 1

        self.research_log['statistics'] = stats

    def save_outputs(self):
        """Save all outputs"""
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)

        # Backup existing
        if Path('model_mapping.json').exists():
            backup_file = f'model_mapping_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy('model_mapping.json', backup_file)
            print(f"✓ Backed up model_mapping.json to {backup_file}")

        # Save new mapping
        with open('model_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.redistributed_models, f, indent=2)
        print(f"✓ Saved model_mapping.json")

        # Save research copy
        with open('redistributed_model_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.redistributed_models, f, indent=2)
        print(f"✓ Saved redistributed_model_mapping.json (research copy)")

        # Save log
        with open('redistribution_log.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.research_log, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved redistribution_log.yaml")

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 80)
        print("REDISTRIBUTION SUMMARY")
        print("=" * 80)

        stats = self.research_log['statistics']

        print(f"\nOriginal Models: {stats['original_model_count']}")
        print(f"New Models: {stats['new_model_count']}")

        print(f"\nBreakdown by Original Model:")
        for original, count in sorted(stats['models_per_original'].items()):
            print(f"  {original}: {count} new models")

        print(f"\nWord Count Distribution:")
        print(f"  Min: {stats['word_count_distribution']['min']:,} words")
        print(f"  Max: {stats['word_count_distribution']['max']:,} words")
        print(f"  Mean: {stats['word_count_distribution']['mean']:,} words")

        print(f"\nTraining Parameters:")
        print(f"  By iterations:")
        for iters, count in sorted(stats['training_parameters']['iterations'].items()):
            print(f"    {iters} iterations: {count} models")
        print(f"  By size category:")
        for category, count in sorted(stats['training_parameters']['size_categories'].items()):
            print(f"    {category}: {count} models")


def main():
    """Main execution"""
    print("=" * 80)
    print("MODEL REDISTRIBUTION - Break Down Large Models")
    print("=" * 80)

    redistributor = ModelRedistributor()

    # Load data
    redistributor.load_existing_models()
    redistributor.load_tier_data()

    # Redistribute
    groups = redistributor.redistribute_all_models()

    if not groups:
        print("\nNo groups created! Check warnings above.")
        return

    # Convert and save
    redistributor.convert_to_mapping_format(groups)
    redistributor.calculate_statistics()
    redistributor.save_outputs()
    redistributor.print_summary()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review redistribution_log.yaml for decisions")
    print("2. Train redistributed models:")
    print("   python combined_model_trainer.py --all-models")
    print("\nmodel_mapping.json now contains optimal 4-novel groups with adaptive parameters.")
    print()


if __name__ == "__main__":
    main()
