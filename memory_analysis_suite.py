#!/usr/bin/env python3
"""
Model Tea - Memory Analysis Suite
Copyright © ChaiQ LLC

Comprehensive measurement and analysis tools for post-memory model behavior.
Provides detailed comparisons, behavioral analysis, and insights into how
episodic memory affects model generation quality and characteristics.
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from memory_enhanced_model import MemoryEnhancedModel, MemoryEnhancedConfig

@dataclass
class AnalysisConfig:
    """Configuration for memory analysis suite"""
    # Test prompt categories
    test_character_prompts: List[str] = None
    test_setting_prompts: List[str] = None
    test_emotional_prompts: List[str] = None
    test_creative_prompts: List[str] = None

    # Analysis parameters
    num_iterations_per_prompt: int = 5  # Multiple runs for statistical significance
    save_detailed_responses: bool = True
    generate_visualizations: bool = True

    # Output settings
    output_dir: str = "memory_analysis_results"

    def __post_init__(self):
        if self.test_character_prompts is None:
            self.test_character_prompts = [
                "Tell me about the main character",
                "Describe the protagonist's personality",
                "Who are the important people in this story?",
                "What motivates the central character?",
                "How do characters interact with each other?"
            ]

        if self.test_setting_prompts is None:
            self.test_setting_prompts = [
                "Where does this story take place?",
                "Describe the atmosphere and setting",
                "What is the environment like?",
                "Tell me about the locations in the story",
                "How does the setting affect the mood?"
            ]

        if self.test_emotional_prompts is None:
            self.test_emotional_prompts = [
                "Describe an emotional scene from the story",
                "What feelings does this story evoke?",
                "Tell me about moments of tension or fear",
                "How do characters express their emotions?",
                "What is the emotional journey in this tale?"
            ]

        if self.test_creative_prompts is None:
            self.test_creative_prompts = [
                "Write a new scene in this style",
                "Continue the story in your own words",
                "Create dialogue between characters",
                "Imagine what happens next",
                "Write a creative interpretation of events"
            ]

class MemoryAnalysisSuite:
    """Comprehensive analysis suite for memory-enhanced models"""

    def __init__(self, model_name: str, config: AnalysisConfig = None):
        self.model_name = model_name
        self.config = config or AnalysisConfig()

        # Initialize memory-enhanced model
        memory_config = MemoryEnhancedConfig()
        self.enhanced_model = MemoryEnhancedModel(model_name, memory_config)

        # Results storage
        self.analysis_results = {
            "model_name": model_name,
            "test_results": [],
            "summary_statistics": {},
            "behavioral_insights": {},
            "memory_effectiveness": {}
        }

        # Create output directory
        self.output_dir = Path(self.config.output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete memory analysis suite"""
        print(f"Model Tea - Memory Analysis Suite")
        print(f"Analyzing: {self.model_name}")
        print("=" * 50)

        start_time = time.time()

        # Test all prompt categories
        all_prompts = [
            ("character", self.config.test_character_prompts),
            ("setting", self.config.test_setting_prompts),
            ("emotional", self.config.test_emotional_prompts),
            ("creative", self.config.test_creative_prompts)
        ]

        for category, prompts in all_prompts:
            print(f"\nTesting {category} prompts...")
            self._test_prompt_category(category, prompts)

        # Generate summary statistics
        self._calculate_summary_statistics()

        # Analyze behavioral patterns
        self._analyze_behavioral_patterns()

        # Evaluate memory effectiveness
        self._evaluate_memory_effectiveness()

        # Save results
        self._save_analysis_results()

        # Generate visualizations if enabled
        if self.config.generate_visualizations:
            self._generate_visualizations()

        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.1f}s")
        print(f"Results saved to: {self.output_dir}")

        return self.analysis_results

    def _test_prompt_category(self, category: str, prompts: List[str]):
        """Test a category of prompts multiple times"""
        category_results = []

        for prompt in prompts:
            prompt_results = []

            # Run multiple iterations for statistical significance
            for iteration in range(self.config.num_iterations_per_prompt):
                try:
                    result = self.enhanced_model.generate_with_memory_volley(prompt)
                    prompt_results.append(result)

                    if iteration == 0:  # Show progress for first iteration
                        baseline_len = len(result['baseline_response'].split())
                        enhanced_len = len(result['response'].split())
                        memories_used = len(result['activated_memories'])
                        print(f"  '{prompt[:40]}...' -> {memories_used} memories, {enhanced_len}/{baseline_len} words")

                except Exception as e:
                    print(f"  Error with prompt '{prompt[:40]}...': {e}")
                    continue

            if prompt_results:
                # Calculate statistics for this prompt
                prompt_analysis = self._analyze_prompt_results(prompt, prompt_results)
                prompt_analysis['category'] = category
                category_results.append(prompt_analysis)

        self.analysis_results['test_results'].extend(category_results)

    def _analyze_prompt_results(self, prompt: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a single prompt across multiple iterations"""
        if not results:
            return {"prompt": prompt, "error": "No valid results"}

        # Extract metrics from all iterations
        metrics = {
            "memory_utilization": [],
            "creativity_scores": [],
            "relevance_improvements": [],
            "length_changes": [],
            "vocabulary_improvements": [],
            "memories_activated": [],
            "generation_times": []
        }

        detailed_responses = []

        for result in results:
            if 'behavior_measurement' in result and result['behavior_measurement']:
                bm = result['behavior_measurement']
                metrics["memory_utilization"].append(bm.get('memory_utilization_score', 0))
                metrics["creativity_scores"].append(bm.get('creativity_score', 0))
                metrics["relevance_improvements"].append(bm.get('relevance_improvement', 0))
                metrics["length_changes"].append(bm.get('length_change_ratio', 0))
                metrics["vocabulary_improvements"].append(bm.get('vocabulary_diversity_change', 0))

            metrics["memories_activated"].append(len(result.get('activated_memories', [])))
            metrics["generation_times"].append(result.get('generation_time', 0))

            if self.config.save_detailed_responses:
                detailed_responses.append({
                    "baseline": result.get('baseline_response', ''),
                    "enhanced": result.get('response', ''),
                    "memories": result.get('activated_memories', [])
                })

        # Calculate statistics
        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[metric] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return {
            "prompt": prompt,
            "statistics": stats,
            "detailed_responses": detailed_responses if self.config.save_detailed_responses else [],
            "overall_improvement": self._calculate_overall_improvement(stats)
        }

    def _calculate_overall_improvement(self, stats: Dict) -> float:
        """Calculate overall improvement score (0-1 scale)"""
        if not stats:
            return 0.0

        # Weight different metrics
        weights = {
            "memory_utilization": 0.3,
            "creativity_scores": 0.2,
            "relevance_improvements": 0.2,
            "vocabulary_improvements": 0.3
        }

        total_score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in stats and 'mean' in stats[metric]:
                total_score += stats[metric]['mean'] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_summary_statistics(self):
        """Calculate overall summary statistics"""
        if not self.analysis_results['test_results']:
            return

        # Aggregate by category
        categories = {}
        for result in self.analysis_results['test_results']:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result['overall_improvement'])

        # Calculate category averages
        category_stats = {}
        for category, improvements in categories.items():
            if improvements:
                category_stats[category] = {
                    "mean_improvement": statistics.mean(improvements),
                    "best_improvement": max(improvements),
                    "consistency": 1.0 - (statistics.stdev(improvements) if len(improvements) > 1 else 0),
                    "prompt_count": len(improvements)
                }

        self.analysis_results['summary_statistics'] = {
            "categories": category_stats,
            "overall_improvement": statistics.mean([r['overall_improvement'] for r in self.analysis_results['test_results']]),
            "total_prompts_tested": len(self.analysis_results['test_results'])
        }

    def _analyze_behavioral_patterns(self):
        """Analyze behavioral patterns in memory usage"""
        memory_types_used = []
        emotional_patterns = []
        timing_patterns = []

        for result in self.analysis_results['test_results']:
            if 'detailed_responses' in result:
                for response in result['detailed_responses']:
                    for memory in response.get('memories', []):
                        memory_types_used.append(memory.get('memory_type', 'unknown'))
                        emotional_patterns.append(memory.get('emotional_tone', 'neutral'))

        # Analyze patterns
        from collections import Counter

        self.analysis_results['behavioral_insights'] = {
            "memory_type_preferences": dict(Counter(memory_types_used).most_common()),
            "emotional_tone_patterns": dict(Counter(emotional_patterns).most_common()),
            "memory_diversity": len(set(memory_types_used)) / len(memory_types_used) if memory_types_used else 0
        }

    def _evaluate_memory_effectiveness(self):
        """Evaluate overall memory system effectiveness"""
        all_results = self.analysis_results['test_results']

        # Memory activation success rate
        total_prompts = len(all_results)
        prompts_with_memories = sum(1 for r in all_results if any(len(resp.get('memories', [])) > 0 for resp in r.get('detailed_responses', [])))

        # Average memory utilization
        utilization_scores = []
        for result in all_results:
            if 'statistics' in result and 'memory_utilization' in result['statistics']:
                utilization_scores.append(result['statistics']['memory_utilization']['mean'])

        self.analysis_results['memory_effectiveness'] = {
            "activation_success_rate": prompts_with_memories / total_prompts if total_prompts > 0 else 0,
            "average_memory_utilization": statistics.mean(utilization_scores) if utilization_scores else 0,
            "system_reliability": len(utilization_scores) / total_prompts if total_prompts > 0 else 0,
            "recommendation": self._generate_recommendation()
        }

    def _generate_recommendation(self) -> str:
        """Generate recommendation based on analysis"""
        effectiveness = self.analysis_results.get('memory_effectiveness', {})
        summary = self.analysis_results.get('summary_statistics', {})

        success_rate = effectiveness.get('activation_success_rate', 0)
        improvement = summary.get('overall_improvement', 0)

        if success_rate > 0.8 and improvement > 0.3:
            return "Excellent: Memory system is highly effective and shows strong improvements"
        elif success_rate > 0.6 and improvement > 0.2:
            return "Good: Memory system shows positive impact with room for optimization"
        elif success_rate > 0.4:
            return "Fair: Memory system activates but limited impact on generation quality"
        else:
            return "Needs improvement: Memory system may need tuning or more diverse memories"

    def _save_analysis_results(self):
        """Save analysis results to files"""
        # Save complete results
        with open(self.output_dir / "complete_analysis.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

        # Save summary report
        summary_report = self._generate_summary_report()
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write(summary_report)

        print(f"Analysis saved: {self.output_dir}")

    def _generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        report = f"""
Model Tea - Memory Analysis Report
Model: {self.model_name}
=====================================

SUMMARY STATISTICS
Overall Improvement Score: {self.analysis_results['summary_statistics'].get('overall_improvement', 0):.3f}
Total Prompts Tested: {self.analysis_results['summary_statistics'].get('total_prompts_tested', 0)}

CATEGORY PERFORMANCE
"""

        for category, stats in self.analysis_results['summary_statistics'].get('categories', {}).items():
            report += f"  {category.title()}: {stats['mean_improvement']:.3f} (consistency: {stats['consistency']:.3f})\n"

        report += f"""
MEMORY EFFECTIVENESS
Activation Success Rate: {self.analysis_results['memory_effectiveness'].get('activation_success_rate', 0):.1%}
Average Memory Utilization: {self.analysis_results['memory_effectiveness'].get('average_memory_utilization', 0):.3f}
System Reliability: {self.analysis_results['memory_effectiveness'].get('system_reliability', 0):.1%}

BEHAVIORAL INSIGHTS
Memory Type Preferences: {list(self.analysis_results['behavioral_insights'].get('memory_type_preferences', {}).keys())[:3]}
Memory Diversity Score: {self.analysis_results['behavioral_insights'].get('memory_diversity', 0):.3f}

RECOMMENDATION
{self.analysis_results['memory_effectiveness'].get('recommendation', 'No recommendation available')}
"""

        return report

    def _generate_visualizations(self):
        """Generate analysis visualizations"""
        try:
            # This would generate charts if matplotlib is available
            # For now, just create a simple data summary

            viz_data = {
                "categories": list(self.analysis_results['summary_statistics'].get('categories', {}).keys()),
                "improvements": [stats['mean_improvement'] for stats in self.analysis_results['summary_statistics'].get('categories', {}).values()],
                "memory_types": list(self.analysis_results['behavioral_insights'].get('memory_type_preferences', {}).keys()),
                "memory_counts": list(self.analysis_results['behavioral_insights'].get('memory_type_preferences', {}).values())
            }

            with open(self.output_dir / "visualization_data.json", 'w') as f:
                json.dump(viz_data, f, indent=2)

        except Exception as e:
            print(f"Visualization generation failed: {e}")

def main():
    """Test the memory analysis suite"""
    print("Model Tea - Memory Analysis Suite Test")
    print("=" * 40)

    # Test with available model
    model_name = "agony_column"  # Assuming this exists

    try:
        config = AnalysisConfig(
            num_iterations_per_prompt=2,  # Reduced for testing
            save_detailed_responses=True,
            generate_visualizations=True
        )

        analyzer = MemoryAnalysisSuite(model_name, config)
        results = analyzer.run_comprehensive_analysis()

        print(f"\nAnalysis complete!")
        print(f"Overall improvement: {results['summary_statistics']['overall_improvement']:.3f}")
        print(f"Memory activation rate: {results['memory_effectiveness']['activation_success_rate']:.1%}")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()