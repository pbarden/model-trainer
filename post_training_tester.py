#!/usr/bin/env python3
"""
Post-Training Test Suite
Comprehensive testing of trained models and memory systems
"""

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from episodic_memory_system import EpisodicMemorySystem
from quality_validator import QualityValidator, ValidationConfig

logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for post-training tests"""
    # Generation test settings
    test_prompts_per_category: int = 3
    generation_length: int = 150
    temperature: float = 0.8

    # Quality scoring
    min_quality_threshold: float = 0.7
    min_coherence_score: float = 0.6

    # Memory test settings
    memory_test_prompts: int = 5
    min_memory_activation: int = 2

    # Performance benchmarks
    max_generation_time: float = 10.0  # seconds
    max_memory_retrieval_time: float = 1.0  # seconds

class PostTrainingTester:
    """Comprehensive post-training test suite"""

    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.quality_validator = QualityValidator(ValidationConfig())

    def run_comprehensive_tests(self,
                               model_path: str,
                               novel_name: str,
                               novel_path: Path,
                               baseline_model_name: str = "gpt2") -> Dict[str, Any]:
        """Run complete test suite on trained model"""

        print(f"\n{'='*60}")
        print(f"POST-TRAINING TEST SUITE: {novel_name.upper()}")
        print(f"{'='*60}")

        test_results = {
            "model_path": model_path,
            "novel_name": novel_name,
            "test_timestamp": time.time(),
            "baseline_model": baseline_model_name,
            "test_config": {
                "generation_length": self.config.generation_length,
                "temperature": self.config.temperature,
                "quality_threshold": self.config.min_quality_threshold
            }
        }

        try:
            # Load trained model
            print("Loading trained model...")
            trained_model, trained_tokenizer = self._load_model(model_path)

            # Load baseline model for comparison
            print("Loading baseline model...")
            baseline_model, baseline_tokenizer = self._load_model(baseline_model_name)

            # Test 1: Generation Quality Tests
            print("\n--- GENERATION QUALITY TESTS ---")
            generation_results = self._test_generation_quality(
                trained_model, trained_tokenizer, novel_name
            )
            test_results["generation_quality"] = generation_results

            # Test 2: Baseline Comparison Tests
            print("\n--- BASELINE COMPARISON TESTS ---")
            comparison_results = self._compare_with_baseline(
                trained_model, trained_tokenizer,
                baseline_model, baseline_tokenizer,
                novel_name
            )
            test_results["baseline_comparison"] = comparison_results

            # Test 3: Memory System Tests
            print("\n--- MEMORY SYSTEM TESTS ---")
            memory_results = self._test_memory_system(
                trained_model, trained_tokenizer, novel_name, novel_path
            )
            test_results["memory_system"] = memory_results

            # Test 4: Performance Benchmarks
            print("\n--- PERFORMANCE BENCHMARKS ---")
            performance_results = self._test_performance(
                trained_model, trained_tokenizer, novel_name, novel_path
            )
            test_results["performance"] = performance_results

            # Test 5: Novel-Specific Tests
            print("\n--- NOVEL-SPECIFIC TESTS ---")
            novel_specific_results = self._test_novel_specific_features(
                trained_model, trained_tokenizer, novel_name
            )
            test_results["novel_specific"] = novel_specific_results

            # Generate Overall Assessment
            overall_assessment = self._generate_overall_assessment(test_results)
            test_results["overall_assessment"] = overall_assessment

            # Print Summary
            self._print_test_summary(test_results)

            return test_results

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            test_results["error"] = str(e)
            test_results["status"] = "failed"
            return test_results

    def _load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        return model, tokenizer

    def _test_generation_quality(self, model, tokenizer, novel_name: str) -> Dict[str, Any]:
        """Test generation quality with various prompts"""

        # Define test prompt categories
        test_prompts = {
            "narrative_start": [
                "It was a dark and stormy night when",
                "The old mansion stood silent until",
                "In the depths of the ancient library"
            ],
            "character_dialogue": [
                "The professor whispered urgently, \"",
                "She looked at him with terror and said, \"",
                "The stranger approached and declared, \""
            ],
            "atmospheric_description": [
                "The air grew thick with an otherworldly presence",
                "Strange shadows danced across the walls",
                "The silence was broken by an eerie sound"
            ],
            "horror_elements": [
                "The forbidden knowledge contained within",
                "Cosmic horror beyond human comprehension",
                "Ancient entities stirred in their slumber"
            ]
        }

        results = {
            "categories_tested": len(test_prompts),
            "total_prompts": sum(len(prompts) for prompts in test_prompts.values()),
            "category_results": {},
            "average_quality": 0.0,
            "quality_distribution": {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        }

        all_scores = []

        for category, prompts in test_prompts.items():
            category_scores = []
            category_samples = []

            for prompt in prompts:
                # Generate text
                generated_text = self._generate_text(model, tokenizer, prompt)

                # Score quality
                quality_score = self.quality_validator._assess_text_quality(generated_text)
                category_scores.append(quality_score)
                all_scores.append(quality_score)

                # Categorize quality
                if quality_score >= 0.9:
                    results["quality_distribution"]["excellent"] += 1
                elif quality_score >= 0.8:
                    results["quality_distribution"]["good"] += 1
                elif quality_score >= 0.7:
                    results["quality_distribution"]["acceptable"] += 1
                else:
                    results["quality_distribution"]["poor"] += 1

                category_samples.append({
                    "prompt": prompt,
                    "generated_text": generated_text[:100] + "...",
                    "quality_score": quality_score
                })

            results["category_results"][category] = {
                "average_score": sum(category_scores) / len(category_scores),
                "samples": category_samples
            }

            print(f"  {category}: {results['category_results'][category]['average_score']:.3f}")

        results["average_quality"] = sum(all_scores) / len(all_scores)
        return results

    def _compare_with_baseline(self, trained_model, trained_tokenizer,
                              baseline_model, baseline_tokenizer, novel_name: str) -> Dict[str, Any]:
        """Compare trained model with baseline"""

        comparison_prompts = [
            "The ancient book contained secrets that",
            "In the moonlight, the figure appeared",
            "The whispered incantation began to",
            "Deep beneath the earth, something stirred"
        ]

        results = {
            "prompts_tested": len(comparison_prompts),
            "comparisons": [],
            "trained_better_count": 0,
            "baseline_better_count": 0,
            "average_improvement": 0.0
        }

        improvements = []

        for prompt in comparison_prompts:
            # Generate from both models
            trained_text = self._generate_text(trained_model, trained_tokenizer, prompt)
            baseline_text = self._generate_text(baseline_model, baseline_tokenizer, prompt)

            # Score both
            trained_score = self.quality_validator._assess_text_quality(trained_text)
            baseline_score = self.quality_validator._assess_text_quality(baseline_text)

            improvement = trained_score - baseline_score
            improvements.append(improvement)

            if improvement > 0:
                results["trained_better_count"] += 1
            else:
                results["baseline_better_count"] += 1

            results["comparisons"].append({
                "prompt": prompt,
                "trained_score": trained_score,
                "baseline_score": baseline_score,
                "improvement": improvement,
                "trained_text_preview": trained_text[:80] + "...",
                "baseline_text_preview": baseline_text[:80] + "..."
            })

            print(f"  Prompt: {prompt[:30]}... | Trained: {trained_score:.3f} | Baseline: {baseline_score:.3f} | Change: {improvement:+.3f}")

        results["average_improvement"] = sum(improvements) / len(improvements)
        return results

    def _test_memory_system(self, model, tokenizer, novel_name: str, novel_path: Path) -> Dict[str, Any]:
        """Test episodic memory system integration"""

        try:
            # Initialize memory system
            from episodic_memory_system import MemoryConfig
            memory_config = MemoryConfig(
                max_memories_per_novel=250,
                memory_chunk_size=35,
                max_retrieved_memories=5
            )

            memory_system = EpisodicMemorySystem(memory_config)
            memory_analysis = memory_system.build_memory_for_model(novel_name, novel_path)

            # Test memory activation
            test_prompts = [
                "Tell me about the main character",
                "Describe the setting",
                "What was the atmosphere like?",
                "What emotions were present?",
                "What themes appeared in the story?"
            ]

            results = {
                "total_memories": memory_analysis.get("total_memories", 0),
                "memory_density": memory_analysis.get("memory_density", 0),
                "activation_tests": [],
                "average_memories_activated": 0.0,
                "memory_types_activated": set()
            }

            total_activated = 0

            for prompt in test_prompts:
                memories, activation = memory_system.activate_memories(novel_name, prompt)

                activated_count = len(memories)
                total_activated += activated_count

                memory_types = [m.memory_type for m in memories]
                results["memory_types_activated"].update(memory_types)

                results["activation_tests"].append({
                    "prompt": prompt,
                    "memories_activated": activated_count,
                    "memory_types": memory_types,
                    "relevance_scores": [m.importance_score for m in memories]
                })

                print(f"  '{prompt}': {activated_count} memories activated")

            results["average_memories_activated"] = total_activated / len(test_prompts)
            results["memory_types_activated"] = list(results["memory_types_activated"])

            return results

        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _test_performance(self, model, tokenizer, novel_name: str, novel_path: Path) -> Dict[str, Any]:
        """Test performance benchmarks"""

        # Test generation speed
        start_time = time.time()
        test_text = self._generate_text(model, tokenizer, "Performance test prompt")
        generation_time = time.time() - start_time

        # Test memory retrieval speed (if available)
        memory_time = 0.0
        try:
            from episodic_memory_system import EpisodicMemorySystem, MemoryConfig
            memory_system = EpisodicMemorySystem(MemoryConfig())

            start_time = time.time()
            memories, _ = memory_system.activate_memories(novel_name, "Test prompt")
            memory_time = time.time() - start_time
        except:
            memory_time = -1  # Indicates memory system not available

        results = {
            "generation_time": generation_time,
            "memory_retrieval_time": memory_time,
            "generation_speed_rating": "fast" if generation_time < 5.0 else "slow",
            "memory_speed_rating": "fast" if memory_time < 1.0 and memory_time > 0 else "slow",
            "performance_summary": {}
        }

        # Performance ratings
        if generation_time < self.config.max_generation_time:
            results["performance_summary"]["generation"] = "PASS"
        else:
            results["performance_summary"]["generation"] = "FAIL"

        if memory_time > 0 and memory_time < self.config.max_memory_retrieval_time:
            results["performance_summary"]["memory"] = "PASS"
        elif memory_time == -1:
            results["performance_summary"]["memory"] = "N/A"
        else:
            results["performance_summary"]["memory"] = "FAIL"

        print(f"  Generation time: {generation_time:.2f}s")
        print(f"  Memory retrieval time: {memory_time:.2f}s" if memory_time > 0 else "  Memory retrieval: N/A")

        return results

    def _test_novel_specific_features(self, model, tokenizer, novel_name: str) -> Dict[str, Any]:
        """Test novel-specific features based on the novel type"""

        # Customize tests based on novel
        if "cthulhu" in novel_name.lower():
            return self._test_lovecraftian_features(model, tokenizer)
        else:
            return self._test_general_fiction_features(model, tokenizer)

    def _test_lovecraftian_features(self, model, tokenizer) -> Dict[str, Any]:
        """Test Lovecraftian horror-specific features"""

        lovecraftian_prompts = [
            "The ancient tome revealed forbidden knowledge about",
            "Cosmic entities beyond human comprehension",
            "The geometry of the non-euclidean structure",
            "Madness crept into his mind as he witnessed"
        ]

        results = {
            "feature_type": "lovecraftian_horror",
            "tests": []
        }

        for prompt in lovecraftian_prompts:
            generated = self._generate_text(model, tokenizer, prompt)

            # Score for Lovecraftian elements
            lovecraft_score = self._score_lovecraftian_elements(generated)

            results["tests"].append({
                "prompt": prompt,
                "generated_text": generated[:100] + "...",
                "lovecraft_score": lovecraft_score
            })

            print(f"  Lovecraftian test: {lovecraft_score:.3f}")

        average_score = sum(test["lovecraft_score"] for test in results["tests"]) / len(results["tests"])
        results["average_lovecraft_score"] = average_score

        return results

    def _test_general_fiction_features(self, model, tokenizer) -> Dict[str, Any]:
        """Test general fiction features"""

        general_prompts = [
            "The character development showed that",
            "The plot twist revealed",
            "The setting came alive when",
            "The dialogue between characters"
        ]

        results = {
            "feature_type": "general_fiction",
            "tests": []
        }

        for prompt in general_prompts:
            generated = self._generate_text(model, tokenizer, prompt)
            quality_score = self.quality_validator._assess_text_quality(generated)

            results["tests"].append({
                "prompt": prompt,
                "generated_text": generated[:100] + "...",
                "quality_score": quality_score
            })

            print(f"  Fiction test: {quality_score:.3f}")

        average_score = sum(test["quality_score"] for test in results["tests"]) / len(results["tests"])
        results["average_fiction_score"] = average_score

        return results

    def _score_lovecraftian_elements(self, text: str) -> float:
        """Score text for Lovecraftian horror elements"""

        lovecraft_keywords = [
            "ancient", "forbidden", "cosmic", "eldritch", "unspeakable", "otherworldly",
            "madness", "horror", "tentacle", "tome", "incantation", "ritual",
            "geometry", "non-euclidean", "entity", "cyclopean", "blasphemous"
        ]

        text_lower = text.lower()
        score = 0.0

        # Check for keyword presence
        for keyword in lovecraft_keywords:
            if keyword in text_lower:
                score += 0.05

        # Check for atmospheric elements
        if any(word in text_lower for word in ["darkness", "shadow", "whisper"]):
            score += 0.1

        # Check for complexity of language
        if len(text.split()) > 20:
            score += 0.1

        return min(1.0, score)

    def _generate_text(self, model, tokenizer, prompt: str) -> str:
        """Generate text with the model"""

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.generation_length,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def _generate_overall_assessment(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of the trained model"""

        assessment = {
            "overall_rating": "unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Analyze generation quality
        if "generation_quality" in test_results:
            avg_quality = test_results["generation_quality"]["average_quality"]
            if avg_quality >= 0.9:
                assessment["strengths"].append("Excellent generation quality")
                assessment["overall_rating"] = "excellent"
            elif avg_quality >= 0.8:
                assessment["strengths"].append("Good generation quality")
                assessment["overall_rating"] = "good"
            elif avg_quality >= 0.7:
                assessment["strengths"].append("Acceptable generation quality")
                assessment["overall_rating"] = "acceptable"
            else:
                assessment["weaknesses"].append("Poor generation quality")
                assessment["overall_rating"] = "poor"

        # Analyze baseline comparison
        if "baseline_comparison" in test_results:
            improvement = test_results["baseline_comparison"]["average_improvement"]
            if improvement > 0.1:
                assessment["strengths"].append("Significant improvement over baseline")
            elif improvement > 0:
                assessment["strengths"].append("Modest improvement over baseline")
            else:
                assessment["weaknesses"].append("No improvement over baseline")
                assessment["recommendations"].append("Consider adjusting training parameters")

        # Analyze memory system
        if "memory_system" in test_results and "error" not in test_results["memory_system"]:
            avg_activated = test_results["memory_system"]["average_memories_activated"]
            if avg_activated >= 3:
                assessment["strengths"].append("Strong memory system integration")
            elif avg_activated >= 1:
                assessment["strengths"].append("Basic memory system integration")
            else:
                assessment["weaknesses"].append("Weak memory activation")

        # Performance analysis
        if "performance" in test_results:
            perf = test_results["performance"]["performance_summary"]
            if perf.get("generation") == "PASS":
                assessment["strengths"].append("Fast generation performance")
            else:
                assessment["weaknesses"].append("Slow generation performance")

        return assessment

    def _print_test_summary(self, test_results: Dict[str, Any]):
        """Print comprehensive test summary"""

        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")

        # Overall Assessment
        if "overall_assessment" in test_results:
            assessment = test_results["overall_assessment"]
            print(f"Overall Rating: {assessment['overall_rating'].upper()}")

            if assessment["strengths"]:
                print(f"\nStrengths:")
                for strength in assessment["strengths"]:
                    print(f"  + {strength}")

            if assessment["weaknesses"]:
                print(f"\nWeaknesses:")
                for weakness in assessment["weaknesses"]:
                    print(f"  - {weakness}")

            if assessment["recommendations"]:
                print(f"\nRecommendations:")
                for rec in assessment["recommendations"]:
                    print(f"  * {rec}")

        # Key Metrics
        print(f"\nKey Metrics:")
        if "generation_quality" in test_results:
            print(f"  Average Generation Quality: {test_results['generation_quality']['average_quality']:.3f}")

        if "baseline_comparison" in test_results:
            improvement = test_results["baseline_comparison"]["average_improvement"]
            print(f"  Improvement over Baseline: {improvement:+.3f}")

        if "memory_system" in test_results and "error" not in test_results["memory_system"]:
            memories = test_results["memory_system"]["total_memories"]
            activated = test_results["memory_system"]["average_memories_activated"]
            print(f"  Total Memories Created: {memories}")
            print(f"  Average Memories Activated: {activated:.1f}")

        if "performance" in test_results:
            gen_time = test_results["performance"]["generation_time"]
            print(f"  Generation Time: {gen_time:.2f}s")

def main():
    """Test the post-training tester"""
    print("Post-Training Test Suite")
    print("=" * 40)

    # This would be called after training completes
    # tester = PostTrainingTester()
    # results = tester.run_comprehensive_tests(
    #     model_path="iterative_models/call_of_cthulhu/final/",
    #     novel_name="call_of_cthulhu",
    #     novel_path=Path("novels/call_of_cthulhu")
    # )

    print("Post-training test suite ready for integration")

if __name__ == "__main__":
    main()