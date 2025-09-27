#!/usr/bin/env python3
"""
Model Tea - Memory Enhanced Model Integration
Copyright © ChaiQ LLC

Implements the memory-enhanced generation flow:
Prompt > Model > Memory(ies) > Model > Output

This creates a volley system where the model generates an initial response,
which triggers memory activation, and then the model refines its output
using the activated memories - mimicking human memory-assisted thinking.
"""

import os
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from episodic_memory_system import EpisodicMemorySystem, MemoryConfig, Memory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEnhancedConfig:
    """Configuration for memory-enhanced model"""
    # Model settings
    model_temperature: float = 0.8
    max_initial_tokens: int = 50  # Initial model response before memory
    max_final_tokens: int = 150   # Final enhanced response

    # Memory integration
    memory_weight: float = 0.3    # How much to weight memory vs model
    memory_integration_style: str = "contextual"  # "contextual" or "direct"

    # Volley settings
    enable_memory_volley: bool = True
    max_memory_context_length: int = 200  # Max chars from each memory

    # Measurement settings
    enable_behavior_measurement: bool = True
    compare_with_baseline: bool = True

class MemoryBehaviorMeasurement:
    """Measures how memory affects model behavior"""

    def __init__(self):
        self.measurements = {
            "coherence_scores": [],
            "creativity_scores": [],
            "relevance_scores": [],
            "memory_utilization": [],
            "response_length_changes": [],
            "vocabulary_diversity": [],
            "emotional_consistency": []
        }

    def measure_response_differences(self,
                                   baseline_response: str,
                                   memory_enhanced_response: str,
                                   activated_memories: List[Memory],
                                   prompt: str) -> Dict[str, Any]:
        """Measure differences between baseline and memory-enhanced responses"""

        # Length comparison
        baseline_words = len(baseline_response.split())
        enhanced_words = len(memory_enhanced_response.split())
        length_change = (enhanced_words - baseline_words) / baseline_words if baseline_words > 0 else 0

        # Vocabulary diversity (unique words ratio)
        baseline_vocab = set(baseline_response.lower().split())
        enhanced_vocab = set(memory_enhanced_response.lower().split())

        baseline_diversity = len(baseline_vocab) / baseline_words if baseline_words > 0 else 0
        enhanced_diversity = len(enhanced_vocab) / enhanced_words if enhanced_words > 0 else 0

        # Memory utilization
        memory_words_used = 0
        for memory in activated_memories:
            memory_vocab = set(memory.content.lower().split())
            overlap = enhanced_vocab.intersection(memory_vocab)
            memory_words_used += len(overlap)

        memory_utilization = memory_words_used / enhanced_words if enhanced_words > 0 else 0

        # Coherence (simple heuristic - sentence structure)
        baseline_sentences = len([s for s in baseline_response.split('.') if s.strip()])
        enhanced_sentences = len([s for s in memory_enhanced_response.split('.') if s.strip()])

        coherence_score = enhanced_sentences / baseline_sentences if baseline_sentences > 0 else 1.0

        # Creativity (presence of novel combinations)
        baseline_bigrams = set(self._extract_bigrams(baseline_response))
        enhanced_bigrams = set(self._extract_bigrams(memory_enhanced_response))
        novel_bigrams = enhanced_bigrams - baseline_bigrams
        creativity_score = len(novel_bigrams) / len(enhanced_bigrams) if enhanced_bigrams else 0

        # Relevance to prompt
        prompt_words = set(prompt.lower().split())
        baseline_relevance = len(baseline_vocab.intersection(prompt_words)) / len(prompt_words) if prompt_words else 0
        enhanced_relevance = len(enhanced_vocab.intersection(prompt_words)) / len(prompt_words) if prompt_words else 0

        measurement = {
            "length_change_ratio": length_change,
            "vocabulary_diversity_change": enhanced_diversity - baseline_diversity,
            "memory_utilization_score": memory_utilization,
            "coherence_ratio": coherence_score,
            "creativity_score": creativity_score,
            "relevance_improvement": enhanced_relevance - baseline_relevance,
            "memories_activated": len(activated_memories),
            "memory_types_used": list(set(m.memory_type for m in activated_memories)),
            "emotional_tones_activated": list(set(m.emotional_tone for m in activated_memories))
        }

        # Store measurements
        self.measurements["coherence_scores"].append(coherence_score)
        self.measurements["creativity_scores"].append(creativity_score)
        self.measurements["relevance_scores"].append(enhanced_relevance)
        self.measurements["memory_utilization"].append(memory_utilization)
        self.measurements["response_length_changes"].append(length_change)
        self.measurements["vocabulary_diversity"].append(enhanced_diversity - baseline_diversity)

        return measurement

    def _extract_bigrams(self, text: str) -> List[str]:
        """Extract word bigrams from text"""
        words = text.lower().split()
        return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all measurements"""
        import statistics

        summary = {}
        for metric, values in self.measurements.items():
            if values:
                summary[metric] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                summary[metric] = {"count": 0}

        return summary

class MemoryEnhancedModel:
    """Model Tea memory-enhanced model with volley generation"""

    def __init__(self, model_name: str, config: MemoryEnhancedConfig = None):
        self.model_name = model_name
        self.config = config or MemoryEnhancedConfig()

        # Initialize memory system
        memory_config = MemoryConfig()
        self.memory_system = EpisodicMemorySystem(memory_config)

        # Load model memories if available
        if not self.memory_system.load_memory_for_model(model_name):
            logger.warning(f"No memories found for {model_name}")

        # Initialize behavior measurement
        self.behavior_measurement = MemoryBehaviorMeasurement() if self.config.enable_behavior_measurement else None

        # Model will be loaded when needed
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_path = Path("iterative_models") / self.model_name / "final"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loaded model for {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_with_memory_volley(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response using memory volley:
        Prompt > Model > Memory(ies) > Model > Output
        """

        if self.model is None:
            self.load_model()

        start_time = time.time()

        # Step 1: Initial model response (Prompt > Model)
        baseline_response = self._generate_baseline_response(prompt)

        if not self.config.enable_memory_volley:
            return {
                "response": baseline_response,
                "baseline_response": baseline_response,
                "activated_memories": [],
                "memory_analysis": {},
                "behavior_measurement": {},
                "generation_time": time.time() - start_time
            }

        # Step 2: Memory activation (Model > Memory(ies))
        # Use both prompt and initial response to activate memories
        memory_trigger = f"{prompt} {baseline_response}"
        activated_memories, memory_analysis = self.memory_system.activate_memories(
            self.model_name, memory_trigger
        )

        # Step 3: Memory-enhanced generation (Memory(ies) > Model > Output)
        if activated_memories:
            enhanced_response = self._generate_memory_enhanced_response(
                prompt, baseline_response, activated_memories
            )
        else:
            enhanced_response = baseline_response

        # Step 4: Measure behavioral differences
        behavior_measurement = {}
        if self.behavior_measurement and activated_memories:
            behavior_measurement = self.behavior_measurement.measure_response_differences(
                baseline_response, enhanced_response, activated_memories, prompt
            )

        generation_time = time.time() - start_time

        return {
            "response": enhanced_response,
            "baseline_response": baseline_response,
            "activated_memories": [self._memory_to_dict(m) for m in activated_memories],
            "memory_analysis": memory_analysis,
            "behavior_measurement": behavior_measurement,
            "generation_time": generation_time,
            "volley_enabled": True
        }

    def _generate_baseline_response(self, prompt: str) -> str:
        """Generate initial response without memory"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.max_initial_tokens,
                temperature=self.config.model_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt
        response = response[len(prompt):].strip()

        return response

    def _generate_memory_enhanced_response(self,
                                         prompt: str,
                                         baseline_response: str,
                                         memories: List[Memory]) -> str:
        """Generate enhanced response using activated memories"""

        # Create memory context
        memory_context = self._create_memory_context(memories)

        # Create enhanced prompt
        if self.config.memory_integration_style == "contextual":
            enhanced_prompt = f"""Context from memories: {memory_context}

Original request: {prompt}

Initial thoughts: {baseline_response}

Enhanced response:"""
        else:  # direct integration
            enhanced_prompt = f"{prompt}\n\nDrawing from: {memory_context}\n\nResponse:"

        # Generate enhanced response
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.max_final_tokens,
                temperature=self.config.model_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        enhanced_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new response
        enhanced_response = enhanced_response[len(enhanced_prompt):].strip()

        return enhanced_response

    def _create_memory_context(self, memories: List[Memory]) -> str:
        """Create context string from activated memories"""
        memory_snippets = []

        for memory in memories:
            # Truncate memory content if needed
            content = memory.content[:self.config.max_memory_context_length]
            if len(memory.content) > self.config.max_memory_context_length:
                content += "..."

            memory_snippets.append(f"[{memory.memory_type}] {content}")

        return " | ".join(memory_snippets)

    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory object to dictionary for JSON serialization"""
        return {
            "content": memory.content[:100] + ("..." if len(memory.content) > 100 else ""),
            "memory_type": memory.memory_type,
            "keywords": memory.keywords,
            "emotional_tone": memory.emotional_tone,
            "importance_score": memory.importance_score,
            "chapter_position": memory.chapter_position
        }

    def get_behavior_summary(self) -> Dict[str, Any]:
        """Get summary of behavioral measurements"""
        if not self.behavior_measurement:
            return {"error": "Behavior measurement not enabled"}

        return self.behavior_measurement.get_summary_statistics()

def main():
    """Test the memory-enhanced model"""
    print("Model Tea - Memory Enhanced Model Test")
    print("=" * 40)

    # Test with agony column if available
    model_name = "agony_column"

    try:
        config = MemoryEnhancedConfig()
        enhanced_model = MemoryEnhancedModel(model_name, config)

        test_prompts = [
            "Write about a mysterious character",
            "Describe a tense scene",
            "Tell me about love and betrayal",
            "What happens during a confrontation?"
        ]

        for prompt in test_prompts:
            print(f"\nTesting: '{prompt}'")
            result = enhanced_model.generate_with_memory_volley(prompt)

            print(f"Baseline: {result['baseline_response'][:100]}...")
            print(f"Enhanced: {result['response'][:100]}...")
            print(f"Memories activated: {len(result['activated_memories'])}")

            if result['behavior_measurement']:
                print(f"Memory utilization: {result['behavior_measurement']['memory_utilization_score']:.2f}")
                print(f"Creativity score: {result['behavior_measurement']['creativity_score']:.2f}")

        # Show behavior summary
        summary = enhanced_model.get_behavior_summary()
        print(f"\nBehavior Summary:")
        for metric, stats in summary.items():
            if 'mean' in stats:
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['stdev']:.3f}")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()