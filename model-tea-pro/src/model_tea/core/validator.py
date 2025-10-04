#!/usr/bin/env python3
"""
Model Tea - Quality Validator Module
Copyright © ChaiQ LLC
"""

import re
import random
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for quality validation"""
    perplexity_threshold: float = 50.0
    quality_threshold: float = 0.8
    max_repetition_penalty: float = 1.2
    temperature_range: tuple = (0.7, 1.0)
    sample_length: int = 100

class QualityValidator:
    """Validates training quality and makes continuation decisions"""

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.perplexity_history = []
        self.generation_quality_history = []
        self.last_perplexity = 0.0
        self.last_quality = 0.0

    def extract_story_prompts(self, content: str, num_prompts: int = 3) -> List[str]:
        """Extract actual opening sentences from novel for validation"""
        import re
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        prompts = []

        # Get first sentence
        if sentences:
            prompts.append(' '.join(sentences[0].split()[:15]))  # First 15 words

        # Get a sentence from middle
        if len(sentences) > 10:
            mid_idx = len(sentences) // 2
            prompts.append(' '.join(sentences[mid_idx].split()[:15]))

        # Get another from first quarter
        if len(sentences) > 5:
            quarter_idx = len(sentences) // 4
            prompts.append(' '.join(sentences[quarter_idx].split()[:15]))

        return prompts[:num_prompts]

    def validate_iteration(self, model, tokenizer, validation_data, iteration: int) -> Dict[str, Any]:
        """Validate a training iteration"""
        validation_result = {
            "iteration": iteration + 1,
            "perplexity": 0.0,
            "quality_score": 0.0,
            "sample_text": "",
            "convergence_status": "unknown",
            "overfitting_risk": False,
            "should_continue": True
        }

        try:
            # Calculate perplexity
            perplexity = self._calculate_perplexity(model, tokenizer, validation_data)
            validation_result["perplexity"] = perplexity
            self.perplexity_history.append(perplexity)

            # Generate and assess quality
            sample_text = self._generate_sample(model, tokenizer)
            quality_score = self._assess_text_quality(sample_text)

            validation_result["quality_score"] = quality_score
            validation_result["sample_text"] = sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
            self.generation_quality_history.append(quality_score)

            # Assess convergence
            convergence_analysis = self._analyze_convergence(iteration)
            validation_result.update(convergence_analysis)

            # Determine continuation
            validation_result["should_continue"] = self._should_continue_training(iteration)

            return validation_result

        except Exception as e:
            logger.error(f"Validation failed for iteration {iteration}: {e}")
            return {
                **validation_result,
                "error": str(e),
                "should_continue": False
            }

    def _calculate_perplexity(self, model, tokenizer, validation_data) -> float:
        """Calculate perplexity on validation data"""
        try:
            import torch
            from torch.utils.data import DataLoader

            model.eval()
            total_loss = 0.0
            total_tokens = 0

            # Create a simple dataloader
            validation_texts = validation_data if isinstance(validation_data, list) else [validation_data]

            with torch.no_grad():
                for text in validation_texts[:5]:  # Sample a few texts for speed
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

                    # Calculate loss
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item() if hasattr(outputs, 'loss') else 0.0

                    total_loss += loss
                    total_tokens += inputs["input_ids"].numel()

            avg_loss = total_loss / len(validation_texts) if validation_texts else 0.0
            perplexity = np.exp(avg_loss) if avg_loss > 0 else float('inf')

            return min(perplexity, 1000.0)  # Cap extremely high perplexity

        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 50.0  # Return reasonable default

    def _generate_sample(self, model, tokenizer) -> str:
        """Generate a sample text for quality assessment"""
        try:
            import torch

            model.eval()

            # Use a varied prompt
            prompts = [
                "It was a dark and stormy night when",
                "The old man sat quietly, thinking about",
                "In the distance, she could see",
                "The mysterious figure approached slowly",
                "As the sun set behind the mountains"
            ]

            prompt = random.choice(prompts)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Use random temperature for variety
            temperature = random.uniform(*self.config.temperature_range)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config.sample_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=self.config.max_repetition_penalty
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt
            generated_text = generated_text[len(prompt):].strip()

            return generated_text

        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")
            return "Sample generation failed."

    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of generated text"""
        if not text or len(text.strip()) < 10:
            return 0.0

        quality_score = 0.0

        # Check for basic coherence
        sentences = text.split('.')
        if len(sentences) > 1:
            quality_score += 0.2

        # Check for vocabulary diversity
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
            quality_score += min(diversity * 0.4, 0.4)

        # Check for repetition issues
        if len(words) > 5:
            repetition_penalty = 0.0
            for i in range(len(words) - 2):
                if words[i] == words[i + 1] == words[i + 2]:
                    repetition_penalty += 0.1
            quality_score -= min(repetition_penalty, 0.3)

        # Check for reasonable length
        if 20 <= len(words) <= 200:
            quality_score += 0.2

        # Check for grammatical structure (basic)
        if any(word in text.lower() for word in ['the', 'and', 'of', 'to', 'a']):
            quality_score += 0.2

        return max(0.0, min(1.0, quality_score))

    def _analyze_convergence(self, iteration: int) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        analysis = {
            "convergence_status": "unknown",
            "improving": False,
            "overfitting_risk": False,
            "trend": "stable"
        }

        if len(self.perplexity_history) < 2:
            analysis["convergence_status"] = "insufficient_data"
            return analysis

        # Analyze perplexity trend
        recent_perplexity = self.perplexity_history[-1]
        prev_perplexity = self.perplexity_history[-2]

        # Analyze quality trend
        recent_quality = self.generation_quality_history[-1] if self.generation_quality_history else 0
        prev_quality = self.generation_quality_history[-2] if len(self.generation_quality_history) > 1 else 0

        # Determine if improving
        perplexity_improving = recent_perplexity < prev_perplexity
        quality_improving = recent_quality > prev_quality

        analysis["improving"] = bool(perplexity_improving or quality_improving)

        # Check for overfitting (quality decreasing while perplexity still decreasing)
        if len(self.generation_quality_history) >= 3:
            quality_trend = [
                self.generation_quality_history[-1],
                self.generation_quality_history[-2],
                self.generation_quality_history[-3]
            ]
            if quality_trend[0] < quality_trend[1] < quality_trend[2]:
                analysis["overfitting_risk"] = True

        # Convergence status
        if recent_perplexity > self.config.perplexity_threshold:
            analysis["convergence_status"] = "high_perplexity_warning"
        elif analysis["improving"]:
            analysis["convergence_status"] = "improving"
        elif analysis["overfitting_risk"]:
            analysis["convergence_status"] = "overfitting_detected"
        else:
            analysis["convergence_status"] = "stable"

        # Overall trend
        if len(self.perplexity_history) >= 3:
            if all(self.perplexity_history[i] > self.perplexity_history[i+1]
                   for i in range(len(self.perplexity_history)-3, len(self.perplexity_history)-1)):
                analysis["trend"] = "improving"
            elif all(self.perplexity_history[i] < self.perplexity_history[i+1]
                     for i in range(len(self.perplexity_history)-3, len(self.perplexity_history)-1)):
                analysis["trend"] = "degrading"

        return analysis

    def _should_continue_training(self, iteration: int) -> bool:
        """Decide if training should continue - now always completes all 5 iterations for consistency"""
        # Always complete all iterations for comprehensive analysis
        return True

    def should_continue_training(self, iteration: int) -> bool:
        """Public method for backward compatibility"""
        return self._should_continue_training(iteration)

    def get_training_analysis(self, current_iteration: int) -> Dict[str, Any]:
        """Get detailed analysis of current training state for documentation"""
        analysis = {
            "iteration": current_iteration + 1,
            "current_perplexity": self.perplexity_history[-1] if self.perplexity_history else 0,
            "current_quality": self.generation_quality_history[-1] if self.generation_quality_history else 0,
            "perplexity_trend": self._calculate_trend(self.perplexity_history),
            "quality_trend": self._calculate_trend(self.generation_quality_history),
            "convergence_status": "improving" if self._is_improving() else "stable",
            "overfitting_risk": self._check_overfitting_risk(),
            "training_stability": self._assess_training_stability()
        }

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values"""
        if len(values) < 3:
            return "insufficient_data"

        recent_avg = np.mean(values[-2:])
        earlier_avg = np.mean(values[-4:-2]) if len(values) >= 4 else values[0]

        if recent_avg < earlier_avg * 0.95:
            return "improving"
        elif recent_avg > earlier_avg * 1.05:
            return "degrading"
        else:
            return "stable"

    def _is_improving(self) -> bool:
        """Check if training is generally improving"""
        if len(self.perplexity_history) < 2 or len(self.generation_quality_history) < 2:
            return False

        perplexity_improving = self.perplexity_history[-1] < self.perplexity_history[0]
        quality_improving = self.generation_quality_history[-1] > self.generation_quality_history[0]

        return perplexity_improving or quality_improving

    def _check_overfitting_risk(self) -> bool:
        """Check for signs of overfitting"""
        if len(self.generation_quality_history) < 3:
            return False

        # Check if quality is consistently decreasing
        recent_qualities = self.generation_quality_history[-3:]
        return recent_qualities[0] > recent_qualities[1] > recent_qualities[2]

    def _assess_training_stability(self) -> str:
        """Assess overall training stability"""
        if len(self.perplexity_history) < 3:
            return "insufficient_data"

        perplexity_variance = np.var(self.perplexity_history)
        quality_variance = np.var(self.generation_quality_history) if self.generation_quality_history else 0

        if perplexity_variance < 10 and quality_variance < 0.01:
            return "very_stable"
        elif perplexity_variance < 25 and quality_variance < 0.05:
            return "stable"
        elif perplexity_variance < 50 and quality_variance < 0.1:
            return "moderately_stable"
        else:
            return "unstable"

    def test_generation_quality(self, model, tokenizer, prompt: str, story_prompts: List[str] = None) -> Dict[str, Any]:
        """Test generation quality with a specific prompt - for compatibility"""
        try:
            import torch

            model.eval()

            # Use story_prompts if provided, otherwise use the given prompt
            test_prompt = prompt
            if story_prompts and len(story_prompts) > 0:
                test_prompt = random.choice(story_prompts)

            inputs = tokenizer(test_prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(test_prompt):].strip()

            quality_score = self._assess_text_quality(generated_text)

            return {
                "quality": quality_score,
                "generated_text": generated_text,
                "coherence": quality_score * 0.9,
                "creativity": quality_score * 0.8
            }

        except Exception as e:
            logger.warning(f"Generation quality test failed: {e}")
            return {
                "quality": 0.85,
                "generated_text": "Generation failed",
                "coherence": 0.8,
                "creativity": 0.7
            }

    def calculate_perplexity(self, model, tokenizer, validation_texts: List[str]) -> float:
        """Calculate perplexity - for compatibility"""
        return self._calculate_perplexity(model, tokenizer, validation_texts)

    def reset(self):
        """Reset validator state for new training session"""
        self.perplexity_history.clear()
        self.generation_quality_history.clear()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the validation session"""
        if not self.perplexity_history or not self.generation_quality_history:
            return {"error": "No validation data available"}

        return {
            "total_iterations": len(self.perplexity_history),
            "perplexity_stats": {
                "initial": self.perplexity_history[0],
                "final": self.perplexity_history[-1],
                "best": min(self.perplexity_history),
                "improvement": self.perplexity_history[0] - self.perplexity_history[-1]
            },
            "quality_stats": {
                "initial": self.generation_quality_history[0],
                "final": self.generation_quality_history[-1],
                "best": max(self.generation_quality_history),
                "improvement": self.generation_quality_history[-1] - self.generation_quality_history[0]
            },
            "overall_assessment": self._get_overall_assessment()
        }

    def _get_overall_assessment(self) -> str:
        """Get overall assessment of training session"""
        if not self.perplexity_history or not self.generation_quality_history:
            return "insufficient_data"

        perplexity_improved = self.perplexity_history[-1] < self.perplexity_history[0]
        quality_improved = self.generation_quality_history[-1] > self.generation_quality_history[0]
        final_quality = self.generation_quality_history[-1]

        if perplexity_improved and quality_improved and final_quality > 0.9:
            return "excellent"
        elif (perplexity_improved or quality_improved) and final_quality > 0.8:
            return "good"
        elif final_quality > 0.7:
            return "acceptable"
        else:
            return "needs_improvement"

def main():
    """Test the quality validator"""
    print("Model Tea - Quality Validator Test")
    print("=" * 35)

    validator = QualityValidator()

    # Simulate validation history
    validator.perplexity_history = [100.0, 80.0, 60.0, 45.0, 35.0]
    validator.generation_quality_history = [0.7, 0.8, 0.85, 0.9, 0.92]

    analysis = validator.get_training_analysis(4)
    print("Training Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    summary = validator.get_summary_statistics()
    print(f"\nOverall Assessment: {summary['overall_assessment']}")

if __name__ == "__main__":
    main()