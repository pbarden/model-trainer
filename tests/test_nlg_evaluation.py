#!/usr/bin/env python3
"""
Tests for NLG evaluation metrics (BLEU, ROUGE, Perplexity)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_tea.analysis.metrics.nlg_evaluation import (
    BLEUEvaluator, ROUGEEvaluator, PerplexityCalculator,
    BenchmarkComparator, HumanEvaluationFramework, NLGEvaluator
)


class TestBLEUEvaluator:
    """Test BLEU score calculations"""

    def setup_method(self):
        self.evaluator = BLEUEvaluator()

    def test_bleu_perfect_match(self):
        """Test BLEU with perfect matches"""
        generated = ["The cat sat on the mat"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_bleu_scores(generated, reference)

        # Perfect match should give high BLEU scores
        assert scores["BLEU-1"] >= 0.9
        assert scores["BLEU-2"] >= 0.9
        assert scores["BLEU-3"] >= 0.9
        assert scores["BLEU-4"] >= 0.9

    def test_bleu_no_match(self):
        """Test BLEU with no matches"""
        generated = ["completely different text here"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_bleu_scores(generated, reference)

        # No overlap should give very low scores
        assert scores["BLEU-1"] <= 0.1
        assert scores["BLEU-2"] <= 0.1

    def test_bleu_partial_match(self):
        """Test BLEU with partial matches"""
        generated = ["The cat was sitting on the mat"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_bleu_scores(generated, reference)

        # Partial match should give moderate scores
        assert 0.3 <= scores["BLEU-1"] <= 0.9
        assert 0.1 <= scores["BLEU-2"] <= 0.8

    def test_bleu_multiple_samples(self):
        """Test BLEU with multiple samples"""
        generated = [
            "The cat sat on the mat",
            "A dog ran in the park",
            "Birds fly in the sky"
        ]
        reference = [
            "The cat was on the mat",
            "A dog runs in the park",
            "Birds are flying in the sky"
        ]

        scores = self.evaluator.calculate_bleu_scores(generated, reference)

        # Should handle multiple samples
        assert all(0 <= score <= 1 for score in scores.values())
        assert len(scores) == 4  # BLEU-1 through BLEU-4


class TestROUGEEvaluator:
    """Test ROUGE score calculations"""

    def setup_method(self):
        self.evaluator = ROUGEEvaluator()

    def test_rouge_perfect_match(self):
        """Test ROUGE with perfect matches"""
        generated = ["The cat sat on the mat"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_rouge_scores(generated, reference)

        # Perfect match should give high ROUGE scores
        assert scores["ROUGE-1"] == 1.0
        assert scores["ROUGE-2"] == 1.0
        assert scores["ROUGE-L"] == 1.0

    def test_rouge_no_match(self):
        """Test ROUGE with no matches"""
        generated = ["completely different text here"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_rouge_scores(generated, reference)

        # No overlap should give zero scores
        assert scores["ROUGE-1"] == 0.0
        assert scores["ROUGE-2"] == 0.0
        assert scores["ROUGE-L"] == 0.0

    def test_rouge_partial_match(self):
        """Test ROUGE with partial matches"""
        generated = ["The cat was sitting on the mat"]
        reference = ["The cat sat on the mat"]

        scores = self.evaluator.calculate_rouge_scores(generated, reference)

        # Partial match should give moderate scores
        assert 0.3 <= scores["ROUGE-1"] <= 0.9
        assert 0.0 <= scores["ROUGE-2"] <= 0.8
        assert 0.3 <= scores["ROUGE-L"] <= 0.9


class TestPerplexityCalculator:
    """Test perplexity calculations"""

    def test_perplexity_from_log_probs(self):
        """Test perplexity calculation from log probabilities"""
        # High probability (low perplexity)
        high_prob_logs = [-0.1, -0.2, -0.15, -0.1]
        perplexity = PerplexityCalculator.calculate_perplexity(high_prob_logs)
        assert perplexity < 2.0

        # Low probability (high perplexity)
        low_prob_logs = [-2.0, -2.5, -2.3, -2.1]
        perplexity = PerplexityCalculator.calculate_perplexity(low_prob_logs)
        assert perplexity > 5.0

    def test_perplexity_from_loss(self):
        """Test perplexity calculation from cross-entropy loss"""
        # Low loss should give low perplexity
        low_loss = 0.5
        perplexity = PerplexityCalculator.calculate_perplexity_from_loss(low_loss)
        assert perplexity < 2.0

        # High loss should give high perplexity
        high_loss = 3.0
        perplexity = PerplexityCalculator.calculate_perplexity_from_loss(high_loss)
        assert perplexity > 10.0

    def test_perplexity_empty_input(self):
        """Test perplexity with empty input"""
        perplexity = PerplexityCalculator.calculate_perplexity([])
        assert perplexity == float('inf')


class TestBenchmarkComparator:
    """Test benchmark comparison functionality"""

    def setup_method(self):
        self.comparator = BenchmarkComparator()

    def test_wikitext_benchmark_comparison(self):
        """Test comparison against WikiText benchmark"""
        model_perplexity = 20.0
        comparison = self.comparator.perplexity_benchmark_comparison(
            model_perplexity, "wikitext_103"
        )

        assert comparison["comparison_available"] == True
        assert comparison["model_perplexity"] == 20.0
        assert "transformer_base_benchmark" in comparison
        assert "vs_transformer_base_ratio" in comparison

    def test_unknown_benchmark(self):
        """Test handling of unknown benchmark"""
        model_perplexity = 20.0
        comparison = self.comparator.perplexity_benchmark_comparison(
            model_perplexity, "unknown_dataset"
        )

        assert comparison["comparison_available"] == False


class TestHumanEvaluationFramework:
    """Test human evaluation framework"""

    def setup_method(self):
        self.framework = HumanEvaluationFramework()

    def test_evaluation_template(self):
        """Test generation of evaluation template"""
        samples = ["Sample 1", "Sample 2", "Sample 3"]
        template = self.framework.human_evaluation_framework(samples)

        assert "evaluation_template" in template
        assert template["evaluation_template"]["num_samples"] == 3
        assert "criteria" in template["evaluation_template"]

    def test_score_aggregation(self):
        """Test aggregation of human scores"""
        samples = ["Sample 1", "Sample 2"]
        scores = [
            {"fluency": 4, "coherence": 3, "relevance": 4},
            {"fluency": 5, "coherence": 4, "relevance": 5}
        ]

        results = self.framework.human_evaluation_framework(samples, scores)

        assert "fluency_mean" in results
        assert "coherence_mean" in results
        assert "overall_quality" in results
        assert 3.0 <= results["overall_quality"] <= 5.0


class TestNLGEvaluator:
    """Test comprehensive NLG evaluator"""

    def setup_method(self):
        self.evaluator = NLGEvaluator()

    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation with all metrics"""
        generated = ["The cat sat on the mat", "A story of adventure"]
        reference = ["The cat was on the mat", "An adventure story"]
        log_probs = [-1.2, -1.5, -1.3, -1.1, -1.4, -1.2]

        results = self.evaluator.comprehensive_evaluation(
            generated, reference, log_probs
        )

        # Check all components are present
        assert results.bleu_scores is not None
        assert results.rouge_scores is not None
        assert results.perplexity > 0
        assert len(results.bleu_scores) == 4
        assert len(results.rouge_scores) == 3

    def test_quick_evaluation(self):
        """Test quick evaluation for single texts"""
        generated = "The cat sat on the mat"
        reference = "The cat was on the mat"

        results = self.evaluator.quick_evaluation(generated, reference)

        # Should have both BLEU and ROUGE scores
        assert "BLEU-1" in results
        assert "ROUGE-1" in results
        assert all(0 <= score <= 1 for score in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])