#!/usr/bin/env python3
"""
Tests for statistical validation and significance testing
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_tea.analysis.metrics.statistical_validation import (
    StatisticalValidator, CorrectionMethod, ConfidenceInterval,
    SignificanceTestResult, EffectSizeResult
)


class TestStatisticalValidator:
    """Test statistical validation functionality"""

    def setup_method(self):
        self.validator = StatisticalValidator(random_state=42)

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation"""
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, 100)

        ci = self.validator.bootstrap_confidence_intervals(
            data.tolist(), confidence_level=0.95
        )

        # Check that CI is reasonable for normal distribution
        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert ci.lower < np.mean(data) < ci.upper
        assert ci.upper - ci.lower > 0  # Non-zero width

    def test_bootstrap_insufficient_data(self):
        """Test bootstrap with insufficient data"""
        data = [5.0]  # Only one sample

        ci = self.validator.bootstrap_confidence_intervals(data)

        assert ci.method == "insufficient_data"
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_paired_t_test_significant_improvement(self):
        """Test paired t-test with significant improvement"""
        # Create data with clear improvement
        baseline = [0.70, 0.72, 0.71, 0.73, 0.69, 0.74, 0.70, 0.72]
        enhanced = [0.78, 0.80, 0.79, 0.81, 0.77, 0.82, 0.78, 0.80]

        result = self.validator.paired_t_test_significance(baseline, enhanced)

        assert isinstance(result, SignificanceTestResult)
        assert result.p_value < 0.05  # Should be significant
        assert "improvement" in result.interpretation.lower()
        assert result.effect_size > 0  # Positive effect

    def test_paired_t_test_no_difference(self):
        """Test paired t-test with no significant difference"""
        # Create data with no real difference
        baseline = [0.75, 0.74, 0.76, 0.75, 0.74, 0.76, 0.75, 0.74]
        enhanced = [0.74, 0.75, 0.75, 0.76, 0.75, 0.75, 0.74, 0.75]

        result = self.validator.paired_t_test_significance(baseline, enhanced)

        assert result.p_value > 0.05  # Should not be significant
        assert "no significant" in result.interpretation.lower()

    def test_multiple_comparison_bonferroni(self):
        """Test Bonferroni multiple comparison correction"""
        p_values = [0.01, 0.03, 0.04, 0.06, 0.08]

        result = self.validator.multiple_comparison_correction(
            p_values, CorrectionMethod.BONFERRONI
        )

        assert result["method"] == "bonferroni"
        assert result["number_of_tests"] == 5
        assert len(result["corrected_p_values"]) == 5
        assert len(result["rejections"]) == 5

        # Bonferroni should be more conservative
        original_significant = sum(1 for p in p_values if p < 0.05)
        corrected_significant = result["significant_tests"]
        assert corrected_significant <= original_significant

    def test_multiple_comparison_benjamini_hochberg(self):
        """Test Benjamini-Hochberg FDR correction"""
        p_values = [0.001, 0.01, 0.03, 0.06, 0.08]

        result = self.validator.multiple_comparison_correction(
            p_values, CorrectionMethod.BENJAMINI_HOCHBERG
        )

        assert result["method"] == "benjamini_hochberg"
        assert result["number_of_tests"] == 5
        # BH should be less conservative than Bonferroni
        assert result["significant_tests"] >= 0

    def test_effect_size_analysis_paired(self):
        """Test effect size analysis for paired samples"""
        baseline = [0.70, 0.72, 0.71, 0.73, 0.69]
        treatment = [0.78, 0.80, 0.79, 0.81, 0.77]

        result = self.validator.effect_size_analysis(baseline, treatment, paired=True)

        assert isinstance(result, EffectSizeResult)
        assert result.cohens_d > 0  # Positive effect
        assert result.magnitude in ["negligible", "small", "medium", "large"]
        assert "positive" in result.interpretation

    def test_effect_size_analysis_independent(self):
        """Test effect size analysis for independent samples"""
        baseline = [0.70, 0.72, 0.71, 0.73, 0.69, 0.74]
        treatment = [0.78, 0.80, 0.79, 0.81, 0.77]

        result = self.validator.effect_size_analysis(baseline, treatment, paired=False)

        assert isinstance(result, EffectSizeResult)
        assert result.glass_delta is not None  # Should be calculated for independent samples
        assert result.cohens_d > 0

    def test_power_analysis_given_sample_size(self):
        """Test power analysis given sample size"""
        result = self.validator.power_analysis(
            effect_size=0.5,
            sample_size=20,
            alpha=0.05,
            test_type="paired"
        )

        assert "power" in result
        assert 0 <= result["power"] <= 1
        assert result["effect_size"] == 0.5
        assert result["sample_size"] == 20

    def test_power_analysis_required_sample_size(self):
        """Test power analysis to find required sample size"""
        result = self.validator.power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            test_type="paired"
        )

        assert "required_sample_size" in result
        assert result["required_sample_size"] > 0
        assert result["desired_power"] == 0.8

    def test_comprehensive_comparison(self):
        """Test comprehensive statistical comparison"""
        # Create baseline and enhanced metrics
        baseline_metrics = {
            "bleu_score": [0.72, 0.74, 0.73, 0.75, 0.71],
            "rouge_score": [0.68, 0.70, 0.69, 0.71, 0.67],
            "quality_score": [0.78, 0.80, 0.79, 0.81, 0.77]
        }

        enhanced_metrics = {
            "bleu_score": [0.79, 0.81, 0.80, 0.82, 0.78],
            "rouge_score": [0.75, 0.77, 0.76, 0.78, 0.74],
            "quality_score": [0.85, 0.87, 0.86, 0.88, 0.84]
        }

        result = self.validator.comprehensive_comparison(
            baseline_metrics, enhanced_metrics
        )

        # Check structure
        assert "individual_results" in result
        assert "multiple_comparison_correction" in result
        assert "overall_assessment" in result

        # Check individual results
        for metric in baseline_metrics.keys():
            assert metric in result["individual_results"]
            individual = result["individual_results"][metric]
            assert "baseline_ci" in individual
            assert "enhanced_ci" in individual
            assert "significance_test" in individual
            assert "effect_size" in individual
            assert individual["improvement"] > 0  # Should show improvement

        # Check overall assessment
        assessment = result["overall_assessment"]
        assert assessment["total_metrics"] == 3
        assert assessment["significant_improvements"] >= 0
        assert 0 <= assessment["improvement_rate"] <= 1

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            self.validator.paired_t_test_significance([1, 2, 3], [1, 2])

        # Test with insufficient samples
        with pytest.raises(ValueError):
            self.validator.paired_t_test_significance([1], [2])

        # Test effect size with mismatched paired samples
        with pytest.raises(ValueError):
            self.validator.effect_size_analysis([1, 2, 3], [1, 2], paired=True)

    def test_statistical_power_calculation(self):
        """Test statistical power calculation accuracy"""
        # Test with known effect size and sample size
        result = self.validator.power_analysis(
            effect_size=0.8,  # Large effect
            sample_size=30,
            alpha=0.05,
            test_type="paired"
        )

        # Large effect with reasonable sample size should have high power
        assert result["power"] > 0.8

        # Test with small effect size
        result_small = self.validator.power_analysis(
            effect_size=0.2,  # Small effect
            sample_size=10,
            alpha=0.05,
            test_type="paired"
        )

        # Small effect with small sample should have low power
        assert result_small["power"] < 0.5

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals have proper coverage"""
        # Generate many samples and check coverage
        np.random.seed(42)
        true_mean = 5.0
        coverage_count = 0
        n_trials = 100

        for _ in range(n_trials):
            sample = np.random.normal(true_mean, 1.0, 20)
            ci = self.validator.bootstrap_confidence_intervals(
                sample.tolist(), confidence_level=0.95
            )

            if ci.lower <= true_mean <= ci.upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials
        # Should be approximately 95% coverage (allow some variation)
        assert 0.85 <= coverage_rate <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])