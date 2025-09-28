"""
Test the model evaluation framework
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from model_tea.core.evaluation import (
    EvaluationMetrics, ModelEvaluator, AccuracyCalculator,
    PrecisionCalculator, RecallCalculator
)


class TestEvaluationMetrics:
    """Test evaluation metrics container"""

    def test_metrics_creation(self):
        """Test metrics can be created"""
        metrics = EvaluationMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.875
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.875

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary"""
        metrics = EvaluationMetrics(
            accuracy=0.95,
            precision=0.90,
            custom_metrics={"mse": 0.05}
        )

        result = metrics.to_dict()
        assert result["accuracy"] == 0.95
        assert result["precision"] == 0.90
        assert result["mse"] == 0.05
        assert "recall" not in result

    def test_metrics_string_representation(self):
        """Test string representation of metrics"""
        metrics = EvaluationMetrics(accuracy=0.95, precision=0.90)

        result = str(metrics)
        assert "accuracy: 0.9500" in result
        assert "precision: 0.9000" in result


class TestAccuracyCalculator:
    """Test accuracy metric calculator"""

    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        calculator = AccuracyCalculator()

        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])

        accuracy = calculator.calculate(y_true, y_pred)
        assert accuracy == 0.8

    def test_perfect_accuracy(self):
        """Test perfect accuracy"""
        calculator = AccuracyCalculator()

        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0])

        accuracy = calculator.calculate(y_true, y_pred)
        assert accuracy == 1.0


class TestPrecisionCalculator:
    """Test precision metric calculator"""

    def test_binary_precision(self):
        """Test binary precision calculation"""
        calculator = PrecisionCalculator(average='binary')

        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])

        precision = calculator.calculate(y_true, y_pred)
        assert precision == 1.0

    def test_precision_with_no_positives(self):
        """Test precision when no positive predictions"""
        calculator = PrecisionCalculator(average='binary')

        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0, 0])

        precision = calculator.calculate(y_true, y_pred)
        assert precision == 0.0

    def test_macro_precision(self):
        """Test macro-averaged precision"""
        calculator = PrecisionCalculator(average='macro')

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])

        precision = calculator.calculate(y_true, y_pred)
        assert precision > 0


class TestRecallCalculator:
    """Test recall metric calculator"""

    def test_binary_recall(self):
        """Test binary recall calculation"""
        calculator = RecallCalculator(average='binary')

        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])

        recall = calculator.calculate(y_true, y_pred)
        assert recall == 2/3

    def test_recall_with_no_true_positives(self):
        """Test recall when no true positives in ground truth"""
        calculator = RecallCalculator(average='binary')

        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0])

        recall = calculator.calculate(y_true, y_pred)
        assert recall == 0.0

    def test_macro_recall(self):
        """Test macro-averaged recall"""
        calculator = RecallCalculator(average='macro')

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])

        recall = calculator.calculate(y_true, y_pred)
        assert recall > 0


class TestModelEvaluator:
    """Test model evaluation framework"""

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized"""
        evaluator = ModelEvaluator()

        assert "accuracy" in evaluator.calculators
        assert "precision" in evaluator.calculators
        assert "recall" in evaluator.calculators

    def test_add_custom_metric(self):
        """Test adding custom metric"""
        evaluator = ModelEvaluator()
        custom_calc = AccuracyCalculator()

        evaluator.add_custom_metric("custom_accuracy", custom_calc)
        assert "custom_accuracy" in evaluator.custom_calculators

    def test_evaluate_with_sklearn_model(self):
        """Test evaluation with sklearn model"""
        from sklearn.linear_model import LogisticRegression

        evaluator = ModelEvaluator()
        model = LogisticRegression(max_iter=100)

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)

        test_data = {"X": X, "y": y}
        metrics = evaluator.evaluate(model, test_data)

        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None

    def test_evaluate_with_mock_model(self):
        """Test evaluation with mock model"""
        evaluator = ModelEvaluator()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 1, 0])

        test_data = {
            "X": np.random.rand(5, 3),
            "y": np.array([1, 0, 1, 0, 0])
        }

        metrics = evaluator.evaluate(mock_model, test_data)

        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None

    def test_evaluate_with_tuple_data(self):
        """Test evaluation with tuple test data"""
        evaluator = ModelEvaluator()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1])

        X_test = np.random.rand(3, 2)
        y_test = np.array([1, 0, 1])
        test_data = (X_test, y_test)

        metrics = evaluator.evaluate(mock_model, test_data)

        assert metrics.accuracy is not None

    def test_evaluate_with_no_predict_method(self):
        """Test evaluation with model that has no predict method"""
        evaluator = ModelEvaluator()

        mock_model = MagicMock()
        del mock_model.predict

        metrics = evaluator.evaluate(mock_model, None)

        assert metrics.accuracy is not None

    def test_f1_score_calculation(self):
        """Test F1 score calculation"""
        evaluator = ModelEvaluator()

        f1 = evaluator._calculate_f1(0.8, 0.6)
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(f1 - expected_f1) < 1e-6

    def test_f1_score_with_none_values(self):
        """Test F1 score with None precision or recall"""
        evaluator = ModelEvaluator()

        assert evaluator._calculate_f1(None, 0.6) is None
        assert evaluator._calculate_f1(0.8, None) is None
        assert evaluator._calculate_f1(0.0, 0.0) is None

    def test_cross_validation(self):
        """Test cross-validation evaluation"""
        from sklearn.linear_model import LogisticRegression

        evaluator = ModelEvaluator()
        model = LogisticRegression(max_iter=100)

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        data = {"X": X, "y": y}

        cv_results = evaluator.cross_validate(model, data, cv_folds=3)

        assert "accuracy" in cv_results
        assert len(cv_results["accuracy"]) == 3

    def test_cross_validation_with_tuple_data(self):
        """Test cross-validation with tuple data"""
        from sklearn.linear_model import LogisticRegression

        evaluator = ModelEvaluator()
        model = LogisticRegression(max_iter=100)

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        data = (X, y)

        cv_results = evaluator.cross_validate(model, data, cv_folds=3, metrics=["accuracy", "precision"])

        assert "accuracy" in cv_results
        assert "precision" in cv_results
        assert len(cv_results["accuracy"]) == 3

    def test_cross_validation_with_model_without_fit(self):
        """Test cross-validation with model that has no fit method"""
        evaluator = ModelEvaluator()

        mock_model = MagicMock()
        del mock_model.fit

        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        data = {"X": X, "y": y}

        cv_results = evaluator.cross_validate(mock_model, data, cv_folds=3)

        assert "accuracy" in cv_results
        assert len(cv_results["accuracy"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])