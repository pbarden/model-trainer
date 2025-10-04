import logging
import numpy as np
from typing import Dict, Any

from .config import IterativeConfig

logger = logging.getLogger(__name__)


class AdaptiveTrainingMonitor:
    """Monitors training progress and detects overfitting for adaptive training decisions"""

    def __init__(self, config: IterativeConfig):
        self.config = config
        self.validation_losses = []
        self.perplexities = []
        self.quality_scores = []
        self.best_validation_loss = float('inf')
        self.best_perplexity = float('inf')
        self.best_quality = 0.0
        self.patience_counter = 0
        self.should_stop = False

    def update_metrics(self, validation_loss: float, perplexity: float, quality: float, iteration: int):
        """Update training metrics and check for stopping conditions"""
        self.validation_losses.append(validation_loss)
        self.perplexities.append(perplexity)
        self.quality_scores.append(quality)

        # Track best metrics
        improved = False
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            improved = True
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity
            improved = True
        if quality > self.best_quality:
            self.best_quality = quality
            improved = True

        # Only track patience AFTER completing min_iterations
        # iteration is 0-based, so iteration >= 5 means we've completed iterations 0-4 (5 total)
        # and are now on iteration 5 or later
        if iteration >= self.config.min_iterations:
            # Reset patience if improved
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            # During first min_iterations (0-4 for min=5), don't track patience at all
            self.patience_counter = 0

        # Check stopping conditions
        self.should_stop = self._should_stop_training(iteration)

        return {
            "improved": improved,
            "patience_counter": self.patience_counter,
            "should_stop": self.should_stop,
            "overfitting_detected": self._detect_overfitting(),
            "learning_rate_adjustment": self._suggest_lr_adjustment(),
            "recommendation": self._get_training_recommendation()
        }

    def _should_stop_training(self, iteration: int) -> bool:
        """Determine if training should stop"""
        if not self.config.adaptive_training:
            # If adaptive training is disabled, use max_iterations as hard limit
            return iteration >= self.config.max_iterations

        # CRITICAL: Must complete at least min_iterations (default 5)
        # iteration is 0-based index checked BEFORE starting that iteration
        # iteration=5 means we've completed 0,1,2,3,4 (5 iterations) and are about to start iteration 5
        # So we only allow stopping when iteration >= min_iterations (i.e., we've completed min_iterations)
        if iteration < self.config.min_iterations:
            return False

        # Maximum iterations check (hard upper bound)
        if iteration >= self.config.max_iterations:
            return True

        # All other checks ONLY apply after min_iterations completed

        # Perplexity explosion check - CRITICAL (only after min_iterations)
        if len(self.perplexities) >= 2:
            recent_perplexity = self.perplexities[-1]
            if recent_perplexity > self.config.perplexity_threshold:
                logger.warning(f"Stopping: Perplexity {recent_perplexity:.1f} exceeds threshold {self.config.perplexity_threshold}")
                return True

        # Target perplexity achieved - EARLY SUCCESS (only after min_iterations)
        if hasattr(self.config, 'target_perplexity') and len(self.perplexities) >= 2:
            if self.best_perplexity <= self.config.target_perplexity:
                logger.info(f"SUCCESS: Target perplexity {self.config.target_perplexity} achieved ({self.best_perplexity:.1f})")
                return True

        # Early stopping based on patience (only after min_iterations)
        if self.patience_counter >= self.config.early_stopping_patience:
            return True

        # Validation loss plateau detection (only after min_iterations)
        if len(self.validation_losses) >= self.config.validation_loss_patience:
            recent_losses = self.validation_losses[-self.config.validation_loss_patience:]
            if max(recent_losses) - min(recent_losses) < 0.01:  # Very small improvement
                return True

        return False

    def _detect_overfitting(self) -> bool:
        """Detect overfitting patterns"""
        if len(self.validation_losses) < self.config.overfitting_detection_window:
            return False

        window = self.config.overfitting_detection_window
        recent_val_losses = self.validation_losses[-window:]
        recent_perplexities = self.perplexities[-window:]
        recent_qualities = self.quality_scores[-window:]

        # Check for increasing validation loss
        val_loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]

        # Check for increasing perplexity
        perplexity_trend = np.polyfit(range(len(recent_perplexities)), recent_perplexities, 1)[0]

        # Check for decreasing quality
        quality_trend = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]

        # Overfitting indicators
        val_loss_increasing = val_loss_trend > 0.01
        perplexity_increasing = perplexity_trend > self.config.perplexity_improvement_threshold
        quality_decreasing = quality_trend < -self.config.quality_degradation_threshold

        return val_loss_increasing and (perplexity_increasing or quality_decreasing)

    def _suggest_lr_adjustment(self) -> Dict[str, Any]:
        """Suggest learning rate adjustments based on training patterns"""
        if len(self.validation_losses) < 2:
            return {"action": "maintain", "factor": 1.0}

        # Check perplexity first - most critical
        if len(self.perplexities) >= 2:
            recent_perplexity = self.perplexities[-1]
            if recent_perplexity > self.config.perplexity_threshold * 0.8:  # 80% of threshold
                return {"action": "reduce", "factor": 0.2, "reason": "high_perplexity"}
            elif recent_perplexity > self.config.target_perplexity * 2:  # 2x target
                return {"action": "reduce", "factor": 0.4, "reason": "above_target_perplexity"}

        recent_improvements = [
            self.validation_losses[i-1] - self.validation_losses[i]
            for i in range(1, min(4, len(self.validation_losses)))
        ]

        avg_improvement = np.mean(recent_improvements)

        if avg_improvement < 0.001:  # Very slow improvement
            return {"action": "reduce", "factor": 0.5, "reason": "slow_convergence"}
        elif avg_improvement < 0:  # Getting worse
            return {"action": "reduce", "factor": 0.3, "reason": "performance_degradation"}
        elif avg_improvement > 0.05:  # Fast improvement
            return {"action": "maintain", "factor": 1.0, "reason": "good_progress"}
        else:
            return {"action": "maintain", "factor": 1.0, "reason": "stable_progress"}

    def _get_training_recommendation(self) -> str:
        """Get overall training recommendation"""
        if self.should_stop:
            if self._detect_overfitting():
                return "stop_overfitting"
            elif self.patience_counter >= self.config.early_stopping_patience:
                return "stop_plateau"
            else:
                return "stop_max_iterations"

        if self._detect_overfitting():
            return "reduce_lr_overfitting"

        if self.patience_counter >= 2:
            return "reduce_lr_plateau"

        return "continue_training"

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if not self.validation_losses:
            return {"status": "no_data"}

        return {
            "total_iterations": len(self.validation_losses),
            "best_validation_loss": self.best_validation_loss,
            "best_perplexity": self.best_perplexity,
            "best_quality": self.best_quality,
            "final_validation_loss": self.validation_losses[-1],
            "final_perplexity": self.perplexities[-1],
            "final_quality": self.quality_scores[-1],
            "improvement_trend": {
                "validation_loss": self.best_validation_loss - self.validation_losses[-1],
                "perplexity": self.best_perplexity - self.perplexities[-1],
                "quality": self.quality_scores[-1] - self.best_quality
            },
            "training_efficiency": self._calculate_efficiency(),
            "overfitting_detected": self._detect_overfitting()
        }

    def _calculate_efficiency(self) -> float:
        """Calculate training efficiency score"""
        if len(self.validation_losses) < 2:
            return 0.0

        # Measure how much improvement per iteration
        total_val_improvement = max(0, self.validation_losses[0] - self.best_validation_loss)
        total_quality_improvement = max(0, self.best_quality - self.quality_scores[0])

        iterations = len(self.validation_losses)

        # Normalize and combine improvements
        efficiency = (total_val_improvement * 0.4 + total_quality_improvement * 0.6) / iterations
        return min(1.0, efficiency * 10)  # Scale to 0-1

