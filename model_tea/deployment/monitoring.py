"""
Deployment monitoring and health checking.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DeploymentMetrics:
    """Metrics for deployment monitoring."""

    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    request_count: int = 0
    error_count: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / max(self.request_count, 1)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - self.error_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_rate,
            'success_rate': self.success_rate,
            'average_response_time_ms': self.average_response_time_ms,
            'p95_response_time_ms': self.p95_response_time_ms,
            'p99_response_time_ms': self.p99_response_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class HealthCheck:
    """Individual health check."""

    def __init__(self, name: str, check_function: Callable[[], bool], timeout_seconds: int = 5):
        self.name = name
        self.check_function = check_function
        self.timeout_seconds = timeout_seconds
        self.last_status = HealthStatus.UNKNOWN
        self.last_check_time: Optional[datetime] = None
        self.consecutive_failures = 0

    def run_check(self) -> HealthStatus:
        """Run the health check."""
        try:
            start_time = time.time()
            is_healthy = self.check_function()
            check_duration = time.time() - start_time

            if check_duration > self.timeout_seconds:
                self.last_status = HealthStatus.DEGRADED
                self.consecutive_failures += 1
            elif is_healthy:
                self.last_status = HealthStatus.HEALTHY
                self.consecutive_failures = 0
            else:
                self.last_status = HealthStatus.UNHEALTHY
                self.consecutive_failures += 1

        except Exception as e:
            logging.getLogger(__name__).error(f"Health check {self.name} failed: {e}")
            self.last_status = HealthStatus.UNHEALTHY
            self.consecutive_failures += 1

        self.last_check_time = datetime.now()
        return self.last_status


class DeploymentMonitor:
    """Monitor deployment health and performance."""

    def __init__(self, deployment_name: str, check_interval_seconds: int = 30):
        self.deployment_name = deployment_name
        self.check_interval_seconds = check_interval_seconds
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[DeploymentMetrics] = []
        self.alert_handlers: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        self.is_monitoring = False
        self.last_metrics: Optional[DeploymentMetrics] = None

    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Added health check: {health_check.name}")

    def add_alert_handler(self, handler: Callable[[str, HealthStatus], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    def collect_metrics(self) -> DeploymentMetrics:
        """Collect current deployment metrics."""
        # Placeholder for actual metrics collection
        # In real implementation, this would gather metrics from the deployment
        metrics = DeploymentMetrics(
            cpu_usage_percent=self._get_cpu_usage(),
            memory_usage_percent=self._get_memory_usage(),
            request_count=self._get_request_count(),
            error_count=self._get_error_count(),
            average_response_time_ms=self._get_avg_response_time(),
            p95_response_time_ms=self._get_p95_response_time(),
            p99_response_time_ms=self._get_p99_response_time()
        )

        self.metrics_history.append(metrics)
        self.last_metrics = metrics

        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        return metrics

    def run_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for name, check in self.health_checks.items():
            status = check.run_check()
            results[name] = status

            # Determine overall status
            if status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

            # Trigger alerts if needed
            if status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                self._trigger_alerts(name, status)

        return results

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        health_results = self.run_health_checks()
        metrics = self.collect_metrics()

        # Determine overall status
        statuses = list(health_results.values())
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return {
            'deployment_name': self.deployment_name,
            'overall_status': overall_status.value,
            'health_checks': {name: status.value for name, status in health_results.items()},
            'metrics': metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

    def start_monitoring(self):
        """Start continuous monitoring."""
        self.logger.info(f"Starting monitoring for deployment: {self.deployment_name}")
        self.is_monitoring = True

        # In a real implementation, this would run in a separate thread or async task
        # For now, it's a placeholder

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.logger.info(f"Stopping monitoring for deployment: {self.deployment_name}")
        self.is_monitoring = False

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {}

        return {
            'time_period_hours': hours,
            'total_requests': sum(m.request_count for m in recent_metrics),
            'total_errors': sum(m.error_count for m in recent_metrics),
            'average_cpu_usage': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            'average_memory_usage': sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics),
            'average_response_time': sum(m.average_response_time_ms for m in recent_metrics) / len(recent_metrics),
            'max_response_time': max(m.p99_response_time_ms for m in recent_metrics),
            'sample_count': len(recent_metrics)
        }

    def _trigger_alerts(self, check_name: str, status: HealthStatus):
        """Trigger alerts for failed health checks."""
        for handler in self.alert_handlers:
            try:
                handler(check_name, status)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        # Placeholder - in real implementation, would get actual CPU metrics
        import random
        return random.uniform(10, 80)

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        # Placeholder - in real implementation, would get actual memory metrics
        import random
        return random.uniform(20, 70)

    def _get_request_count(self) -> int:
        """Get current request count."""
        # Placeholder - in real implementation, would get actual request metrics
        import random
        return random.randint(100, 1000)

    def _get_error_count(self) -> int:
        """Get current error count."""
        # Placeholder - in real implementation, would get actual error metrics
        import random
        return random.randint(0, 50)

    def _get_avg_response_time(self) -> float:
        """Get average response time in milliseconds."""
        # Placeholder - in real implementation, would get actual response time metrics
        import random
        return random.uniform(50, 500)

    def _get_p95_response_time(self) -> float:
        """Get 95th percentile response time in milliseconds."""
        # Placeholder - in real implementation, would get actual P95 metrics
        return self._get_avg_response_time() * 1.5

    def _get_p99_response_time(self) -> float:
        """Get 99th percentile response time in milliseconds."""
        # Placeholder - in real implementation, would get actual P99 metrics
        return self._get_avg_response_time() * 2.0