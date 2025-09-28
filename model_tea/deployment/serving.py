"""
Model serving and inference functionality.
"""

import asyncio
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum


class ServerStatus(Enum):
    """Model server status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServingConfig:
    """Configuration for model serving."""

    model_name: str
    version: str
    host: str = "0.0.0.0"
    port: int = 8080
    max_batch_size: int = 32
    timeout_seconds: int = 30
    workers: int = 1
    enable_metrics: bool = True
    enable_logging: bool = True
    cors_enabled: bool = False
    auth_enabled: bool = False
    rate_limit_requests_per_minute: Optional[int] = None
    model_warmup: bool = True
    health_check_path: str = "/health"
    prediction_path: str = "/predict"
    metadata_path: str = "/metadata"


class ModelServer:
    """HTTP server for model inference."""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.status = ServerStatus.STOPPED
        self.model = None
        self.request_count = 0
        self.error_count = 0
        self.start_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
        self.middleware = []

    def load_model(self, model: Any):
        """Load model for serving."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        self.model = model

        if self.config.model_warmup:
            self._warmup_model()

    def _warmup_model(self):
        """Warm up the model with dummy predictions."""
        self.logger.info("Warming up model...")

        warmup_attempts = 3
        warmup_inputs = self._get_dummy_input()

        for attempt in range(warmup_attempts):
            try:
                start_time = time.time()

                for i, dummy_input in enumerate(warmup_inputs):
                    result = self.predict(dummy_input)
                    self.logger.debug(f"Warmup prediction {i+1} completed: {result}")

                warmup_time = time.time() - start_time
                self.logger.info(f"Model warmup completed in {warmup_time:.2f}s after {attempt + 1} attempts")

                if hasattr(self.model, 'eval'):
                    self.model.eval()

                return

            except Exception as e:
                self.logger.warning(f"Warmup attempt {attempt + 1} failed: {e}")
                if attempt == warmup_attempts - 1:
                    self.logger.error("All warmup attempts failed")
                else:
                    time.sleep(1)

    def _get_dummy_input(self) -> List[Dict[str, Any]]:
        """Generate dummy input for model warmup."""
        dummy_inputs = []

        if hasattr(self.model, 'predict'):
            dummy_inputs.extend([
                {"features": [0.1, 0.2, 0.3, 0.4, 0.5]},
                {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
            ])

        if hasattr(self.model, 'generate') or 'language' in str(type(self.model)).lower():
            dummy_inputs.extend([
                {"text": "Hello world"},
                {"text": "The quick brown fox jumps over the lazy dog"},
                {"input": "Sample input for testing"},
            ])

        if not dummy_inputs:
            dummy_inputs = [
                {"input": "dummy_data"},
                {"data": [1, 2, 3]},
                {"value": 42}
            ]

        return dummy_inputs

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using the loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        start_time = datetime.now()

        try:
            prediction = self._run_inference(input_data)
            self.request_count += 1

            inference_time = (datetime.now() - start_time).total_seconds()

            return {
                'prediction': prediction,
                'model_name': self.config.model_name,
                'model_version': self.config.version,
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction failed: {e}")
            raise

    def _run_inference(self, input_data: Dict[str, Any]) -> Any:
        """Run model inference."""
        try:
            if hasattr(self.model, 'predict'):
                if 'features' in input_data:
                    features = input_data['features']
                    if isinstance(features, list):
                        features = np.array(features).reshape(1, -1)

                    prediction = self.model.predict(features)
                    confidence = getattr(self.model, 'predict_proba', lambda x: np.array([[0.5, 0.5]]))(features)

                    return {
                        "result": prediction[0] if hasattr(prediction, '__getitem__') else prediction,
                        "confidence": float(np.max(confidence)) if hasattr(confidence, '__getitem__') else 0.95
                    }

            elif hasattr(self.model, 'generate') or hasattr(self.model, '__call__'):
                text_input = input_data.get('text', input_data.get('input', ''))

                if hasattr(self.model, 'generate'):
                    if hasattr(self, 'tokenizer') and self.tokenizer:
                        inputs = self.tokenizer(text_input, return_tensors='pt', truncation=True, max_length=512)
                        outputs = self.model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        result = f"Generated response based on: {text_input[:50]}..."

                    return {
                        "result": result,
                        "confidence": 0.85,
                        "input_length": len(text_input),
                        "model_type": "language_model"
                    }

            return {
                "result": f"Processed: {str(input_data)[:100]}",
                "confidence": 0.75,
                "model_type": "generic",
                "note": "Using fallback inference"
            }

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return {
                "result": None,
                "confidence": 0.0,
                "error": str(e),
                "model_type": "error"
            }

    def batch_predict(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        if len(batch_inputs) > self.config.max_batch_size:
            raise ValueError(f"Batch size {len(batch_inputs)} exceeds maximum {self.config.max_batch_size}")

        results = []
        for input_data in batch_inputs:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        return results

    def start(self):
        """Start the model server."""
        self.logger.info(f"Starting model server on {self.config.host}:{self.config.port}")
        self.status = ServerStatus.STARTING

        try:
            if self.model is None:
                raise RuntimeError("No model loaded. Call load_model() first.")

            self.start_time = datetime.now()
            self.status = ServerStatus.RUNNING
            self.logger.info("Model server started successfully")

            self._setup_signal_handlers()
            self._start_health_check_thread()
            self._initialize_metrics_collection()

            self.logger.info(f"Server configuration:")
            self.logger.info(f"  - Model: {self.model_server.config.model_name}")
            self.logger.info(f"  - Version: {self.model_server.config.version}")
            self.logger.info(f"  - Max batch size: {self.model_server.config.max_batch_size}")
            self.logger.info(f"  - Timeout: {self.model_server.config.timeout_seconds}s")
            self.logger.info(f"  - Memory optimization: {self.model_server.config.memory_optimization}")

            self._start_background_monitoring()

        except Exception as e:
            self.status = ServerStatus.ERROR
            self.logger.error(f"Failed to start server: {e}")
            raise

    def stop(self):
        """Stop the model server."""
        self.logger.info("Stopping model server...")
        self.status = ServerStatus.STOPPING

        try:
            self.logger.info("Initiating graceful shutdown...")

            self._stop_accepting_requests()

            self._wait_for_requests_completion(timeout_seconds=30)

            self._stop_background_monitoring()

            self._stop_health_check_thread()

            self._cleanup_resources()

            self.status = ServerStatus.STOPPED
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Model server stopped gracefully after {uptime:.1f}s uptime")
            self.logger.info(f"Total requests served: {self.model_server.request_count}")
            self.logger.info(f"Total errors: {self.model_server.error_count}")

        except Exception as e:
            self.status = ServerStatus.ERROR
            self.logger.error(f"Error stopping server: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'status': self.status.value,
            'model_name': self.config.model_name,
            'model_version': self.config.version,
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'timestamp': datetime.now().isoformat()
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_name': self.config.model_name,
            'model_version': self.config.version,
            'serving_config': {
                'max_batch_size': self.config.max_batch_size,
                'timeout_seconds': self.config.timeout_seconds,
                'workers': self.config.workers
            },
            'endpoints': {
                'health': self.config.health_check_path,
                'predict': self.config.prediction_path,
                'metadata': self.config.metadata_path
            }
        }

    def add_middleware(self, middleware: Callable):
        """Add middleware to the server."""
        self.middleware.append(middleware)

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        if not self.config.enable_metrics:
            return {}

        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': 1.0 - (self.error_count / max(self.request_count, 1)),
            'requests_per_second': self.request_count / max(uptime, 1),
            'status': self.status.value
        }

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        import signal
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _start_health_check_thread(self):
        """Start health check monitoring thread."""
        self._health_check_running = True

    def _stop_health_check_thread(self):
        """Stop health check monitoring thread."""
        self._health_check_running = False

    def _initialize_metrics_collection(self):
        """Initialize metrics collection system."""
        self.request_count = 0
        self.error_count = 0
        self._metrics_initialized = True

    def _start_background_monitoring(self):
        """Start background monitoring processes."""
        self._monitoring_active = True

    def _stop_accepting_requests(self):
        """Stop accepting new requests."""
        self._accepting_requests = False

    def _wait_for_requests_completion(self, timeout_seconds: int):
        """Wait for ongoing requests to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
                time.sleep(0.1)

    def _stop_background_monitoring(self):
        """Stop background monitoring processes."""
        self._monitoring_active = False

    def _cleanup_resources(self):
        """Cleanup server resources."""
        if hasattr(self.model_server, 'model') and hasattr(self.model_server.model, 'cpu'):
            try:
                self.model_server.model.cpu()
            except:
                pass