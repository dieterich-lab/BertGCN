#!/usr/bin/env python3
"""
Monitoring and Metrics Module for BertGCN

Provides monitoring utilities, metrics collection, and performance tracking.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for prediction performance."""

    num_samples: int
    inference_time: float
    avg_time_per_sample: float
    model_uri: str
    timestamp: datetime


class PerformanceMonitor:
    """Monitor model performance and system metrics."""

    def __init__(self):
        self.metrics_history = []

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to MLflow and local storage."""
        try:
            # Log to MLflow if available
            if MLFLOW_AVAILABLE and mlflow.active_run():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))

            # Store locally
            self.metrics_history.append(
                {"timestamp": datetime.now(), "metrics": metrics}
            )

            logger.debug(f"Logged metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")

    def get_recent_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            entry for entry in self.metrics_history if entry["timestamp"] > cutoff_time
        ]


# Global monitor instance
performance_monitor = PerformanceMonitor()


def log_prediction_metrics(
    num_samples: int,
    inference_time: float,
    model_uri: str,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log prediction performance metrics.

    Args:
        num_samples: Number of samples processed
        inference_time: Total inference time in seconds
        model_uri: URI of the model used
        additional_metrics: Optional additional metrics to log
    """
    try:
        avg_time_per_sample = inference_time / num_samples if num_samples > 0 else 0

        metrics = {
            "prediction_num_samples": num_samples,
            "prediction_inference_time": inference_time,
            "prediction_avg_time_per_sample": avg_time_per_sample,
            "prediction_throughput": (
                num_samples / inference_time if inference_time > 0 else 0
            ),
            "model_uri": model_uri,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Log to performance monitor
        performance_monitor.log_metrics(metrics)

        # Create structured metrics object
        prediction_metrics = PredictionMetrics(
            num_samples=num_samples,
            inference_time=inference_time,
            avg_time_per_sample=avg_time_per_sample,
            model_uri=model_uri,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Prediction metrics logged: {num_samples} samples, {inference_time:.3f}s total, {avg_time_per_sample:.3f}s/sample"
        )

    except Exception as e:
        logger.error(f"Failed to log prediction metrics: {str(e)}")


def log_training_metrics(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    val_f1: Optional[float] = None,
    learning_rate: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log training metrics.

    Args:
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_f1: Validation F1 score
        learning_rate: Current learning rate
        additional_metrics: Optional additional metrics
    """
    try:
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "timestamp": datetime.now().isoformat(),
        }

        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if val_f1 is not None:
            metrics["val_f1"] = val_f1
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        if additional_metrics:
            metrics.update(additional_metrics)

        performance_monitor.log_metrics(metrics)

        logger.info(
            f"Training metrics logged for epoch {epoch}: train_loss={train_loss:.4f}"
        )

    except Exception as e:
        logger.error(f"Failed to log training metrics: {str(e)}")


def log_system_metrics(
    cpu_usage: Optional[float] = None,
    memory_usage: Optional[float] = None,
    gpu_usage: Optional[float] = None,
    gpu_memory: Optional[float] = None,
) -> None:
    """
    Log system resource metrics.

    Args:
        cpu_usage: CPU usage percentage
        memory_usage: Memory usage percentage
        gpu_usage: GPU usage percentage
        gpu_memory: GPU memory usage percentage
    """
    try:
        metrics = {"timestamp": datetime.now().isoformat()}

        if cpu_usage is not None:
            metrics["system_cpu_usage"] = cpu_usage
        if memory_usage is not None:
            metrics["system_memory_usage"] = memory_usage
        if gpu_usage is not None:
            metrics["system_gpu_usage"] = gpu_usage
        if gpu_memory is not None:
            metrics["system_gpu_memory"] = gpu_memory

        performance_monitor.log_metrics(metrics)

    except Exception as e:
        logger.error(f"Failed to log system metrics: {str(e)}")


def create_metrics_dashboard_data() -> Dict[str, Any]:
    """
    Create data for metrics dashboard.

    Returns:
        Dictionary containing dashboard data
    """
    try:
        recent_metrics = performance_monitor.get_recent_metrics(hours=24)

        if not recent_metrics:
            return {"message": "No recent metrics available"}

        # Extract different types of metrics
        prediction_metrics = []
        training_metrics = []
        system_metrics = []

        for entry in recent_metrics:
            metrics = entry["metrics"]
            if "prediction_num_samples" in metrics:
                prediction_metrics.append(metrics)
            elif "epoch" in metrics:
                training_metrics.append(metrics)
            elif "system_cpu_usage" in metrics:
                system_metrics.append(metrics)

        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_predictions": sum(
                    m.get("prediction_num_samples", 0) for m in prediction_metrics
                ),
                "avg_inference_time": (
                    sum(
                        m.get("prediction_inference_time", 0)
                        for m in prediction_metrics
                    )
                    / len(prediction_metrics)
                    if prediction_metrics
                    else 0
                ),
                "recent_training_epochs": len(training_metrics),
                "last_val_f1": (
                    training_metrics[-1].get("val_f1") if training_metrics else None
                ),
            },
            "prediction_metrics": prediction_metrics[-50:],  # Last 50 predictions
            "training_metrics": training_metrics[-100:],  # Last 100 training steps
            "system_metrics": system_metrics[-100:],  # Last 100 system measurements
        }

        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to create dashboard data: {str(e)}")
        return {"error": str(e)}


class MetricsCollector:
    """Collect and aggregate metrics over time."""

    def __init__(self):
        self.collected_metrics = []

    def collect(self, metrics: Dict[str, Any]) -> None:
        """Collect metrics with timestamp."""
        timestamped_metrics = {"timestamp": datetime.now(), **metrics}
        self.collected_metrics.append(timestamped_metrics)

    def get_aggregated_metrics(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get aggregated metrics for a time window."""
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_metrics = [
            m for m in self.collected_metrics if m["timestamp"] > cutoff_time
        ]

        if not recent_metrics:
            return {}

        # Aggregate numeric metrics
        aggregated = {}
        numeric_keys = set()

        for metrics in recent_metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key != "timestamp":
                    numeric_keys.add(key)

        for key in numeric_keys:
            values = [
                m[key]
                for m in recent_metrics
                if key in m and isinstance(m[key], (int, float))
            ]
            if values:
                aggregated[f"{key}_avg"] = sum(values) / len(values)
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_count"] = len(values)

        return aggregated
