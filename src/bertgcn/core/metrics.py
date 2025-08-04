#!/usr/bin/env python3
"""
Metrics Module for BertGCN

Comprehensive metrics calculation and tracking for model evaluation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for model metrics."""

    accuracy: float
    f1_score: float
    precision: float
    recall: float
    specificity: Optional[float] = None
    auc_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


class ModelMetrics:
    """Comprehensive metrics calculator for BertGCN models."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history = []

    def calculate_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_proba: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> MetricsResult:
        """
        Calculate comprehensive metrics for model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for AUC calculation)
            class_names: Names of classes for reporting

        Returns:
            MetricsResult with calculated metrics
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            precision = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Classification report
            target_names = class_names if class_names else None
            class_report = classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            )

            # Specificity (for binary classification)
            specificity = None
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # AUC score (if probabilities provided)
            auc_score = None
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        auc_score = roc_auc_score(y_true, [p[1] for p in y_proba])
                    else:
                        # Multi-class classification
                        auc_score = roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="macro"
                        )
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC score: {str(e)}")

            result = MetricsResult(
                accuracy=accuracy,
                f1_score=f1,
                precision=precision,
                recall=recall,
                specificity=specificity,
                auc_score=auc_score,
                confusion_matrix=cm,
                classification_report=class_report,
            )

            logger.info(
                f"Metrics calculated: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            raise

    def calculate_final_metrics(
        self, test_results: List[Dict[str, Any]], validation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate final metrics combining test and validation results.

        Args:
            test_results: Test results from trainer
            validation_results: Validation results from model validator

        Returns:
            Dictionary of final metrics
        """
        try:
            final_metrics = {}

            # Extract test metrics
            if test_results:
                test_metrics = (
                    test_results[0] if isinstance(test_results, list) else test_results
                )

                for key, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        final_metrics[f"test_{key}"] = value

            # Add validation metrics
            if validation_results:
                for key, value in validation_results.items():
                    if isinstance(value, (int, float)):
                        final_metrics[f"val_{key}"] = value

            # Calculate aggregate metrics
            test_accuracy = final_metrics.get("test_accuracy", 0.0)
            test_f1 = final_metrics.get("test_f1", 0.0)
            val_accuracy = final_metrics.get("val_accuracy", 0.0)
            val_f1 = final_metrics.get("val_f1", 0.0)

            # Overall performance score (weighted average)
            overall_score = test_f1 * 0.7 + val_f1 * 0.3
            final_metrics["overall_score"] = overall_score

            # Performance grade
            if overall_score >= 0.9:
                grade = "A"
            elif overall_score >= 0.8:
                grade = "B"
            elif overall_score >= 0.7:
                grade = "C"
            elif overall_score >= 0.6:
                grade = "D"
            else:
                grade = "F"

            final_metrics["performance_grade"] = grade

            # Store in history
            self.metrics_history.append(final_metrics)

            logger.info(
                f"Final metrics calculated. Overall score: {overall_score:.4f}, Grade: {grade}"
            )

            return final_metrics

        except Exception as e:
            logger.error(f"Failed to calculate final metrics: {str(e)}")
            return {}

    def compare_metrics(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float],
        metric_name: str = "f1_score",
    ) -> Dict[str, Any]:
        """
        Compare two sets of metrics.

        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            metric_name: Primary metric for comparison

        Returns:
            Comparison results
        """
        try:
            comparison = {
                "metric_name": metric_name,
                "metrics1_value": metrics1.get(metric_name, 0.0),
                "metrics2_value": metrics2.get(metric_name, 0.0),
            }

            # Calculate improvement
            value1 = metrics1.get(metric_name, 0.0)
            value2 = metrics2.get(metric_name, 0.0)

            if value1 > 0:
                improvement = ((value2 - value1) / value1) * 100
            else:
                improvement = 0.0

            comparison["improvement_percent"] = improvement
            comparison["is_better"] = value2 > value1
            comparison["difference"] = value2 - value1

            # Compare all common metrics
            common_metrics = set(metrics1.keys()) & set(metrics2.keys())
            detailed_comparison = {}

            for metric in common_metrics:
                if isinstance(metrics1[metric], (int, float)) and isinstance(
                    metrics2[metric], (int, float)
                ):
                    detailed_comparison[metric] = {
                        "value1": metrics1[metric],
                        "value2": metrics2[metric],
                        "difference": metrics2[metric] - metrics1[metric],
                        "improvement": (
                            (
                                (metrics2[metric] - metrics1[metric])
                                / metrics1[metric]
                                * 100
                            )
                            if metrics1[metric] > 0
                            else 0.0
                        ),
                    }

            comparison["detailed_comparison"] = detailed_comparison

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare metrics: {str(e)}")
            return {}

    def get_metrics_summary(self, num_recent: int = 5) -> Dict[str, Any]:
        """
        Get summary of recent metrics.

        Args:
            num_recent: Number of recent metrics to include

        Returns:
            Summary of metrics
        """
        if not self.metrics_history:
            return {"message": "No metrics history available"}

        recent_metrics = self.metrics_history[-num_recent:]

        # Calculate trends
        summary = {
            "num_experiments": len(self.metrics_history),
            "recent_experiments": len(recent_metrics),
            "best_overall_score": max(
                m.get("overall_score", 0) for m in self.metrics_history
            ),
            "latest_overall_score": (
                recent_metrics[-1].get("overall_score", 0) if recent_metrics else 0
            ),
            "recent_metrics": recent_metrics,
        }

        # Calculate average metrics for recent experiments
        if recent_metrics:
            avg_metrics = {}
            for key in [
                "test_accuracy",
                "test_f1",
                "val_accuracy",
                "val_f1",
                "overall_score",
            ]:
                values = [m.get(key, 0) for m in recent_metrics if key in m]
                if values:
                    avg_metrics[f"avg_{key}"] = sum(values) / len(values)

            summary["average_metrics"] = avg_metrics

        return summary

    def export_metrics(self, output_path: str) -> bool:
        """
        Export metrics history to file.

        Args:
            output_path: Path to save metrics

        Returns:
            True if successful
        """
        try:
            import json
            from pathlib import Path

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_timestamp": str(pd.Timestamp.now()),
                "num_experiments": len(self.metrics_history),
                "metrics_history": self.metrics_history,
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Metrics exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return False


def calculate_model_metrics(
    y_true: List[int], y_pred: List[int], y_proba: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """
    Convenience function to calculate basic model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities

    Returns:
        Dictionary of metrics
    """
    metrics_calculator = ModelMetrics()
    result = metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)

    return {
        "accuracy": result.accuracy,
        "f1_score": result.f1_score,
        "precision": result.precision,
        "recall": result.recall,
        "specificity": result.specificity or 0.0,
        "auc_score": result.auc_score or 0.0,
    }
