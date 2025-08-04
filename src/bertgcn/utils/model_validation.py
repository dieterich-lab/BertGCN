#!/usr/bin/env python3
"""
Model Validation Module for BertGCN

Provides comprehensive model validation, testing, and quality assurance.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""

    passes_validation: bool
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    validation_time: float


@dataclass
class ValidationThresholds:
    """Thresholds for model validation."""

    min_accuracy: float = 0.8
    min_f1_score: float = 0.75
    min_precision: float = 0.7
    min_recall: float = 0.7
    max_inference_time_ms: float = 1000.0
    max_memory_usage_mb: float = 2048.0


class ModelValidator:
    """Comprehensive model validator for BertGCN models."""

    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        """
        Initialize model validator.

        Args:
            thresholds: Validation thresholds
        """
        self.thresholds = thresholds or ValidationThresholds()

    def validate_model(
        self, model, test_dataloader, device: str = "cpu"
    ) -> ValidationResult:
        """
        Comprehensive model validation.

        Args:
            model: Model to validate
            test_dataloader: Test data loader
            device: Device to run validation on

        Returns:
            ValidationResult with validation status and metrics
        """
        start_time = time.time()

        try:
            logger.info("Starting comprehensive model validation")

            errors = []
            warnings = []
            metrics = {}

            # Performance validation
            perf_metrics = self._validate_performance(model, test_dataloader, device)
            metrics.update(perf_metrics)

            # Check performance thresholds
            if perf_metrics.get("accuracy", 0) < self.thresholds.min_accuracy:
                errors.append(
                    f"Accuracy {perf_metrics.get('accuracy', 0):.3f} below threshold {self.thresholds.min_accuracy}"
                )

            if perf_metrics.get("f1_score", 0) < self.thresholds.min_f1_score:
                errors.append(
                    f"F1 score {perf_metrics.get('f1_score', 0):.3f} below threshold {self.thresholds.min_f1_score}"
                )

            if perf_metrics.get("precision", 0) < self.thresholds.min_precision:
                warnings.append(
                    f"Precision {perf_metrics.get('precision', 0):.3f} below threshold {self.thresholds.min_precision}"
                )

            if perf_metrics.get("recall", 0) < self.thresholds.min_recall:
                warnings.append(
                    f"Recall {perf_metrics.get('recall', 0):.3f} below threshold {self.thresholds.min_recall}"
                )

            # Inference speed validation
            speed_metrics = self._validate_inference_speed(
                model, test_dataloader, device
            )
            metrics.update(speed_metrics)

            if (
                speed_metrics.get("avg_inference_time_ms", 0)
                > self.thresholds.max_inference_time_ms
            ):
                warnings.append(
                    f"Average inference time {speed_metrics.get('avg_inference_time_ms', 0):.1f}ms above threshold {self.thresholds.max_inference_time_ms}ms"
                )

            # Memory usage validation
            memory_metrics = self._validate_memory_usage(model, device)
            metrics.update(memory_metrics)

            if (
                memory_metrics.get("memory_usage_mb", 0)
                > self.thresholds.max_memory_usage_mb
            ):
                warnings.append(
                    f"Memory usage {memory_metrics.get('memory_usage_mb', 0):.1f}MB above threshold {self.thresholds.max_memory_usage_mb}MB"
                )

            # Model structure validation
            structure_errors = self._validate_model_structure(model)
            errors.extend(structure_errors)

            # Numerical stability validation
            stability_warnings = self._validate_numerical_stability(
                model, test_dataloader, device
            )
            warnings.extend(stability_warnings)

            validation_time = time.time() - start_time
            passes_validation = len(errors) == 0

            logger.info(
                f"Model validation completed in {validation_time:.2f}s. Passes: {passes_validation}"
            )

            return ValidationResult(
                passes_validation=passes_validation,
                metrics=metrics,
                errors=errors,
                warnings=warnings,
                validation_time=validation_time,
            )

        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return ValidationResult(
                passes_validation=False,
                metrics={},
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                validation_time=time.time() - start_time,
            )

    def _validate_performance(
        self, model, test_dataloader, device: str
    ) -> Dict[str, float]:
        """Validate model performance metrics."""
        model.eval()
        model.to(device)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:  # (graph, indices, labels)
                        graph, indices, labels = batch
                        outputs = model(graph, indices)
                    else:
                        inputs, labels = batch
                        outputs = model(inputs)
                else:
                    # Dictionary batch
                    labels = batch["labels"]
                    outputs = model(
                        **{k: v.to(device) for k, v in batch.items() if k != "labels"}
                    )

                predictions = torch.argmax(outputs, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        precision = precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "num_test_samples": len(all_labels),
        }

    def _validate_inference_speed(
        self, model, test_dataloader, device: str
    ) -> Dict[str, float]:
        """Validate model inference speed."""
        model.eval()
        model.to(device)

        inference_times = []
        num_samples = 0

        with torch.no_grad():
            for batch in test_dataloader:
                start_time = time.time()

                # Handle different batch formats
                try:
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:  # (graph, indices, labels)
                            graph, indices, labels = batch
                            outputs = model(graph, indices)
                            batch_size = len(indices)
                        else:
                            inputs, labels = batch
                            outputs = model(inputs)
                            batch_size = len(labels)
                    else:
                        # Dictionary batch
                        outputs = model(
                            **{
                                k: v.to(device)
                                for k, v in batch.items()
                                if k != "labels"
                            }
                        )
                        batch_size = len(batch["labels"])

                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    inference_times.append(inference_time)
                    num_samples += batch_size

                    # Test only a few batches for speed validation
                    if len(inference_times) >= 10:
                        break

                except Exception as e:
                    logger.warning(f"Speed validation batch failed: {str(e)}")
                    continue

        if not inference_times:
            return {"avg_inference_time_ms": 0, "total_inference_samples": 0}

        avg_time_ms = np.mean(inference_times)
        avg_time_per_sample_ms = avg_time_ms / (num_samples / len(inference_times))

        return {
            "avg_inference_time_ms": avg_time_ms,
            "avg_time_per_sample_ms": avg_time_per_sample_ms,
            "total_inference_samples": num_samples,
        }

    def _validate_memory_usage(self, model, device: str) -> Dict[str, float]:
        """Validate model memory usage."""
        model.to(device)

        # Calculate model size
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        # Get GPU memory usage if available
        gpu_memory_mb = 0
        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024

        return {
            "model_size_mb": model_size_mb,
            "memory_usage_mb": max(model_size_mb, gpu_memory_mb),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

    def _validate_model_structure(self, model) -> List[str]:
        """Validate model structure and configuration."""
        errors = []

        try:
            # Check if model has required methods
            required_methods = ["forward"]
            for method in required_methods:
                if not hasattr(model, method):
                    errors.append(f"Model missing required method: {method}")

            # Check for nan/inf parameters
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    errors.append(f"NaN values found in parameter: {name}")
                if torch.isinf(param).any():
                    errors.append(f"Inf values found in parameter: {name}")

            # Check model is in correct mode
            if model.training:
                errors.append(
                    "Model is in training mode, should be in eval mode for validation"
                )

        except Exception as e:
            errors.append(f"Structure validation failed: {str(e)}")

        return errors

    def _validate_numerical_stability(
        self, model, test_dataloader, device: str
    ) -> List[str]:
        """Validate numerical stability of model outputs."""
        warnings = []

        try:
            model.eval()
            model.to(device)

            output_values = []

            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    if i >= 3:  # Check only a few batches
                        break

                    try:
                        # Handle different batch formats
                        if isinstance(batch, (list, tuple)):
                            if len(batch) == 3:  # (graph, indices, labels)
                                graph, indices, labels = batch
                                outputs = model(graph, indices)
                            else:
                                inputs, labels = batch
                                outputs = model(inputs)
                        else:
                            # Dictionary batch
                            outputs = model(
                                **{
                                    k: v.to(device)
                                    for k, v in batch.items()
                                    if k != "labels"
                                }
                            )

                        output_values.extend(outputs.cpu().numpy().flatten())

                        # Check for nan/inf in outputs
                        if torch.isnan(outputs).any():
                            warnings.append("NaN values detected in model outputs")
                        if torch.isinf(outputs).any():
                            warnings.append("Inf values detected in model outputs")

                    except Exception as e:
                        warnings.append(
                            f"Numerical stability check failed on batch: {str(e)}"
                        )

            # Check output distribution
            if output_values:
                output_std = np.std(output_values)
                output_mean = np.mean(output_values)

                if output_std > 100:
                    warnings.append(
                        f"High output variance detected (std: {output_std:.2f})"
                    )
                if abs(output_mean) > 50:
                    warnings.append(
                        f"Large output bias detected (mean: {output_mean:.2f})"
                    )

        except Exception as e:
            warnings.append(f"Numerical stability validation failed: {str(e)}")

        return warnings

    def validate_model_performance(
        self,
        model_uri: str,
        test_data_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        High-level method to validate model performance from URI.

        Args:
            model_uri: MLflow model URI
            test_data_path: Path to test data
            progress_callback: Optional progress callback function

        Returns:
            Dictionary with validation results
        """
        try:
            import mlflow

            if progress_callback:
                progress_callback(10)

            # Load model
            model = mlflow.pytorch.load_model(model_uri)

            if progress_callback:
                progress_callback(30)

            # Load test data (placeholder - would need actual implementation)
            # For now, create dummy validation
            if test_data_path and test_data_path.exists():
                # Would load actual test data here
                pass

            if progress_callback:
                progress_callback(50)

            # Create dummy dataloader for validation
            # In practice, this would use actual test data
            class DummyDataLoader:
                def __iter__(self):
                    # Create dummy batch
                    dummy_input = torch.randn(
                        2, 512
                    )  # Batch size 2, sequence length 512
                    dummy_labels = torch.tensor([0, 1])
                    yield (dummy_input, dummy_labels)

            dummy_dataloader = DummyDataLoader()

            if progress_callback:
                progress_callback(70)

            # Run validation
            result = self.validate_model(model, dummy_dataloader)

            if progress_callback:
                progress_callback(100)

            # Format results for CLI display
            formatted_results = {}
            for metric, value in result.metrics.items():
                formatted_results[metric] = {
                    "value": value,
                    "threshold": getattr(self.thresholds, f"min_{metric}", 0.0),
                    "passes": value >= getattr(self.thresholds, f"min_{metric}", 0.0),
                }

            return formatted_results

        except Exception as e:
            logger.error(f"Model performance validation failed: {str(e)}")
            raise
