#!/usr/bin/env python3
"""
Data Validation Module for BertGCN

Provides comprehensive data validation and quality checks for clinical text data.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


class DataValidator:
    """Comprehensive data validator for clinical text datasets."""

    def __init__(self, min_text_length: int = 10, max_text_length: int = 10000):
        """
        Initialize the data validator.

        Args:
            min_text_length: Minimum required text length
            max_text_length: Maximum allowed text length
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    def validate_dataset(self, data_path: Path) -> ValidationResult:
        """
        Validate a complete dataset.

        Args:
            data_path: Path to the dataset file

        Returns:
            ValidationResult with validation status and details
        """
        logger.info(f"Validating dataset: {data_path}")

        errors = []
        warnings = []
        metrics = {}

        try:
            # Load data
            if data_path.suffix == ".csv":
                df = pd.read_csv(data_path)
            elif data_path.suffix == ".json":
                df = pd.read_json(data_path)
            else:
                errors.append(f"Unsupported file format: {data_path.suffix}")
                return ValidationResult(False, errors, warnings, metrics)

            # Basic structure validation
            structure_errors = self._validate_structure(df)
            errors.extend(structure_errors)

            # Text quality validation
            text_errors, text_warnings, text_metrics = self._validate_text_quality(df)
            errors.extend(text_errors)
            warnings.extend(text_warnings)
            metrics.update(text_metrics)

            # Label validation
            label_errors, label_metrics = self._validate_labels(df)
            errors.extend(label_errors)
            metrics.update(label_metrics)

            # Statistical validation
            stats_warnings, stats_metrics = self._validate_statistics(df)
            warnings.extend(stats_warnings)
            metrics.update(stats_metrics)

            is_valid = len(errors) == 0

            logger.info(
                f"Validation completed. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}"
            )

            return ValidationResult(is_valid, errors, warnings, metrics)

        except Exception as e:
            errors.append(f"Validation failed with error: {str(e)}")
            return ValidationResult(False, errors, warnings, metrics)

    def _validate_structure(self, df: pd.DataFrame) -> List[str]:
        """Validate basic dataset structure."""
        errors = []

        # Check if DataFrame is empty
        if df.empty:
            errors.append("Dataset is empty")
            return errors

        # Check for required columns
        required_columns = ["text", "label"]  # Adjust based on your needs
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for duplicate rows
        if df.duplicated().any():
            errors.append("Dataset contains duplicate rows")

        return errors

    def _validate_text_quality(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate text quality."""
        errors = []
        warnings = []
        metrics = {}

        if "text" not in df.columns:
            return errors, warnings, metrics

        text_col = df["text"]

        # Check for null/empty texts
        null_count = text_col.isnull().sum()
        empty_count = (text_col == "").sum()

        if null_count > 0:
            errors.append(f"Found {null_count} null text entries")
        if empty_count > 0:
            errors.append(f"Found {empty_count} empty text entries")

        # Text length validation
        text_lengths = text_col.str.len()
        too_short = (text_lengths < self.min_text_length).sum()
        too_long = (text_lengths > self.max_text_length).sum()

        if too_short > 0:
            warnings.append(
                f"Found {too_short} texts shorter than {self.min_text_length} characters"
            )
        if too_long > 0:
            warnings.append(
                f"Found {too_long} texts longer than {self.max_text_length} characters"
            )

        # Calculate text statistics
        metrics.update(
            {
                "total_texts": len(text_col),
                "null_texts": null_count,
                "empty_texts": empty_count,
                "avg_text_length": text_lengths.mean(),
                "median_text_length": text_lengths.median(),
                "min_text_length": text_lengths.min(),
                "max_text_length": text_lengths.max(),
                "texts_too_short": too_short,
                "texts_too_long": too_long,
            }
        )

        return errors, warnings, metrics

    def _validate_labels(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Validate label distribution and quality."""
        errors = []
        metrics = {}

        if "label" not in df.columns:
            return errors, metrics

        label_col = df["label"]

        # Check for null labels
        null_labels = label_col.isnull().sum()
        if null_labels > 0:
            errors.append(f"Found {null_labels} null labels")

        # Label distribution
        label_counts = label_col.value_counts()
        unique_labels = len(label_counts)

        # Check for class imbalance
        if unique_labels > 1:
            min_class_size = label_counts.min()
            max_class_size = label_counts.max()
            imbalance_ratio = max_class_size / min_class_size

            metrics.update(
                {
                    "num_classes": unique_labels,
                    "label_distribution": label_counts.to_dict(),
                    "min_class_size": min_class_size,
                    "max_class_size": max_class_size,
                    "class_imbalance_ratio": imbalance_ratio,
                }
            )

        return errors, metrics

    def _validate_statistics(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Validate statistical properties."""
        warnings = []
        metrics = {}

        # Dataset size checks
        total_samples = len(df)
        metrics["total_samples"] = total_samples

        if total_samples < 100:
            warnings.append(f"Dataset is quite small ({total_samples} samples)")
        elif total_samples < 1000:
            warnings.append(f"Dataset is relatively small ({total_samples} samples)")

        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        metrics["memory_usage_mb"] = memory_usage

        if memory_usage > 1000:  # 1GB
            warnings.append(f"Dataset is large ({memory_usage:.1f} MB)")

        return warnings, metrics

    def generate_report(
        self, validation_result: ValidationResult, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a human-readable validation report.

        Args:
            validation_result: Result from validate_dataset
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 50,
            "DATA VALIDATION REPORT",
            "=" * 50,
            f"Status: {'✅ VALID' if validation_result.is_valid else '❌ INVALID'}",
            f"Errors: {len(validation_result.errors)}",
            f"Warnings: {len(validation_result.warnings)}",
            "",
        ]

        if validation_result.errors:
            report_lines.extend(["ERRORS:", "-" * 20])
            for error in validation_result.errors:
                report_lines.append(f"❌ {error}")
            report_lines.append("")

        if validation_result.warnings:
            report_lines.extend(["WARNINGS:", "-" * 20])
            for warning in validation_result.warnings:
                report_lines.append(f"⚠️ {warning}")
            report_lines.append("")

        if validation_result.metrics:
            report_lines.extend(["METRICS:", "-" * 20])
            for key, value in validation_result.metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"📊 {key}: {value:.2f}")
                else:
                    report_lines.append(f"📊 {key}: {value}")

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def validate_clinical_data(
    data_path: Path, output_dir: Optional[Path] = None
) -> ValidationResult:
    """
    Convenience function to validate clinical data with default settings.

    Args:
        data_path: Path to the dataset
        output_dir: Optional directory to save validation report

    Returns:
        ValidationResult
    """
    validator = DataValidator()
    result = validator.validate_dataset(data_path)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"validation_report_{data_path.stem}.txt"
        validator.generate_report(result, report_path)

    return result
