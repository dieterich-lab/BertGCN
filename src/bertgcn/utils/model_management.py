#!/usr/bin/env python3
"""
Model Management Module for BertGCN

Provides model lifecycle management, versioning, and deployment utilities.
"""

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    version: str
    stage: str
    uri: str
    metrics: Dict[str, float]
    tags: Dict[str, str]
    creation_time: datetime
    size_mb: float


class ModelManager:
    """Comprehensive model management for BertGCN."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the model manager.

        Args:
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

    def list_models(self, model_name: Optional[str] = None) -> List[ModelInfo]:
        """
        List all available models.

        Args:
            model_name: Optional model name filter

        Returns:
            List of ModelInfo objects
        """
        try:
            if model_name:
                registered_models = [self.client.get_registered_model(model_name)]
            else:
                registered_models = self.client.search_registered_models()

            models = []
            for rm in registered_models:
                for version in rm.latest_versions:
                    # Get metrics
                    run = self.client.get_run(version.run_id)
                    metrics = run.data.metrics

                    # Calculate model size (approximate)
                    artifacts = self.client.list_artifacts(version.run_id)
                    size_mb = sum(
                        artifact.file_size or 0
                        for artifact in artifacts
                        if artifact.file_size
                    ) / (1024 * 1024)

                    model_info = ModelInfo(
                        name=rm.name,
                        version=version.version,
                        stage=version.current_stage,
                        uri=f"models:/{rm.name}/{version.version}",
                        metrics=metrics,
                        tags=version.tags,
                        creation_time=datetime.fromtimestamp(
                            version.creation_timestamp / 1000
                        ),
                        size_mb=size_mb,
                    )
                    models.append(model_info)

            return sorted(models, key=lambda x: x.creation_time, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise

    def get_model_info(self, model_uri: str) -> ModelInfo:
        """
        Get detailed information about a specific model.

        Args:
            model_uri: Model URI (e.g., "models:/model_name/version")

        Returns:
            ModelInfo object
        """
        try:
            # Parse URI
            if model_uri.startswith("models:/"):
                parts = model_uri.split("/")
                model_name = parts[1]
                version = parts[2] if len(parts) > 2 else "latest"
            else:
                raise ValueError(f"Unsupported model URI format: {model_uri}")

            # Get model version
            if version == "latest":
                model_version = self.client.get_latest_versions(model_name)[0]
            else:
                model_version = self.client.get_model_version(model_name, version)

            # Get run info
            run = self.client.get_run(model_version.run_id)

            # Calculate size
            artifacts = self.client.list_artifacts(model_version.run_id)
            size_mb = sum(
                artifact.file_size or 0 for artifact in artifacts if artifact.file_size
            ) / (1024 * 1024)

            return ModelInfo(
                name=model_name,
                version=model_version.version,
                stage=model_version.current_stage,
                uri=model_uri,
                metrics=run.data.metrics,
                tags=model_version.tags,
                creation_time=datetime.fromtimestamp(
                    model_version.creation_timestamp / 1000
                ),
                size_mb=size_mb,
            )

        except Exception as e:
            logger.error(f"Failed to get model info for {model_uri}: {str(e)}")
            raise

    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """
        Promote a model to a specific stage.

        Args:
            model_name: Name of the model
            version: Version to promote
            stage: Target stage (Staging, Production, Archived)

        Returns:
            True if successful
        """
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")

        try:
            # If promoting to Production, archive current production models
            if stage == "Production":
                current_prod = self.client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                for prod_model in current_prod:
                    self.client.transition_model_version_stage(
                        model_name, prod_model.version, "Archived"
                    )
                    logger.info(
                        f"Archived previous production model: {model_name} v{prod_model.version}"
                    )

            # Promote the new model
            self.client.transition_model_version_stage(model_name, version, stage)
            logger.info(f"Promoted {model_name} v{version} to {stage}")

            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {str(e)}")
            raise

    def delete_model_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.

        Args:
            model_name: Name of the model
            version: Version to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted {model_name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model version: {str(e)}")
            raise

    def compare_models(
        self, model_uris: List[str], metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models across specified metrics.

        Args:
            model_uris: List of model URIs to compare
            metrics: List of metric names to compare

        Returns:
            Dictionary with comparison results
        """
        comparison = {}

        for uri in model_uris:
            try:
                model_info = self.get_model_info(uri)
                comparison[f"{model_info.name}_v{model_info.version}"] = {
                    metric: model_info.metrics.get(metric, 0.0) for metric in metrics
                }
            except Exception as e:
                logger.warning(f"Failed to get metrics for {uri}: {str(e)}")
                comparison[uri] = {metric: 0.0 for metric in metrics}

        return comparison

    def export_model(
        self, model_uri: str, output_path: Path, format: str = "pytorch"
    ) -> Path:
        """
        Export a model to a local path.

        Args:
            model_uri: Model URI to export
            output_path: Local path to save the model
            format: Export format (pytorch, onnx, etc.)

        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            if format == "pytorch":
                # Download the model
                model = mlflow.pytorch.load_model(model_uri)

                # Save as PyTorch state dict
                model_path = output_path / "model.pt"
                torch.save(model.state_dict(), model_path)

                # Save model info
                info_path = output_path / "model_info.json"
                model_info = self.get_model_info(model_uri)

                import json

                with open(info_path, "w") as f:
                    json.dump(
                        {
                            "name": model_info.name,
                            "version": model_info.version,
                            "stage": model_info.stage,
                            "metrics": model_info.metrics,
                            "export_date": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                    )

                logger.info(f"Model exported to {output_path}")
                return model_path

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            raise

    def cleanup_old_models(self, model_name: str, keep_versions: int = 5) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.

        Args:
            model_name: Name of the model to clean
            keep_versions: Number of versions to keep

        Returns:
            Number of versions deleted
        """
        try:
            # Get all versions
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            # Sort by creation time (newest first)
            sorted_versions = sorted(
                model_versions, key=lambda x: x.creation_timestamp, reverse=True
            )

            # Keep production and staging models
            protected_stages = ["Production", "Staging"]
            protected_versions = [
                v for v in sorted_versions if v.current_stage in protected_stages
            ]

            # Get versions to delete (excluding protected ones)
            versions_to_delete = []
            kept_count = 0

            for version in sorted_versions:
                if version.current_stage in protected_stages:
                    continue

                if kept_count < keep_versions:
                    kept_count += 1
                else:
                    versions_to_delete.append(version)

            # Delete old versions
            deleted_count = 0
            for version in versions_to_delete:
                try:
                    self.client.delete_model_version(model_name, version.version)
                    deleted_count += 1
                    logger.info(f"Deleted {model_name} v{version.version}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete {model_name} v{version.version}: {str(e)}"
                    )

            logger.info(
                f"Cleanup completed: {deleted_count} versions deleted, {kept_count + len(protected_versions)} kept"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup models: {str(e)}")
            raise
