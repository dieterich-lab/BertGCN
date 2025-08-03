#!/usr/bin/env python3
"""
Modern MLOps Training Pipeline for BertGCN

This module provides a production-ready training pipeline with:
- Experiment tracking with MLflow
- Hyperparameter optimization
- Model versioning and registry
- Automated model validation
- Performance monitoring
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import mlflow
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from rich.console import Console
from rich.table import Table

from bertgcn.core.models import BertGCNTrainer
from bertgcn.core.data import DataModule
from bertgcn.core.metrics import ModelMetrics
from bertgcn.utils.notifications import send_training_alert
from bertgcn.utils.model_validation import ModelValidator

console = Console()
logger = logging.getLogger(__name__)


class MLOpsTrainingPipeline:
    """Production MLOps training pipeline."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.experiment_id = None
        self.run_id = None
        self.metrics = ModelMetrics()
        self.validator = ModelValidator()
        
    def setup_experiment_tracking(self) -> None:
        """Setup MLflow experiment tracking."""
        mlflow.set_tracking_uri(self.config.logging.mlflow.tracking_uri)
        
        experiment_name = f"{self.config.experiment.name}_{self.config.data.name}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            self.run_id = run.info.run_id
            
            # Log configuration
            mlflow.log_params(OmegaConf.to_container(self.config, resolve=True))
            
            # Log experiment metadata
            mlflow.set_tags({
                "version": self.config.experiment.version,
                "description": self.config.experiment.description,
                "model_type": "BertGCN",
                "data_type": self.config.data.name,
                **{f"tag_{i}": tag for i, tag in enumerate(self.config.experiment.tags)}
            })
    
    def create_loggers(self) -> list:
        """Create experiment loggers."""
        loggers = []
        
        # MLflow logger
        if self.config.logging.mlflow.enabled:
            mlflow_logger = MLFlowLogger(
                experiment_name=f"{self.config.experiment.name}_{self.config.data.name}",
                tracking_uri=self.config.logging.mlflow.tracking_uri,
                log_model=True
            )
            loggers.append(mlflow_logger)
        
        # Weights & Biases logger
        if self.config.logging.wandb.enabled:
            wandb_logger = WandbLogger(
                project=self.config.experiment.name,
                name=f"run_{int(time.time())}",
                tags=self.config.experiment.tags
            )
            loggers.append(wandb_logger)
            
        return loggers
    
    def create_callbacks(self) -> list:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.config.paths.models_dir) / "checkpoints",
            filename="{epoch:02d}-{val_f1:.3f}",
            monitor=self.config.training.checkpointing.monitor,
            mode=self.config.training.checkpointing.mode,
            save_top_k=self.config.training.checkpointing.save_top_k,
            save_last=self.config.training.checkpointing.save_last,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.training.early_stopping.monitor,
            patience=self.config.training.early_stopping.patience,
            mode=self.config.training.early_stopping.mode,
            min_delta=self.config.training.early_stopping.min_delta,
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        # Rich progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
        
        return callbacks
    
    def train_model(self) -> Dict[str, Any]:
        """Train the BertGCN model."""
        console.print("[bold blue]🚀 Starting BertGCN Training Pipeline[/bold blue]")
        
        try:
            # Setup experiment tracking
            self.setup_experiment_tracking()
            
            # Create data module
            data_module = DataModule(self.config.data)
            data_module.setup()
            
            # Create model
            model = BertGCNTrainer(self.config.model)
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.config.training.trainer.max_epochs,
                accelerator="auto",
                devices="auto",
                precision=self.config.training.trainer.precision,
                callbacks=self.create_callbacks(),
                logger=self.create_loggers(),
                log_every_n_steps=self.config.training.trainer.log_every_n_steps,
                val_check_interval=self.config.training.trainer.val_check_interval,
                enable_progress_bar=self.config.training.trainer.enable_progress_bar,
            )
            
            # Training
            console.print("[yellow]📚 Training model...[/yellow]")
            trainer.fit(model, data_module)
            
            # Testing
            console.print("[yellow]🧪 Testing model...[/yellow]")
            test_results = trainer.test(model, data_module)
            
            # Model validation
            validation_results = self.validator.validate_model(
                model, data_module.test_dataloader()
            )
            
            # Calculate final metrics
            final_metrics = self.metrics.calculate_final_metrics(
                test_results, validation_results
            )
            
            # Log final metrics
            with mlflow.start_run(run_id=self.run_id):
                for metric_name, metric_value in final_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Register model if validation passes
                if validation_results["passes_validation"]:
                    self._register_model(model, final_metrics)
            
            # Send success notification
            send_training_alert(
                status="success",
                metrics=final_metrics,
                config=self.config
            )
            
            return {
                "status": "success",
                "metrics": final_metrics,
                "model_uri": f"runs:/{self.run_id}/model",
                "experiment_id": self.experiment_id,
                "run_id": self.run_id
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            
            # Send failure notification
            send_training_alert(
                status="failure",
                error=str(e),
                config=self.config
            )
            
            raise e
    
    def _register_model(self, model: pl.LightningModule, metrics: Dict[str, Any]) -> None:
        """Register model in MLflow Model Registry."""
        model_name = f"bertgcn_{self.config.data.name}"
        
        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name,
            extra_files=["configs/"],
        )
        
        # Add model version description
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(
            model_name, stages=["None"]
        )[0]
        
        client.update_model_version(
            name=model_name,
            version=latest_version.version,
            description=f"BertGCN model trained on {self.config.data.name} dataset. "
                       f"Val F1: {metrics.get('val_f1', 0):.3f}, "
                       f"Test F1: {metrics.get('test_f1', 0):.3f}"
        )
        
        console.print(f"[green]✅ Model registered: {model_name} v{latest_version.version}[/green]")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Set random seeds
    pl.seed_everything(config.seed.global)
    
    # Create training pipeline
    pipeline = MLOpsTrainingPipeline(config)
    
    # Train model
    results = pipeline.train_model()
    
    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for metric, value in results["metrics"].items():
        table.add_row(metric, f"{value:.4f}" if isinstance(value, float) else str(value))
    
    console.print(table)
    console.print(f"[green]🎉 Training completed successfully![/green]")
    console.print(f"[blue]📊 Experiment ID: {results['experiment_id']}[/blue]")
    console.print(f"[blue]🏃 Run ID: {results['run_id']}[/blue]")


if __name__ == "__main__":
    main()