#!/usr/bin/env python3
"""
Inference Pipeline for BertGCN

Production-ready inference pipeline with:
- Batch prediction support
- Model caching
- Performance monitoring
- Error handling
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import torch
from transformers import AutoTokenizer

from bertgcn.core.models import BertGCN
from bertgcn.utils.monitoring import log_prediction_metrics

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Production inference pipeline for BertGCN models."""

    def __init__(self, model_uri: str, device: Optional[str] = None):
        """
        Initialize the inference pipeline.

        Args:
            model_uri: MLflow model URI or local path to model
            device: Device to run inference on (cpu, cuda, auto)
        """
        self.model_uri = model_uri
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_uri}")

            # Load model from MLflow or local path
            if self.model_uri.startswith("models:/") or self.model_uri.startswith(
                "runs:/"
            ):
                self.model = mlflow.pytorch.load_model(self.model_uri)
            else:
                # Load from local path
                checkpoint = torch.load(self.model_uri, map_location=self.device)
                self.model = (
                    checkpoint["model"] if "model" in checkpoint else checkpoint
                )

            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer (assuming it's saved with the model)
            # For now, use a default one - this would be improved in production
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Make predictions on input texts.

        Args:
            texts: Single text or list of texts to classify
            return_probabilities: Whether to return class probabilities
            batch_size: Batch size for processing

        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Tokenize texts
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move to device
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            predictions = []
            probabilities = []

            # Process in batches
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_input_ids = input_ids[i : i + batch_size]
                    batch_attention_mask = attention_mask[i : i + batch_size]

                    # For this example, we'll use a simple forward pass
                    # In practice, this would depend on your specific model architecture
                    outputs = self.model(batch_input_ids, batch_attention_mask)

                    if return_probabilities:
                        probs = torch.softmax(outputs, dim=-1)
                        probabilities.extend(probs.cpu().numpy())

                    preds = torch.argmax(outputs, dim=-1)
                    predictions.extend(preds.cpu().numpy())

            inference_time = time.time() - start_time

            # Log metrics
            log_prediction_metrics(
                num_samples=len(texts),
                inference_time=inference_time,
                model_uri=self.model_uri,
            )

            result = {
                "predictions": predictions,
                "num_samples": len(texts),
                "inference_time": inference_time,
                "model_uri": self.model_uri,
            }

            if return_probabilities:
                result["probabilities"] = probabilities

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """
        Batch prediction method for large datasets.

        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing

        Returns:
            Dictionary containing batch predictions
        """
        return self.predict(texts, batch_size=batch_size)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_uri": self.model_uri,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }


class BatchInferencePipeline(InferencePipeline):
    """Specialized pipeline for batch inference on large datasets."""

    def __init__(
        self, model_uri: str, device: Optional[str] = None, max_batch_size: int = 64
    ):
        super().__init__(model_uri, device)
        self.max_batch_size = max_batch_size

    def process_dataset(
        self, texts: List[str], output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Process a large dataset in batches.

        Args:
            texts: List of texts to process
            output_path: Optional path to save results

        Returns:
            Dictionary containing all predictions and metadata
        """
        logger.info(
            f"Processing {len(texts)} samples in batches of {self.max_batch_size}"
        )

        all_predictions = []
        total_time = 0

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i : i + self.max_batch_size]
            logger.info(
                f"Processing batch {i//self.max_batch_size + 1}/{(len(texts)-1)//self.max_batch_size + 1}"
            )

            batch_results = self.predict(batch_texts, batch_size=self.max_batch_size)
            all_predictions.extend(batch_results["predictions"])
            total_time += batch_results["inference_time"]

        results = {
            "predictions": all_predictions,
            "total_samples": len(texts),
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(texts),
            "model_uri": self.model_uri,
        }

        # Save results if output path specified
        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results
