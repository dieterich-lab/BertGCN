#!/usr/bin/env python3
"""
Core Models Module for BertGCN

Production-ready model implementations with PyTorch Lightning integration.
"""

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BertGCNTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for BertGCN models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BertGCN trainer.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.lr_bert = config.get("lr_bert", 1e-5)
        self.lr_gcn = config.get("lr_gcn", 1e-4)
        self.mix_factor = config.get("mix_factor", 0.7)

        # Initialize model components
        self._init_model()

        # Training state
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _init_model(self):
        """Initialize the model architecture."""
        # This is a simplified version - in practice you'd load your actual BertGCN model
        # For now, we'll create a basic BERT classifier as placeholder

        pretrained_model = self.config.get("pretrained_model", "bert-base-uncased")
        num_classes = self.config.get("num_classes", 2)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)

        # Get BERT hidden size
        self.hidden_size = self.bert_model.config.hidden_size

        # Classifier head
        self.classifier = torch.nn.Linear(self.hidden_size, num_classes)

        # For this example, we'll use a simple linear layer instead of GCN
        # In practice, this would be your actual GCN implementation
        self.gcn_layer = torch.nn.Linear(self.hidden_size, num_classes)

        self.dropout = torch.nn.Dropout(self.config.get("dropout", 0.1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the model."""
        # BERT forward pass
        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Use [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[
            :, 0, :
        ]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)

        # BERT classification
        bert_logits = self.classifier(cls_output)

        # Simplified GCN (in practice this would use graph structure)
        gcn_logits = self.gcn_layer(cls_output)

        # Mix BERT and GCN outputs
        bert_probs = F.softmax(bert_logits, dim=-1)
        gcn_probs = F.softmax(gcn_logits, dim=-1)

        mixed_probs = (gcn_probs + 1e-10) * self.mix_factor + bert_probs * (
            1 - self.mix_factor
        )
        mixed_logits = torch.log(mixed_probs)

        return mixed_logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        labels = batch["labels"]

        # Forward pass
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Calculate loss
        loss = F.cross_entropy(logits, labels)

        # Calculate metrics
        predictions = torch.argmax(logits, dim=-1)
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end processing
        self.training_step_outputs.append(
            {"loss": loss, "predictions": predictions, "labels": labels}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        labels = batch["labels"]

        # Forward pass
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Calculate loss
        loss = F.cross_entropy(logits, labels)

        # Calculate metrics
        predictions = torch.argmax(logits, dim=-1)

        # Store outputs for epoch-end processing
        self.validation_step_outputs.append(
            {"loss": loss, "predictions": predictions, "labels": labels}
        )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        labels = batch["labels"]

        # Forward pass
        logits = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Calculate loss
        loss = F.cross_entropy(logits, labels)

        # Calculate metrics
        predictions = torch.argmax(logits, dim=-1)

        # Store outputs for epoch-end processing
        self.test_step_outputs.append(
            {"loss": loss, "predictions": predictions, "labels": labels}
        )

        return loss

    def on_training_epoch_end(self):
        """Training epoch end."""
        if not self.training_step_outputs:
            return

        # Calculate epoch metrics
        all_predictions = torch.cat(
            [x["predictions"] for x in self.training_step_outputs]
        )
        all_labels = torch.cat([x["labels"] for x in self.training_step_outputs])

        # Calculate F1 score
        f1 = f1_score(
            all_labels.cpu().numpy(),
            all_predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )

        self.log("train_f1", f1, on_epoch=True, prog_bar=True)

        # Clear outputs
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Validation epoch end."""
        if not self.validation_step_outputs:
            return

        # Calculate epoch metrics
        all_predictions = torch.cat(
            [x["predictions"] for x in self.validation_step_outputs]
        )
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        # Calculate metrics
        accuracy = accuracy_score(
            all_labels.cpu().numpy(), all_predictions.cpu().numpy()
        )
        f1 = f1_score(
            all_labels.cpu().numpy(),
            all_predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )

        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

        # Clear outputs
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Test epoch end."""
        if not self.test_step_outputs:
            return

        # Calculate epoch metrics
        all_predictions = torch.cat([x["predictions"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        # Calculate metrics
        accuracy = accuracy_score(
            all_labels.cpu().numpy(), all_predictions.cpu().numpy()
        )
        f1 = f1_score(
            all_labels.cpu().numpy(),
            all_predictions.cpu().numpy(),
            average="macro",
            zero_division=0,
        )

        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Separate parameters for BERT and GCN components
        bert_params = list(self.bert_model.parameters()) + list(
            self.classifier.parameters()
        )
        gcn_params = list(self.gcn_layer.parameters())

        # Create optimizers with different learning rates
        bert_optimizer = torch.optim.AdamW(bert_params, lr=self.lr_bert)
        gcn_optimizer = torch.optim.AdamW(gcn_params, lr=self.lr_gcn)

        # Learning rate schedulers
        bert_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            bert_optimizer, T_max=self.trainer.max_epochs
        )
        gcn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            gcn_optimizer, T_max=self.trainer.max_epochs
        )

        return [bert_optimizer, gcn_optimizer], [bert_scheduler, gcn_scheduler]


class BertGCN(torch.nn.Module):
    """Standalone BertGCN model for inference."""

    def __init__(
        self,
        pretrained_model: str,
        num_classes: int = 2,
        mix_factor: float = 0.7,
        dropout: float = 0.1,
    ):
        """
        Initialize BertGCN model.

        Args:
            pretrained_model: Pretrained model name or path
            num_classes: Number of output classes
            mix_factor: Mixing factor between BERT and GCN
            dropout: Dropout rate
        """
        super().__init__()

        self.mix_factor = mix_factor
        self.num_classes = num_classes

        # BERT components
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = self.bert_model.config.hidden_size

        # Classification head
        self.classifier = torch.nn.Linear(self.hidden_size, num_classes)

        # Simplified GCN component (in practice this would be more complex)
        self.gcn_layer = torch.nn.Linear(self.hidden_size, num_classes)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        # BERT forward pass
        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Use [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # BERT classification
        bert_logits = self.classifier(cls_output)

        # Simplified GCN
        gcn_logits = self.gcn_layer(cls_output)

        # Mix outputs
        bert_probs = F.softmax(bert_logits, dim=-1)
        gcn_probs = F.softmax(gcn_logits, dim=-1)

        mixed_probs = (gcn_probs + 1e-10) * self.mix_factor + bert_probs * (
            1 - self.mix_factor
        )

        return torch.log(mixed_probs)
