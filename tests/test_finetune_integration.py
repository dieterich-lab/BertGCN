import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
from datasets import Dataset

import bertgcn.finetune_bert as fb


def make_dummy_trainer(out_dir):
    class DummyTrainer:
        def __init__(self):
            self._out = Path(out_dir)

        def train(self):
            # create a dummy model file to simulate saving
            self._out.mkdir(parents=True, exist_ok=True)
            (self._out / "pytorch_model.bin").write_text("dummy")
            return SimpleNamespace()

        def evaluate(self, *args, **kwargs):
            return {"f1": 0.0}

        def predict(self, ds):
            # Return deterministic predictions (zeros)
            preds = np.zeros((len(ds), 1))
            labels = (
                np.array(ds["labels"])
                if "labels" in ds.column_names
                else np.zeros(len(ds), dtype=int)
            )
            return SimpleNamespace(predictions=preds, label_ids=labels)

        def save_model(self, out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "saved_model.bin").write_text("saved")

    return DummyTrainer()


def test_finetune_dryrun_monkeypatched(tmp_path, monkeypatch):
    # Prepare tiny HF dataset with required columns
    texts = ["foo bar", "baz qux"]
    input_ids = [[1, 2, 0], [3, 4, 0]]
    attention_mask = [[1, 1, 0], [1, 1, 0]]
    labels = [0, 1]
    med_id = [0, 1]
    ds = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "med_id": med_id,
        }
    )
    ds = ds.with_format("torch")

    # Monkeypatch load_processed_dataset to return our small ds and a simple LabelEncoder-like object
    class DummyLE:
        def __init__(self):
            self.classes_ = np.array(["A", "B"])

    def fake_load_processed_dataset(cfg):
        return ds, DummyLE()

    monkeypatch.setattr(fb, "load_processed_dataset", fake_load_processed_dataset)

    # Monkeypatch AutoTokenizer and model to simple stand-ins
    class SimpleTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text('{"tok":true}')

        def tokenize(self, text):
            return text.split()

    monkeypatch.setattr(
        fb,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda p: SimpleTokenizer()),
    )

    # Simple torch model that returns logits
    class SimpleModel(torch.nn.Module):
        def __init__(self, num_labels=2):
            super().__init__()
            self.fc = torch.nn.Linear(3, num_labels)
            self.config = SimpleNamespace(
                id2label={0: "A", 1: "B"}, label2id={"A": 0, "B": 1}
            )

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            bs = input_ids.shape[0]
            logits = torch.randn(bs, 2)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        fb,
        "AutoModelForSequenceClassification",
        SimpleNamespace(
            from_pretrained=lambda p, num_labels=None: SimpleModel(
                num_labels=num_labels
            )
        ),
    )

    # Monkeypatch setup_trainer to return a dummy trainer that will save a model file
    def fake_setup_trainer(
        model, tokenizer, train_ds, val_ds, out_dir, cfg, class_weights=None
    ):
        return make_dummy_trainer(out_dir)

    monkeypatch.setattr(fb, "setup_trainer", fake_setup_trainer)

    # Redirect MLflow tracking to tmp and ensure experiment name exists
    tmp_mlruns = tmp_path / "mlruns"
    tmp_mlruns.mkdir()
    orig_set_uri = mlflow.set_tracking_uri
    mlflow.set_tracking_uri(str(tmp_mlruns))

    # Build a minimal cfg matching what main expects
    cfg = {
        "hparams": {
            "seed": 42,
            "model_name_or_path": "dummy",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "num_train_epochs": 1,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "eval_steps": 1,
            "use_stratified_split": False,
            "use_class_weights": False,
        },
        "hydra": {"run": {"dir": "outputs/tmp_run"}},
        "mlflow_experiment_name": "test_integration",
    }

    # Convert to OmegaConf DictConfig
    from omegaconf import OmegaConf

    dict_cfg = OmegaConf.create(cfg)

    # Call the wrapped main function (bypass hydra decorator runtime)
    fb.main.__wrapped__(dict_cfg)

    # After run, check that mlruns has experiment folder and a run with artifacts
    experiments = list(tmp_mlruns.iterdir())
    assert len(experiments) > 0

    # Reset MLflow tracking uri
    mlflow.set_tracking_uri(
        orig_set_uri.__self__ if hasattr(orig_set_uri, "__self__") else str(tmp_mlruns)
    )
