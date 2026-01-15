import os
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf

import bertgcn.train_bert as fb


@pytest.fixture()
def tiny_dataset():
    texts = ["a b", "c d", "e f"]
    input_ids = [[1, 2, 0], [3, 4, 0], [5, 6, 0]]
    attention_mask = [[1, 1, 0], [1, 1, 0], [1, 1, 0]]
    labels = [0, 1, 0]
    med_id = [0, 1, 2]
    ds = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "med_id": med_id,
        }
    )
    return ds.with_format("torch")


class DummyLE:
    def __init__(self):
        self.classes_ = np.array(["A", "B"])


def dummy_model(num_labels=2):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, num_labels)
            self.config = type(
                "Cfg", (), {"id2label": {0: "A", 1: "B"}, "label2id": {"A": 0, "B": 1}}
            )()

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            bs = input_ids.shape[0]
            logits = torch.zeros(bs, 2)
            return type("Out", (), {"logits": logits})()

    return M()


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path, monkeypatch):
    ml_path = tmp_path / "mlruns"
    ml_path.mkdir()
    mlflow.set_tracking_uri(str(ml_path))
    yield


@pytest.fixture
def monkeypatch_hf(monkeypatch, tiny_dataset):
    def fake_load(cfg):
        return tiny_dataset, DummyLE()

    monkeypatch.setattr(fb, "load_processed_dataset", fake_load)
    monkeypatch.setattr(
        fb,
        "AutoTokenizer",
        type(
            "TokWrap",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda p: type(
                        "Tok",
                        (),
                        {
                            "save_pretrained": lambda self, p: Path(p).mkdir(
                                parents=True, exist_ok=True
                            )
                        },
                    )()
                )
            },
        ),
    )
    monkeypatch.setattr(
        fb,
        "AutoModelForSequenceClassification",
        type(
            "ModWrap",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda p, num_labels=None: dummy_model(num_labels)
                )
            },
        ),
    )

    def fake_setup(
        model, tokenizer, train_ds, val_ds, out_dir, cfg, class_weights=None
    ):
        class T:
            def train(self):
                # simulate a checkpoint file inside out_dir
                Path(out_dir, "checkpoint-1").mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "checkpoint-1" / "pytorch_model.bin").write_text(
                    "ckpt"
                )

            def evaluate(self, *a, **k):
                return {"f1": 0.0}

            def predict(self, ds):
                preds = np.zeros((len(ds), 2))
                labels = np.array([0] * len(ds))
                return type("Pred", (), {"predictions": preds, "label_ids": labels})()

            def save_model(self, out_dir):
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "saved_model.bin").write_text("model")

        return T()

    monkeypatch.setattr(fb, "setup_trainer", fake_setup)


@pytest.mark.parametrize("override_dir", [None, "custom/run_dir/test_case"])
def test_script_controlled_run_dir_and_no_local_dup(
    monkeypatch_hf, tmp_path, override_dir
):
    # Build minimal cfg - note: hydra dir config is now ignored by script
    base_cfg = {
        "hparams": {
            "seed": 1,
            "model_name_or_path": "dummy",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "num_train_epochs": 1,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "eval_steps": 1,
            "use_stratified_split": False,
            "use_class_weights": False,
            "keep_local_copy": False,
        },
        "mlflow_experiment_name": "test_script_controlled_behavior",
        # Hydra config is present but ignored by script
        "hydra": {"run": {"dir": "ignored/hydra/path"}},
    }
    cfg = OmegaConf.create(base_cfg)

    # Execute main (bypassing hydra decorator)
    fb.main.__wrapped__(cfg)

    # Script should create multirun/bert_* directory regardless of hydra config
    multirun_dir = Path("multirun")
    assert multirun_dir.exists(), "Expected multirun directory to be created by script"

    bert_dirs = list(multirun_dir.glob("bert_*"))
    assert bert_dirs, "Expected at least one bert_* directory in multirun"
    run_dir = bert_dirs[0]  # Use the first one found

    # Check checkpoint presence
    ckpts = list(run_dir.glob("checkpoint-*/pytorch_model.bin"))
    assert ckpts, "Expected at least one checkpoint file in run dir"

    # Evaluation subfolder should exist; confusion_matrix.json should NOT be local
    eval_dir = run_dir / "evaluation"
    assert eval_dir.exists(), "Expected evaluation subfolder to be created"
    assert not (
        eval_dir / "confusion_matrix.json"
    ).exists(), "confusion_matrix.json should not be persisted locally (logged in-memory to MLflow)"

    # Ensure no extra top-level model save when keep_local_copy=False and autolog path taken
    extraneous = list(Path(".").glob("model")) + list(Path(".").glob("saved_model.bin"))
    assert (
        not extraneous
    ), f"Unexpected extraneous model artifacts at repo root: {extraneous}"
