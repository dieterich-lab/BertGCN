import torch
from transformers import (
    AutoModelForSequenceClassification,
    DistilBertConfig,
    Trainer,
    TrainingArguments,
)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size: int, seq_len: int, vocab_size: int, num_labels: int):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_labels = num_labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        labels = torch.tensor(idx % self.num_labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def data_collator(batch):
    # stack tensors to form batch
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def test_trainer_runs_one_step(tmp_path):
    import mlflow

    # Ensure MLflow uses a local temporary tracking directory for the test and
    # create/set a test experiment so the FileStore has a valid experiment id.
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(str(mlruns_dir))

    # Ensure the default experiment exists (FileStore expects experiment id 0)
    try:
        mlflow.set_experiment("Default")
    except Exception:
        pass

    exp_name = "test_smoke_experiment"
    if mlflow.get_experiment_by_name(exp_name) is None:
        # create_experiment returns the new experiment id
        mlflow.create_experiment(exp_name, artifact_location=str(mlruns_dir))
    mlflow.set_experiment(exp_name)

    # Tiny model config
    config = DistilBertConfig(
        vocab_size=32,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        seq_classif_dropout=0.2,
        num_labels=3,
    )
    model = AutoModelForSequenceClassification.from_config(config)

    train_ds = RandomDataset(size=8, seq_len=16, vocab_size=32, num_labels=3)

    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        max_steps=1,
        learning_rate=1e-4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    # Should not raise and should perform a single optimization step
    trainer.train()
    # Basic smoke assertions
    assert trainer.state.global_step == 1
    assert trainer.args.output_dir is not None
