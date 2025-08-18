from torch.utils.tensorboard import SummaryWriter
import optuna
from ray import tune
import hydra
from hydra import compose, initialize
import subprocess
import sys

import hydra
import lime
import mlflow
import mlflow.pytorch
import shap
from omegaconf import OmegaConf


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return None


def log_environment():
    import platform

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "packages": sys.modules.keys(),
    }
    mlflow.log_dict(env, "environment.json")


"""
Fine-tune a Hugging Face Transformer model for sequence classification.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.training_args import IntervalStrategy

from bertgcn.clinic_datasets import CleanClinicDataset
from bertgcn.core import get_logger, setup_environment
from bertgcn.params import BertGCNParameters

logger = get_logger(__name__)


def load_or_create_dataset(
    tokenizer, model_name: str, doc_level: str, task: str = "MIC", clean: bool = False
) -> CleanClinicDataset:
    dataset_file = Path("data") / f"{task.lower()}_{model_name}_{doc_level}.pkl"
    if dataset_file.exists():
        logger.info(f"Loading dataset from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("Creating dataset")
        dataset = CleanClinicDataset(
            tokenizer=tokenizer, task=task, doclevel=doc_level, clean=clean
        )
        os.makedirs(dataset_file.parent, exist_ok=True)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
    return dataset


def split_dataset(dataset, test_unclear: bool = False) -> Tuple[Subset, Subset, Subset]:
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    if not test_unclear:
        train_idx = idx[: int(len(idx) * 0.7)]
        val_idx = idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)]
        test_idx = idx[int(len(idx) * 0.8) :]
    else:
        train_val_idx, test_idx = [], []
        for i, x in enumerate(dataset):
            if "unklar" in dataset.LE.classes_[x["labels"]]:
                test_idx.append(i)
            else:
                train_val_idx.append(i)
        np.random.shuffle(train_val_idx)
        split_idx = int(len(train_val_idx) * 0.9)
        train_idx = train_val_idx[:split_idx]
        val_idx = train_val_idx[split_idx:]
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "accuracy": np.mean(predictions == labels),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def save_metrics(trainer, dataset, output_dir: Union[str, Path], split: str = "test"):
    from sklearn.metrics import classification_report, confusion_matrix

    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    class_names = getattr(dataset.dataset, "LE", None)
    if class_names is not None:
        class_names = class_names.classes_
    else:
        class_names = [str(i) for i in range(len(np.unique(labels)))]
    cr = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )
    cr_df = pd.DataFrame(cr).transpose()
    cm = confusion_matrix(labels, preds)
    os.makedirs(output_dir, exist_ok=True)
    cr_df.to_csv(os.path.join(output_dir, f"{split}_classification_report.csv"))
    np.save(os.path.join(output_dir, f"{split}_confusion_matrix.npy"), cm)
    logger.info(f"\n{cr_df.to_string()}")
    logger.info(f"\nConfusion Matrix:\n{cm}")


def main(
    doclevel: str = typer.Option("letter", help="Document level"),
    bertmodel: str = typer.Option("medbert", help="BERT model"),
    window_size: int = typer.Option(20, help="Window size"),
    batch_size: int = typer.Option(1000, help="Batch size"),
    bidirectional_tfidf: bool = typer.Option(True, help="Bidirectional TF-IDF"),
    min_pmi: float = typer.Option(0.0, help="Minimum PMI"),
    seed: int = typer.Option(42, help="Random seed"),
    testunklar: bool = typer.Option(False, help="Test unclear samples"),
    model_name_or_path: str = typer.Option(
        "bert-base-uncased", help="Model name or path"
    ),
    output_dir: str = typer.Option("outputs", help="Output directory"),
    eval_steps: int = typer.Option(100, help="Evaluation steps"),
    save_steps: int = typer.Option(100, help="Save steps"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    num_train_epochs: int = typer.Option(3, help="Number of training epochs"),
    nepochs: int = typer.Option(3, help="Number of epochs (fallback)"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    fp16: bool = typer.Option(False, help="Use FP16"),
    gradient_accumulation_steps: int = typer.Option(
        1, help="Gradient accumulation steps"
    ),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio"),
    patience: int = typer.Option(3, help="Early stopping patience"),
    testonly: bool = typer.Option(False, help="Test only, skip training"),
):
    # Hydra config management (hierarchical)
    config_dict = {
        "doclevel": doclevel,
        "bertmodel": bertmodel,
        "window_size": window_size,
        "batch_size": batch_size,
        "bidirectional_tfidf": bidirectional_tfidf,
        "min_pmi": min_pmi,
        "seed": seed,
        "testunklar": testunklar,
        "model_name_or_path": model_name_or_path,
        "output_dir": output_dir,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "nepochs": nepochs,
        "weight_decay": weight_decay,
        "fp16": fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "patience": patience,
        "testonly": testonly,
    }
    config = OmegaConf.create(config_dict)
    OmegaConf.save(config, "run_config.yaml")

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard"))

    args = BertGCNParameters(
        doclevel=doclevel,
        bertmodel=bertmodel,
        window_size=window_size,
        batch_size=batch_size,
        bidirectional_tfidf=bidirectional_tfidf,
        min_pmi=min_pmi,
        seed=seed,
        testunklar=testunklar,
    )
    setup_environment(args.seed)
    set_seed(args.seed)
    now_str = Path().joinpath("logs", "finetune", args.doclevel)
    os.makedirs(now_str, exist_ok=True)
    logger.info(f"Starting fine-tuning with parameters: {OmegaConf.to_yaml(config)}")
    model_name = Path(model_name_or_path).name
    output_dir = Path(output_dir) / args.doclevel / f"{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Start MLflow experiment
    mlflow.set_experiment(f"BertGCN_{args.doclevel}")
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        log_environment()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        dataset = load_or_create_dataset(tokenizer, model_name, args.doclevel)
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, args.testunklar
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=len(dataset.LE.classes_)
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=eval_steps,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=save_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=num_train_epochs or nepochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            logging_dir=now_str,
            logging_steps=50,
            save_total_limit=3,
            report_to=["tensorboard"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        if not testonly:
            logger.info("Starting training")
            trainer.train()
            logger.info(f"Saving best model to {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.pytorch.log_model(model, "model")
        logger.info("Evaluating on test set")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
        mlflow.log_metrics(test_results)
        save_metrics(trainer, test_dataset, output_dir, "test")
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)
        # TensorBoard logging
        for k, v in test_results.items():
            tb_writer.add_scalar(f"test/{k}", v)
        tb_writer.flush()
        # Explainability (SHAP)
        try:
            explainer = shap.Explainer(model, tokenizer)
            shap_values = explainer([dataset[0]["input_ids"]])
            shap.summary_plot(shap_values, show=False)
            shap.save_html(os.path.join(output_dir, "shap_summary.html"), shap_values)
            mlflow.log_artifact(os.path.join(output_dir, "shap_summary.html"))
        except Exception as e:
            logger.warning(f"SHAP explainability failed: {e}")
        # Explainability (LIME)
        try:
            from lime.lime_text import LimeTextExplainer
            lime_explainer = LimeTextExplainer(class_names=list(dataset.LE.classes_))
            lime_exp = lime_explainer.explain_instance(
                dataset[0]["text"],
                lambda x: model(tokenizer(x, return_tensors="pt")["input_ids"]).logits.detach().numpy(),
            )
            lime_exp.save_to_file(os.path.join(output_dir, "lime_explanation.html"))
            mlflow.log_artifact(os.path.join(output_dir, "lime_explanation.html"))
        except Exception as e:
            logger.warning(f"LIME explainability failed: {e}")
        logger.info("Fine-tuning completed")
# Example Optuna sweep function (for hyperparameter optimization)
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # ...existing code for training loop using these hyperparameters...
    # Return validation metric
    return 0.0  # Replace with actual validation score

# Example Ray Tune sweep function
def tune_train(config):
    # ...existing code for training loop using config...
    tune.report(val_f1=0.0)  # Replace with actual validation score
        # Explainability (SHAP)
        try:
            explainer = shap.Explainer(model, tokenizer)
            shap_values = explainer([dataset[0]["input_ids"]])
            shap.summary_plot(shap_values, show=False)
            shap.save_html(os.path.join(output_dir, "shap_summary.html"), shap_values)
            mlflow.log_artifact(os.path.join(output_dir, "shap_summary.html"))
        except Exception as e:
            logger.warning(f"SHAP explainability failed: {e}")
        # Explainability (LIME)
        try:
            from lime.lime_text import LimeTextExplainer

            lime_explainer = LimeTextExplainer(class_names=list(dataset.LE.classes_))
            lime_exp = lime_explainer.explain_instance(
                dataset[0]["text"],
                lambda x: model(tokenizer(x, return_tensors="pt")["input_ids"])
                .logits.detach()
                .numpy(),
            )
            lime_exp.save_to_file(os.path.join(output_dir, "lime_explanation.html"))
            mlflow.log_artifact(os.path.join(output_dir, "lime_explanation.html"))
        except Exception as e:
            logger.warning(f"LIME explainability failed: {e}")
        logger.info("Fine-tuning completed")


if __name__ == "__main__":
    typer.run(main)
