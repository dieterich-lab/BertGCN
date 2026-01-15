"""Prediction script for BertGCN GCN model on test set."""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def load_gcn_model(cfg: DictConfig, n_classes: int):
    """Load the trained GCN model from MLflow artifact."""
    import logging

    logger = logging.getLogger("predict_gcn")

    client = mlflow.tracking.MlflowClient()
    exp_name = "train_gcn"  # Should match the GCN experiment name
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"No MLflow experiment named {exp_name} found.")
    runs = client.search_runs(
        exp.experiment_id,
        "attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished GCN runs found in MLflow.")
    run = runs[0]

    # Log GCN run info
    run_date = run.info.start_time if hasattr(run.info, "start_time") else None
    import datetime

    if run_date:
        run_date_str = datetime.datetime.fromtimestamp(run_date / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    else:
        run_date_str = "unknown"

    gcn_params = {}
    if run.data and run.data.params:
        for k, v in run.data.params.items():
            if any(
                param in k
                for param in [
                    "mix_factor",
                    "gcn_layers",
                    "n_hidden",
                    "dropout",
                    "lr",
                    "epochs",
                    "batch_size",
                ]
            ):
                gcn_params[k] = v

    logger.info(
        f"Loaded trained GCN from MLflow run_id={run.info.run_id}, experiment={exp_name}"
    )
    logger.info(f"GCN training date: {run_date_str}")
    if gcn_params:
        logger.info(f"GCN hyperparameters: {gcn_params}")

    artifact_path = "final_model"
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path
    )
    # Load model state dict
    state_dict = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu")
    # Load config for model params
    # Use cfg for model params
    model = BertGCN(
        pretrained_model=cfg.hparams.model_name_or_path,
        nb_class=n_classes,
        m=cfg.gcn.mix_factor,
        gcn_layers=cfg.gcn.gcn_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_gcn(cfg: DictConfig):
    print("Starting predict_gcn", flush=True)
    # Load dataset and label encoder
    print("Loading processed dataset", flush=True)
    dataset, label_encoder = load_processed_dataset(cfg)
    print("Loaded processed dataset", flush=True)
    n_classes = len(label_encoder.classes_)
    # Load graph data
    print("Loading graph data", flush=True)
    data = load_graph_data_from_disk(cfg)
    print("Loaded graph data", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Using device: {device}", flush=True)
    model = load_gcn_model(cfg, n_classes).to(device)
    print("Loaded model", flush=True)
    # Get test indices
    test_mask = data["test_mask"]
    test_indices = torch.where(test_mask)[0]
    print(f"Test indices: {len(test_indices)}", flush=True)
    # Run prediction in batches
    batch_size = 16
    preds = []
    probs = []
    print("Starting prediction loop", flush=True)
    with torch.no_grad():
        for i in range(0, len(test_indices), batch_size):
            print(f"Batch {i//batch_size}", flush=True)
            batch_idx = test_indices[i : i + batch_size]
            input_ids_batch = data["input_ids"][batch_idx]
            attention_mask_batch = data["attention_mask"][batch_idx]
            out = model(
                data["features"].to(device),
                data["edge_index"].to(device),
                data["edge_weight"].to(device),
                input_ids_batch.to(device),
                attention_mask_batch.to(device),
                batch_idx.to(device),
            )
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            preds.extend(pred)
            probs.extend(prob)
    print("Prediction loop done", flush=True)
    # Prepare output
    output_data = []
    for i, (pred, prob, idx) in enumerate(zip(preds, probs, test_indices)):
        row = {
            "index": int(idx),
            "predicted_label": label_encoder.inverse_transform([pred])[0],
            "predicted_class": int(pred),
        }
        for j, p in enumerate(prob):
            row[f"prob_class_{j}"] = float(p)
        true_label = int(data["labels"][idx])
        row["true_label"] = label_encoder.inverse_transform([true_label])[0]
        row["true_class"] = true_label
        output_data.append(row)
    df = pd.DataFrame(output_data)
    output_file = Path(cfg.inference.output_file)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def main():
    import hydra

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def _main(cfg: DictConfig):
        OmegaConf.set_struct(cfg, False)
        predict_gcn(cfg)

    _main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("\n\n==== EXCEPTION OCCURRED ====")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
