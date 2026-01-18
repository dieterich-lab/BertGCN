"""Document-level influence via SHAP-style edge perturbation (Approach C).

Idea: For a target doc and its predicted class, iteratively drop each incoming
edge (neighbor j → target) and measure the drop in target-class probability.
The probability delta is the neighbor's importance (SHAP-like, leave-one-out
edge perturbation).

Usage:
    poetry run python -m bertgcn.interpret_docs_shap
Output:
    outputs/gcn/interpret/document_influence_shap.csv
Config:
    interpretation.top_k (default 5), interpretation.max_docs (optional)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger
from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset

logger = get_logger(__name__)


def _resolve_model_dir(cfg: DictConfig) -> Path:
    """Pick model_dir in priority: cfg.interpretation.model_dir -> MLflow artifacts -> hydra/gcn/**/final_model."""

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        # Fallback when not running under Hydra
        project_root = Path.cwd()

    interp = cfg.get("interpretation", {}) if hasattr(cfg, "get") else {}
    explicit = interp.get("model_dir") if isinstance(interp, dict) else None
    if explicit:
        model_dir = (project_root / explicit).expanduser().resolve()
        print(f"Using explicitly configured model directory: {model_dir}")
        return model_dir

    # Try to find latest model from MLflow artifacts
    logger.info("Looking for latest GCN model in MLflow...")
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        exp_name = "train_gcn"
        exp = client.get_experiment_by_name(exp_name)
        if exp is not None:
            runs = client.search_runs(
                exp.experiment_id,
                "attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                run = runs[0]
                artifact_path = "final_model"
                model_dir = Path(
                    mlflow.artifacts.download_artifacts(
                        run_id=run.info.run_id, artifact_path=artifact_path
                    )
                )
                logger.info(f"Found GCN model in MLflow run {run.info.run_id[:8]}...")
                return model_dir
    except Exception as e:
        logger.warning(f"MLflow search failed: {e}")

    # Fallback to hydra/gcn/**/final_model
    try:
        candidates = sorted(
            (p for p in project_root.glob("hydra/gcn/**/final_model")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        candidates = []

    fallback = project_root / "models" / "final_model"
    for cand in candidates:
        if cand.is_dir():
            logger.info(f"Using fallback model directory: {cand}")
            return cand
    logger.warning(
        f"No model directory found, checked MLflow and local hydra dirs. Please ensure a model is available."
    )
    raise FileNotFoundError(
        "No valid model directory found. Run training first or specify model_dir in config."
    )


def _load_model(cfg: DictConfig, n_classes: int, n_features: int) -> BertGCN:
    # Load complete BERT+GCN model from clean MLflow artifact structure
    logger.info("Loading complete BERT+GCN model from MLflow artifacts...")
    model_dir = _resolve_model_dir(cfg)

    try:
        from transformers import AutoModel, AutoTokenizer

        # Load BERT model and tokenizer using standard HF loading
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bert_model = AutoModel.from_pretrained(model_dir)
        feat_dim = bert_model.config.hidden_size

        # Load GCN components from the single state dict file
        gcn_checkpoint = torch.load(model_dir / "gcn_components.pt", map_location="cpu")
        m = gcn_checkpoint["m"]
        nb_class = gcn_checkpoint["nb_class"]

        # Create BertGCN model using the "loading from saved state" path
        model = BertGCN(
            bert_model=bert_model,
            tokenizer=tokenizer,
            feat_dim=feat_dim,
            nb_class=nb_class,
            m=m,
            gcn_layers=1,  # dummy value, will be overridden
            n_hidden=cfg.gcn.n_hidden,
            dropout=cfg.gcn.dropout,
        )

        # Load the saved weights
        model.classifier.load_state_dict(gcn_checkpoint["classifier"])
        model.gcn.load_state_dict(gcn_checkpoint["gcn"])

        logger.info("✓ Loaded complete BERT+GCN model from clean MLflow artifacts")
        model.eval()
        return model

    except Exception as e:
        logger.error(f"Failed to load from MLflow artifacts: {e}")
        raise


def run_document_shap(cfg: DictConfig):
    top_k = cfg.get("interpretation", {}).get("top_k", 5)
    max_docs = cfg.get("interpretation", {}).get("max_docs", None)

    dataset, label_encoder = load_processed_dataset(cfg)
    n_classes = len(label_encoder.classes_)
    data = load_graph_data_from_disk(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    # Load tokenizer to decode texts
    model_dir = _resolve_model_dir(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # Decode texts from input_ids
    texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in dataset["input_ids"]
    ]

    model = _load_model(
        cfg, n_classes=n_classes, n_features=data["features"].shape[1]
    ).to(device)

    # Compute correct features from BERT
    print("Computing BERT features for documents...", flush=True)
    model = model.to(device)
    features_list = []
    for i, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(device)
        with torch.no_grad():
            outputs = model.bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        features_list.append(embedding.cpu())
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(texts)} documents for features", flush=True)
    features_doc = torch.stack(features_list)
    n_docs = len(dataset)
    n_nodes = data["features"].shape[0]
    features = torch.zeros(n_nodes, 768)
    features[:n_docs] = features_doc
    data["features"] = features.to(device)

    with torch.no_grad():
        log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        probs = torch.exp(log_probs)
        pred_class = probs.argmax(dim=1)

    # Build incoming edges
    src, dst = data["edge_index"]
    edge_w = data["edge_weight"]
    incoming = [[] for _ in range(data["features"].shape[0])]
    for i, (s, d, w) in enumerate(zip(src.tolist(), dst.tolist(), edge_w.tolist())):
        incoming[d].append((s, w, i))  # (neighbor, weight, edge_idx)

    n_nodes = data["features"].shape[0]
    target_nodes = range(n_docs if max_docs is None else min(max_docs, n_docs))

    rows: List[Dict] = []
    for i, t in enumerate(target_nodes):
        c = pred_class[t].item()
        base_prob = probs[t, c].item()

        contribs: List[Tuple[int, float]] = []
        for neigh, w, edge_idx in incoming[t]:
            if neigh >= n_docs:  # Skip edges from word nodes
                continue
            # Temporarily drop the edge by zeroing its weight
            saved_w = data["edge_weight"][edge_idx].item()
            data["edge_weight"][edge_idx] = 0.0
            with torch.no_grad():
                pert_log = model.gcn(
                    data["features"], data["edge_index"], data.get("edge_weight")
                )
                pert_prob = torch.exp(pert_log)[t, c].item()
            data["edge_weight"][edge_idx] = saved_w
            delta = base_prob - pert_prob  # importance of this neighbor
            contribs.append((neigh, delta))

        contribs.sort(key=lambda x: x[1], reverse=True)
        top = contribs[:top_k]
        rows.append(
            {
                "doc_id": t,
                "pred_label": label_encoder.inverse_transform([c])[0],
                "top_neighbors": [n for n, _ in top],
                "neighbor_scores": [float(s) for _, s in top],
            }
        )
        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(target_nodes)} documents for SHAP", flush=True
            )

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "document_influence_shap.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Document-level SHAP-like influence saved to {out_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_document_shap(cfg)


if __name__ == "__main__":
    main()
