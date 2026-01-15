"""Document-level interpretability for BertGCN (Approach A: neighbor scoring).

Idea: For a target document t and its predicted class c, each incoming neighbor
node j contributes
    score = edge_weight(j→t) * P_c(j)
where P_c(j) is the GCN class probability of neighbor j. We rank neighbors by
this score and return top-k influential documents.

Run:
    poetry run python -m bertgcn.interpret_docs_neighbors
Output:
    outputs/gcn/interpret/document_influence.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.train_gcn import (
    BertGCN,
    SimpleGCN,
    load_graph_data_from_disk,
    load_processed_dataset,
)


def _resolve_model_dir(cfg: DictConfig) -> Path:
    """Pick model_dir in priority: cfg.interpretation.model_dir -> MLflow artifacts -> outputs/gcn/**/final_model -> models/final_model."""

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        # Fallback when not running under Hydra
        project_root = Path.cwd()

    interp = cfg.get("interpretation", {}) if hasattr(cfg, "get") else {}
    explicit = interp.get("model_dir") if isinstance(interp, dict) else None
    if explicit:
        return (project_root / explicit).expanduser().resolve()

    # First, try to find latest model from MLflow artifacts
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        exp_name = "train_gcn"  # Should match the GCN experiment name
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
                model_dir = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path=artifact_path
                )
                if Path(model_dir).is_dir():
                    return Path(model_dir)
    except Exception:
        pass  # Fall back to other methods

    # Fallback: look for models in outputs/gcn/**/final_model
    candidates = []
    try:
        candidates = sorted(
            (p for p in project_root.glob("outputs/gcn/**/final_model")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        candidates = []

    fallback = project_root / "models" / "final_model"
    for cand in candidates + [fallback]:
        if cand.is_dir():
            return cand
    return fallback


def _load_model(cfg: DictConfig, n_classes: int, n_features: int) -> BertGCN:
    # Get model directory from MLflow GCN artifacts
    model_dir = _resolve_model_dir(cfg)

    # Try to load BERT model and tokenizer from GCN artifacts
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        bert_model = AutoModel.from_pretrained(model_dir)
        feat_dim = bert_model.config.hidden_size
        load_from_artifacts = True
    except (ValueError, OSError):
        # Fall back to old method if BERT config not saved in artifacts
        load_from_artifacts = False
        # Get GCN config from MLflow run parameters
        model_config = None
        try:
            import ast

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
                    gcn_params_str = run.data.params.get("gcn")
                    if gcn_params_str:
                        model_config = ast.literal_eval(gcn_params_str)
        except Exception:
            pass

        # Use MLflow config if available, otherwise fall back to cfg
        if model_config:
            gcn_config = model_config
        else:
            gcn_config = cfg.gcn

        # Create model using old method
        model = BertGCN(
            pretrained_model=cfg.hparams.model_name_or_path,
            nb_class=n_classes,
            m=gcn_config.get("mix_factor", cfg.gcn.mix_factor),
            gcn_layers=gcn_config.get("gcn_layers", cfg.gcn.gcn_layers),
            n_hidden=gcn_config.get("n_hidden", cfg.gcn.n_hidden),
            dropout=gcn_config.get("dropout", cfg.gcn.dropout),
        )

        # Load the saved state dict
        ckpt_path = model_dir / "pytorch_model.bin"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        model.eval()
        return model

    # If we get here, load_from_artifacts is True
    # Get GCN config from MLflow run parameters
    model_config = None
    try:
        import ast

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
                gcn_params_str = run.data.params.get("gcn")
                if gcn_params_str:
                    model_config = ast.literal_eval(gcn_params_str)
    except Exception:
        pass

    # Use MLflow config if available, otherwise fall back to cfg
    if model_config:
        gcn_config = model_config
    else:
        gcn_config = cfg.gcn

    # Create BertGCN model manually (loading BERT from artifacts)
    import torch.nn as nn

    model = BertGCN.__new__(BertGCN)  # Create instance without calling __init__
    model.m = gcn_config.get("mix_factor", cfg.gcn.mix_factor)
    model.nb_class = n_classes
    model.tokenizer = tokenizer
    model.bert_model = bert_model
    model.feat_dim = feat_dim
    model.classifier = nn.Linear(feat_dim, n_classes)
    model.gcn = SimpleGCN(
        n_features=feat_dim,
        n_hidden=gcn_config.get("n_hidden", cfg.gcn.n_hidden),
        n_classes=n_classes,
        dropout=gcn_config.get("dropout", cfg.gcn.dropout),
    )

    # Load the saved state dict
    ckpt_path = model_dir / "pytorch_model.bin"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    model.eval()
    return model


def _compute_neighbor_scores(
    probs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    top_k: int,
    node_indices: List[int] = None,
    doc_mask: torch.Tensor = None,
) -> List[List[Tuple[int, float]]]:
    """Compute top-k influential neighbors per node for its predicted class.

    For documents: only consider word neighbors (TF-IDF edges)
    For words: consider both word neighbors (PMI) and document neighbors (TF-IDF)

    probs: [N, C] softmax probabilities from GCN.
    edge_index: [2, E]
    edge_weight: [E]
    node_indices: if provided, only compute for these nodes
    doc_mask: boolean mask indicating which nodes are documents
    Returns list of length len(node_indices) with (neighbor_id, score) sorted desc.
    """

    n_nodes = probs.size(0)
    pred_class = probs.argmax(dim=1)

    # Build adjacency mapping from edge_index: incoming edges to target (dst)
    incoming = [[] for _ in range(n_nodes)]
    src, dst = edge_index
    for s, d, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
        incoming[d].append((s, w))

    if node_indices is None:
        node_indices = list(range(n_nodes))

    results: List[List[Tuple[int, float]]] = []
    for node in node_indices:
        c = pred_class[node].item()
        neigh_scores: List[Tuple[int, float]] = []
        for neigh, w in incoming[node]:
            # For document nodes, only consider word neighbors (TF-IDF edges)
            if doc_mask is not None and doc_mask[node] and doc_mask[neigh]:
                continue  # Skip document-document edges

            if doc_mask is not None and doc_mask[neigh]:  # Document neighbor
                score = w * float(probs[neigh, c])
            else:  # Word neighbor - use edge weight (TF-IDF or PMI)
                score = w
            neigh_scores.append((neigh, score))
        # sort and trim
        neigh_scores.sort(key=lambda x: x[1], reverse=True)
        results.append(neigh_scores[:top_k])
    return results


def run_document_influence(cfg: DictConfig):
    # Defaults
    top_k = cfg.get("interpretation", {}).get("top_k", 5)

    # Load data
    dataset, label_encoder = load_processed_dataset(cfg)
    n_classes = len(label_encoder.classes_)
    data = load_graph_data_from_disk(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    # Load model
    model = _load_model(
        cfg, n_classes=n_classes, n_features=data["features"].shape[1]
    ).to(device)

    # Only analyze document nodes (exclude word nodes)
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    doc_indices = torch.where(doc_mask.cpu())[0]

    # Get GCN probabilities for the full graph (needed for neighbor scoring)
    with torch.no_grad():
        gcn_log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        gcn_probs = torch.exp(gcn_log_probs).cpu()

    # Run batch-wise hybrid predictions for document predictions
    batch_size = 64
    all_hybrid_probs = []

    with torch.no_grad():
        for i in range(0, len(doc_indices), batch_size):
            batch_idx = doc_indices[i : i + batch_size]
            input_ids_batch = data["input_ids"][batch_idx]
            attention_mask_batch = data["attention_mask"][batch_idx]

            # Get hybrid predictions for this batch
            log_probs = model(
                data["features"],
                data["edge_index"],
                data["edge_weight"],
                input_ids_batch,
                attention_mask_batch,
                batch_idx,
            )
            probs = torch.exp(log_probs).cpu()
            all_hybrid_probs.append(probs)

    # Concatenate all batch predictions
    hybrid_probs = torch.cat(all_hybrid_probs, dim=0)

    # Use GCN probabilities for neighbor scoring (full graph context)
    neighbor_scores = _compute_neighbor_scores(
        gcn_probs,
        data["edge_index"].cpu(),
        data["edge_weight"].cpu(),
        top_k,
        doc_indices.tolist(),
        doc_mask.cpu(),
    )

    # Prepare output - only for document nodes
    rows: List[Dict] = []
    for i, doc_idx in enumerate(doc_indices.tolist()):
        neigh_list = neighbor_scores[
            i
        ]  # Use i since neighbor_scores is indexed by position in doc_indices
        rows.append(
            {
                "doc_id": doc_idx,
                "pred_label": label_encoder.inverse_transform(
                    [hybrid_probs[i].argmax().item()]
                )[0],
                "top_neighbors": [n for n, _ in neigh_list],
                "neighbor_scores": [s for _, s in neigh_list],
            }
        )

    import pandas as pd

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        project_root = Path.cwd()

    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "document_influence.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(
        f"Document-level influence saved to {out_file} ({len(rows)} documents analyzed)"
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_document_influence(cfg)


if __name__ == "__main__":
    main()
