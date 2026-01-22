"""Document-level influence via Integrated Gradients (Approach B).

Idea: Run IG on the full document feature matrix for a target doc and its
predicted class; aggregate IG attributions per document (sum over feature
dimensions) and rank documents by importance. Self-attribution is zeroed before
ranking.

Usage:
    poetry run python -m bertgcn.interpret_docs_ig
Output:
    outputs/gcn/interpret/document_influence_ig.csv
Config:
    interpretation.top_k (default 5), interpretation.max_docs (optional)
"""

import sys

# Enable line buffering for stdout to ensure real-time log output
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger

logger = get_logger(__name__)
from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


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


def integrated_gradients(
    model: BertGCN,
    data: Dict[str, torch.Tensor],
    target_idx: int,
    target_class: int,
    steps: int = 16,
    debug: bool = False,
) -> torch.Tensor:
    """Compute IG attributions over all document features for a target doc/class.

    Returns: attribution tensor shaped like features (N, D).
    """

    features = data["features"].detach()
    baseline = torch.zeros_like(features)
    edge_index = data["edge_index"]
    edge_weight = data.get("edge_weight")

    if debug:
        print(
            f"Debug: features mean={features.mean().item():.6f}, std={features.std().item():.6f}, shape={features.shape}"
        )
        print(f"Debug: features[0][:10] = {features[0][:10].tolist()}")

    total_grad = torch.zeros_like(features)
    for i, alpha in enumerate(torch.linspace(0.0, 1.0, steps)):
        x = (baseline + alpha * (features - baseline)).clone().requires_grad_(True)
        # Forward only for GCN
        log_probs = model.gcn(x, edge_index, edge_weight)
        logit = log_probs[target_idx, target_class]
        logit.backward(retain_graph=True)
        if debug and i == 0:
            print(
                f"Debug: alpha={alpha.item():.3f}, logit={logit.item():.6f}, x.grad sum={x.grad.sum().item():.6f}"
            )
        total_grad += x.grad.detach()

    avg_grad = total_grad / steps
    ig = (features - baseline) * avg_grad
    if debug:
        print(
            f"Debug: avg_grad mean={avg_grad.mean().item():.6f}, std={avg_grad.std().item():.6f}"
        )
        print(
            f"Debug: ig abs sum={ig.abs().sum().item():.6f}, ig sum={ig.sum().item():.6f}"
        )
    return ig


def run_document_ig(cfg: DictConfig):
    top_k = cfg.get("interpretation", {}).get("top_k", 5)
    max_docs = cfg.get("interpretation", {}).get("max_docs", None)

    # Data
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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Decode texts from input_ids
    texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in dataset["input_ids"]
    ]

    # Model
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

    # Predictions
    with torch.no_grad():
        log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        probs = torch.exp(log_probs)
        pred_class = probs.argmax(dim=1)

    n_nodes = data["features"].shape[0]
    target_nodes = range(n_docs if max_docs is None else min(max_docs, n_docs))

    rows: List[Dict] = []
    for i, idx in enumerate(target_nodes):
        c = pred_class[idx].item()
        ig = integrated_gradients(model, data, idx, c, debug=(idx < 3))  # (N, D)
        # aggregate per document
        doc_scores = ig.sum(dim=1).cpu()
        if idx < 3:
            top_vals, top_inds = doc_scores.topk(10)
            print(
                f"Debug idx {idx}: doc_scores max={doc_scores.max().item():.10f}, top 5 values: {[f'{v:.10f}' for v in top_vals.tolist()[:5]]}, inds: {top_inds.tolist()[:5]}",
                flush=True,
            )
        # zero out self to avoid trivial self-importance dominating
        doc_scores[idx] = 0.0
        # take top_k
        vals, inds = torch.topk(doc_scores, k=min(top_k, len(doc_scores)))
        rows.append(
            {
                "doc_id": idx,
                "pred_label": label_encoder.inverse_transform([c])[0],
                "top_neighbors": inds.tolist(),
                "neighbor_scores": [float(v) for v in vals],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"Processed IG for {i + 1}/{len(target_nodes)} documents", flush=True)

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "document_influence_ig.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Document-level IG influence saved to {out_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_document_ig(cfg)


if __name__ == "__main__":
    main()
