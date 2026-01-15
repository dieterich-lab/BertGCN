"""Document-level influence via Integrated Gradients (Approach B).

Idea: Run IG on the full document feature matrix for a target doc and its
predicted class; aggregate IG attributions per document (sum over feature
dimensions) and rank documents by importance. Self-attribution is zeroed before
ranking.

Usage:
    poetry run python -m bertgcn.interpret_docs_ig
Output:
    hydra/gcn/interpret/document_influence_ig.csv
Config:
    interpretation.top_k (default 5), interpretation.max_docs (optional)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def _resolve_model_dir(cfg: DictConfig) -> Path:
    """Pick model_dir in priority: cfg.interpretation.model_dir -> newest hydra/gcn/**/final_model -> models/final_model."""

    project_root = Path(get_original_cwd())
    interp = cfg.get("interpretation", {}) if hasattr(cfg, "get") else {}
    explicit = interp.get("model_dir") if isinstance(interp, dict) else None
    if explicit:
        return (project_root / explicit).expanduser().resolve()

    try:
        candidates = sorted(
            (p for p in project_root.glob("hydra/gcn/**/final_model")),
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
    model = BertGCN(
        pretrained_model=cfg.hparams.model_name_or_path,
        nb_class=n_classes,
        m=cfg.gcn.mix_factor,
        gcn_layers=cfg.gcn.gcn_layers,
        n_hidden=cfg.gcn.n_hidden,
        dropout=cfg.gcn.dropout,
    )
    model_dir = _resolve_model_dir(cfg)
    ckpt_path = model_dir / "pytorch_model.bin"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    model.eval()
    return model


def integrated_gradients(
    model: BertGCN,
    data: Dict[str, torch.Tensor],
    target_idx: int,
    target_class: int,
    steps: int = 16,
) -> torch.Tensor:
    """Compute IG attributions over all document features for a target doc/class.

    Returns: attribution tensor shaped like features (N, D).
    """

    features = data["features"].detach()
    baseline = torch.zeros_like(features)
    edge_index = data["edge_index"]
    edge_weight = data.get("edge_weight")

    total_grad = torch.zeros_like(features)
    for alpha in torch.linspace(0.0, 1.0, steps):
        x = (baseline + alpha * (features - baseline)).clone().requires_grad_(True)
        # Forward only for GCN
        log_probs = model.gcn(x, edge_index, edge_weight)
        logit = log_probs[target_idx, target_class]
        logit.backward(retain_graph=True)
        total_grad += x.grad.detach()

    avg_grad = total_grad / steps
    ig = (features - baseline) * avg_grad
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

    # Model
    model = _load_model(
        cfg, n_classes=n_classes, n_features=data["features"].shape[1]
    ).to(device)

    # Predictions
    with torch.no_grad():
        log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        probs = torch.exp(log_probs)
        pred_class = probs.argmax(dim=1)

    n_nodes = data["features"].shape[0]
    target_nodes = range(n_nodes if max_docs is None else min(max_docs, n_nodes))

    rows: List[Dict] = []
    for idx in target_nodes:
        c = pred_class[idx].item()
        ig = integrated_gradients(model, data, idx, c)  # (N, D)
        # aggregate per document
        doc_scores = ig.sum(dim=1).cpu()
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

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "hydra" / "gcn" / "interpret"
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
