"""Document-level influence via SHAP-style edge perturbation (Approach C).

Idea: For a target doc and its predicted class, iteratively drop each incoming
edge (neighbor j → target) and measure the drop in target-class probability.
The probability delta is the neighbor's importance (SHAP-like, leave-one-out
edge perturbation).

Usage:
    poetry run python -m bertgcn.interpret_docs_shap
Output:
    outputs/train_gcn/interpret/document_influence_shap.csv
Config:
    interpretation.top_k (default 5), interpretation.max_docs (optional)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def _resolve_model_dir(cfg: DictConfig) -> Path:
    """Pick model_dir in priority: cfg.interpretation.model_dir -> newest outputs/gcn/**/final_model -> models/final_model."""

    project_root = Path(get_original_cwd())
    interp = cfg.get("interpretation", {}) if hasattr(cfg, "get") else {}
    explicit = interp.get("model_dir") if isinstance(interp, dict) else None
    if explicit:
        return (project_root / explicit).expanduser().resolve()

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

    model = _load_model(
        cfg, n_classes=n_classes, n_features=data["features"].shape[1]
    ).to(device)

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
    target_nodes = range(n_nodes if max_docs is None else min(max_docs, n_nodes))

    rows: List[Dict] = []
    for t in target_nodes:
        c = pred_class[t].item()
        base_prob = probs[t, c].item()

        contribs: List[Tuple[int, float]] = []
        for neigh, w, edge_idx in incoming[t]:
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

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "outputs" / "train_gcn" / "interpret"
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
