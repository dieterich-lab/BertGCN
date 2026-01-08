"""Document-level interpretability for BertGCN (Approach A: neighbor scoring).

Idea: For a target document t and its predicted class c, each incoming neighbor
node j contributes
    score = edge_weight(j→t) * P_c(j)
where P_c(j) is the GCN class probability of neighbor j. We rank neighbors by
this score and return top-k influential documents.

Run:
    poetry run python -m bertgcn.interpret_docs_neighbors
Output:
    outputs/train_gcn/interpret/document_influence.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def _load_model(cfg: DictConfig, n_classes: int, n_features: int) -> BertGCN:
    model = BertGCN(
        pretrained_model=cfg.hparams.model_name_or_path,
        nb_class=n_classes,
        m=cfg.gcn.mix_factor,
        gcn_layers=cfg.gcn.gcn_layers,
        n_hidden=cfg.gcn.n_hidden,
        dropout=cfg.gcn.dropout,
    )
    # Load checkpoint from the default final model path
    project_root = Path(get_original_cwd())
    ckpt_path = project_root / "models" / "final_model" / "pytorch_model.bin"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _compute_neighbor_scores(
    probs: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, top_k: int
) -> List[List[Tuple[int, float]]]:
    """Compute top-k influential neighbors per node for its predicted class.

    probs: [N, C] softmax probabilities from GCN.
    edge_index: [2, E]
    edge_weight: [E]
    Returns list of length N with (neighbor_id, score) sorted desc.
    """

    n_nodes = probs.size(0)
    pred_class = probs.argmax(dim=1)
    # build adjacency mapping from edge_index: incoming edges to target (dst)
    incoming = [[] for _ in range(n_nodes)]
    src, dst = edge_index
    for s, d, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
        incoming[d].append((s, w))

    results: List[List[Tuple[int, float]]] = []
    for node in range(n_nodes):
        c = pred_class[node].item()
        neigh_scores: List[Tuple[int, float]] = []
        for neigh, w in incoming[node]:
            score = w * float(probs[neigh, c])
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

    with torch.no_grad():
        log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        probs = torch.exp(log_probs)

    neighbor_scores = _compute_neighbor_scores(
        probs.cpu(), data["edge_index"].cpu(), data["edge_weight"].cpu(), top_k
    )

    # Prepare output
    rows: List[Dict] = []
    for doc_id, neigh_list in enumerate(neighbor_scores):
        rows.append(
            {
                "doc_id": doc_id,
                "pred_label": label_encoder.inverse_transform(
                    [probs[doc_id].argmax().item()]
                )[0],
                "top_neighbors": [n for n, _ in neigh_list],
                "neighbor_scores": [s for _, s in neigh_list],
            }
        )

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "outputs" / "train_gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "document_influence.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Document-level influence saved to {out_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_document_influence(cfg)


if __name__ == "__main__":
    main()
