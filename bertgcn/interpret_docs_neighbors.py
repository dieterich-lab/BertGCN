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
from bertgcn.train_gcn import load_graph_data_from_disk, load_processed_dataset

logger = get_logger(__name__)

try:
    import numpy as np
    import scipy.sparse as sp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, falling back to slower implementation")


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
        logger.info(f"Using explicitly configured model directory: {model_dir}")
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
                model_dir = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path=artifact_path
                )
                if Path(model_dir).is_dir():
                    logger.info(
                        f"Found GCN model in MLflow run {run.info.run_id[:8]}..."
                    )
                    return Path(model_dir)
    except Exception as e:
        logger.warning(f"Failed to load from MLflow: {e}")

    # Fallback: look for models in hydra/gcn/**/final_model
    logger.info("Checking local model directories...")
    candidates = []
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
            logger.info(f"Using local model directory: {cand}")
            return cand
    logger.warning(
        f"No model directory found, checked MLflow and local hydra dirs. Please ensure a model is available."
    )
    raise FileNotFoundError(
        "No valid model directory found. Run training first or specify model_dir in config."
    )


def _load_model(cfg: DictConfig, n_classes: int, n_features: int):
    # Load complete BERT+GCN model from clean MLflow artifact structure
    logger.info("Loading complete BERT+GCN model from MLflow artifacts...")
    model_dir = _resolve_model_dir(cfg)

    try:
        from transformers import AutoModel, AutoTokenizer

        from bertgcn.train_gcn import BertGCN

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
        logger.error(f"✗ Failed to load complete BERT+GCN model from artifacts: {e}")
        logger.error("This likely means the model was saved with an old format")
        logger.error("Please retrain the GCN model to save complete artifacts")
        raise RuntimeError(
            "Model artifacts are incomplete or corrupted. Please retrain with the updated saving code."
        ) from e


def _compute_2hop_neighbor_scores_optimized(
    probs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    top_k: int,
    node_indices: List[int] = None,
    doc_mask: torch.Tensor = None,
) -> List[List[Tuple[int, float]]]:
    """Optimized version: Compute top-k influential 2-hop neighbors per document node.

    Uses sparse matrices and vectorization for ~10-100x speedup.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, using original implementation")
        return _compute_2hop_neighbor_scores(
            probs, edge_index, edge_weight, top_k, node_indices, doc_mask
        )

    n_nodes = probs.size(0)
    pred_class = probs.argmax(dim=1)

    # Convert to numpy for scipy operations
    edge_index_np = edge_index.numpy()
    edge_weight_np = edge_weight.numpy()
    probs_np = probs.numpy()
    doc_mask_np = doc_mask.numpy() if doc_mask is not None else None

    logger.info("Building sparse adjacency matrix...")
    # Create sparse adjacency matrix (n_nodes x n_nodes)
    adj = sp.csr_matrix(
        (edge_weight_np, (edge_index_np[0], edge_index_np[1])), shape=(n_nodes, n_nodes)
    )

    if node_indices is None:
        node_indices = list(range(n_nodes))

    # Filter to document nodes only
    if doc_mask_np is not None:
        doc_nodes = np.where(doc_mask_np)[0]
        node_indices = [n for n in node_indices if n in doc_nodes]

    logger.info(f"Computing 2-hop scores for {len(node_indices)} document nodes...")

    results: List[List[Tuple[int, float]]] = []

    # Pre-compute word nodes (non-document nodes)
    if doc_mask_np is not None:
        word_nodes = np.where(~doc_mask_np)[0]
    else:
        word_nodes = np.arange(n_nodes)

    # Process documents in batches for memory efficiency
    batch_size = 100
    for batch_start in range(0, len(node_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(node_indices))
        batch_nodes = node_indices[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_start//batch_size + 1}: nodes {batch_start}-{batch_end-1}"
        )

        # For this batch of documents, compute all 2-hop paths
        batch_scores = {}  # doc_id -> {hop2_doc -> score}

        for doc_idx in batch_nodes:
            c = pred_class[doc_idx].item()
            batch_scores[doc_idx] = {}

            # Get 1-hop neighbors (words) - only outgoing edges from this doc
            hop1_weights = adj[doc_idx, word_nodes].toarray().flatten()
            connected_words = word_nodes[hop1_weights > 0]
            hop1_weights = hop1_weights[hop1_weights > 0]

            if len(connected_words) == 0:
                continue

            # For each connected word, get its outgoing edges to documents
            for word_idx, w1 in zip(connected_words, hop1_weights):
                # Get 2-hop neighbors (documents) from this word
                hop2_weights = adj[word_idx, :].toarray().flatten()
                connected_docs = np.where(
                    (hop2_weights > 0)
                    & (
                        doc_mask_np
                        if doc_mask_np is not None
                        else np.ones(n_nodes, dtype=bool)
                    )
                )[0]
                hop2_weights = hop2_weights[connected_docs]

                # Exclude self
                valid_mask = connected_docs != doc_idx
                connected_docs = connected_docs[valid_mask]
                hop2_weights = hop2_weights[valid_mask]

                if len(connected_docs) == 0:
                    continue

                # Vectorized score calculation: w1 * w2 * P_c(d2)
                path_scores = w1 * hop2_weights * probs_np[connected_docs, c]

                # Aggregate scores for each 2-hop document
                for hop2_doc, score in zip(connected_docs, path_scores):
                    if hop2_doc in batch_scores[doc_idx]:
                        batch_scores[doc_idx][hop2_doc] += score
                    else:
                        batch_scores[doc_idx][hop2_doc] = score

        # Convert batch results to sorted lists
        for doc_idx in batch_nodes:
            hop2_list = [
                (doc_id, score) for doc_id, score in batch_scores[doc_idx].items()
            ]
            hop2_list.sort(key=lambda x: x[1], reverse=True)
            results.append(hop2_list[:top_k])

        if (batch_end) % 500 == 0 or batch_end == len(node_indices):
            logger.info(
                f"Processed {batch_end}/{len(node_indices)} nodes for 2-hop scoring"
            )

    logger.info(f"Completed optimized 2-hop scoring for {len(results)} nodes")
    return results


def _compute_2hop_neighbor_scores(
    probs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    top_k: int,
    node_indices: List[int] = None,
    doc_mask: torch.Tensor = None,
) -> List[List[Tuple[int, float]]]:
    """Compute top-k influential 2-hop neighbors per document node.

    For each document t, find paths t -> w -> d2 where w is a word and d2 is another document.
    Score each d2 based on the path importance.

    probs: [N, C] softmax probabilities from GCN.
    edge_index: [2, E]
    edge_weight: [E]
    node_indices: document nodes to analyze (should all be documents)
    doc_mask: boolean mask indicating which nodes are documents
    Returns list of length len(node_indices) with (2hop_doc_id, score) sorted desc.
    """

    n_nodes = probs.size(0)
    pred_class = probs.argmax(dim=1)

    # Build adjacency mapping: outgoing edges from source (src) to target (dst)
    logger.info("Building adjacency mapping for 2-hop computation...")
    outgoing = [[] for _ in range(n_nodes)]
    src, dst = edge_index
    for s, d, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
        outgoing[s].append((d, w))

    if node_indices is None:
        node_indices = list(range(n_nodes))

    logger.info(f"Computing 2-hop scores for {len(node_indices)} nodes...")
    results: List[List[Tuple[int, float]]] = []
    for idx, node in enumerate(node_indices):
        if doc_mask is not None and not doc_mask[node]:
            # Skip non-document nodes
            results.append([])
            continue

        c = pred_class[node].item()
        hop2_scores: Dict[int, float] = {}  # doc_id -> aggregated_score

        # Get 1-hop neighbors (should be words for documents)
        hop1_neighbors = outgoing[node]

        for hop1_node, w1 in hop1_neighbors:
            if doc_mask is not None and doc_mask[hop1_node]:
                continue  # Skip document-document edges, focus on word neighbors

            # Get 2-hop neighbors from this intermediate word
            hop2_neighbors = outgoing[hop1_node]

            for hop2_node, w2 in hop2_neighbors:
                if doc_mask is None or not doc_mask[hop2_node]:
                    continue  # Only consider document nodes as 2-hop targets
                if hop2_node == node:
                    continue  # Don't count self as 2-hop

                # Calculate 2-hop score: path weight * probability of target document
                path_score = w1 * w2 * float(probs[hop2_node, c])

                # Aggregate scores for each unique 2-hop document
                if hop2_node in hop2_scores:
                    hop2_scores[hop2_node] += path_score
                else:
                    hop2_scores[hop2_node] = path_score

        # Convert to sorted list of tuples
        hop2_list = [(doc_id, score) for doc_id, score in hop2_scores.items()]
        hop2_list.sort(key=lambda x: x[1], reverse=True)
        results.append(hop2_list[:top_k])

        # Log progress every 100 nodes
        if (idx + 1) % 100 == 0:
            logger.info(
                f"Processed {idx + 1}/{len(node_indices)} nodes for 2-hop scoring"
            )

    logger.info(f"Completed 2-hop scoring for {len(results)} nodes")
    return results


def run_document_influence(cfg: DictConfig):
    # Defaults
    top_k = cfg.get("interpretation", {}).get("top_k", 5)

    logger.info(f"Starting document influence analysis with top_k={top_k}")

    # Load data
    logger.info("Loading processed dataset...")
    dataset, label_encoder = load_processed_dataset(cfg)
    n_classes = len(label_encoder.classes_)
    logger.info(
        f"Loaded dataset with {len(dataset)} samples, {n_classes} classes: {list(label_encoder.classes_)}"
    )

    logger.info("Loading graph data from disk...")
    data = load_graph_data_from_disk(cfg)
    logger.info(
        f"Loaded graph with {data['features'].shape[0]} nodes, {data['edge_index'].shape[1]} edges"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    data = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    # Load model
    logger.info("Loading BertGCN model...")
    model = _load_model(
        cfg, n_classes=n_classes, n_features=data["features"].shape[1]
    ).to(device)
    logger.info("Model loaded successfully")

    # Only analyze document nodes (exclude word nodes)
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    doc_indices = torch.where(doc_mask.cpu())[0]
    logger.info(
        f"Analyzing {len(doc_indices)} document nodes out of {doc_mask.shape[0]} total nodes"
    )

    # Get GCN probabilities for the full graph (needed for neighbor scoring)
    logger.info("Computing GCN probabilities for full graph...")
    with torch.no_grad():
        gcn_log_probs = model.gcn(
            data["features"], data["edge_index"], data.get("edge_weight")
        )
        gcn_probs = torch.exp(gcn_log_probs).cpu()
    logger.info("GCN probabilities computed")

    # Run batch-wise hybrid predictions for document predictions
    batch_size = 64
    all_hybrid_probs = []
    logger.info(f"Computing hybrid BERT+GCN predictions in batches of {batch_size}...")

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

            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(doc_indices):
                logger.info(
                    f"Processed {min(i + batch_size, len(doc_indices))}/{len(doc_indices)} documents"
                )

    # Concatenate all batch predictions
    hybrid_probs = torch.cat(all_hybrid_probs, dim=0)
    logger.info("Hybrid predictions completed")

    # Use GCN probabilities for 2-hop neighbor scoring (full graph context)
    logger.info("Computing 2-hop neighbor scores...")
    neighbor_scores = _compute_2hop_neighbor_scores_optimized(
        gcn_probs,
        data["edge_index"].cpu(),
        data["edge_weight"].cpu(),
        top_k,
        doc_indices.tolist(),
        doc_mask.cpu(),
    )
    logger.info("2-hop neighbor scoring completed")

    # Prepare output - create one row per document-neighbor pair for better analysis
    logger.info("Preparing output data...")
    rows: List[Dict] = []

    # Get ground truth labels if available
    has_labels = hasattr(dataset, "labels") and dataset.labels is not None
    true_labels = dataset.labels if has_labels else None

    for i, doc_idx in enumerate(doc_indices.tolist()):
        neigh_list = neighbor_scores[
            i
        ]  # Use i since neighbor_scores is indexed by position in doc_indices

        # Get source document info
        source_pred_class = hybrid_probs[i].argmax().item()
        source_pred_label = label_encoder.inverse_transform([source_pred_class])[0]
        source_confidence = hybrid_probs[i].max().item()
        source_true_label = (
            label_encoder.inverse_transform([true_labels[doc_idx]])[0]
            if has_labels
            else None
        )

        # Create one row per neighbor (or one row for documents with no neighbors)
        if not neigh_list:
            rows.append(
                {
                    "source_doc_id": doc_idx,
                    "source_pred_label": source_pred_label,
                    "source_pred_confidence": source_confidence,
                    "source_true_label": source_true_label,
                    "neighbor_rank": None,
                    "neighbor_doc_id": None,
                    "neighbor_score": None,
                    "neighbor_pred_label": None,
                    "neighbor_pred_confidence": None,
                    "neighbor_true_label": None,
                    "neighbor_same_class_as_source": None,
                }
            )
        else:
            for rank, (neigh_id, score) in enumerate(neigh_list, 1):
                # Find the position of this neighbor in doc_indices to get hybrid_probs
                try:
                    neigh_pos = doc_indices.tolist().index(neigh_id)
                    neigh_pred_class = hybrid_probs[neigh_pos].argmax().item()
                    neigh_pred_label = label_encoder.inverse_transform(
                        [neigh_pred_class]
                    )[0]
                    neigh_confidence = hybrid_probs[neigh_pos].max().item()
                    neigh_true_label = (
                        label_encoder.inverse_transform([true_labels[neigh_id]])[0]
                        if has_labels
                        else None
                    )
                    same_class = source_pred_class == neigh_pred_class
                except (ValueError, IndexError):
                    # Neighbor not in analyzed documents (shouldn't happen for 2-hop docs)
                    neigh_pred_label = "unknown"
                    neigh_confidence = 0.0
                    neigh_true_label = None
                    same_class = False

                rows.append(
                    {
                        "source_doc_id": doc_idx,
                        "source_pred_label": source_pred_label,
                        "source_pred_confidence": source_confidence,
                        "source_true_label": source_true_label,
                        "neighbor_rank": rank,
                        "neighbor_doc_id": neigh_id,
                        "neighbor_score": score,
                        "neighbor_pred_label": neigh_pred_label,
                        "neighbor_pred_confidence": neigh_confidence,
                        "neighbor_true_label": neigh_true_label,
                        "neighbor_same_class_as_source": same_class,
                    }
                )

    import pandas as pd

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        project_root = Path.cwd()

    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "document_influence_2hop.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    logger.info(
        f"Document-level 2-hop influence saved to {out_file} ({len(rows)} rows, {len(doc_indices)} documents analyzed)"
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_document_influence(cfg)


if __name__ == "__main__":
    main()
