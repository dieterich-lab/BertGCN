"""Smart precedent selection: Top documents + Top sentences.

Idea: First identify the most influential documents using IG/SHAP, then extract
the most important sentences from those top documents. This provides focused,
clinically relevant precedent information for doctor evaluation.

Usage:
    poetry run python -m bertgcn.select_precedents
Output:
    outputs/gcn/interpret/smart_precedents.csv
Config:
    interpretation.top_docs (default 3), interpretation.top_sentences_per_doc (default 2)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re
import pandas as pd

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger

logger = get_logger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    # Simple sentence splitting - could be improved with NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compute_sentence_importance(text: str, method: str = "length") -> List[Tuple[str, float]]:
    """Compute importance scores for sentences in a document.

    Args:
        text: Full document text
        method: Scoring method ("length", "position", "keyword")

    Returns:
        List of (sentence, score) tuples
    """
    sentences = split_into_sentences(text)

    if method == "length":
        # Score by sentence length (longer = potentially more informative)
        scored_sentences = [(sent, len(sent.split())) for sent in sentences]
    elif method == "position":
        # Score by position (earlier sentences might be more important)
        scored_sentences = [(sent, 1.0 / (i + 1)) for i, sent in enumerate(sentences)]
    elif method == "keyword":
        # Simple keyword-based scoring (could be extended with medical keywords)
        keywords = ["diagnosis", "treatment", "patient", "clinical", "symptoms", "therapy"]
        scored_sentences = []
        for sent in sentences:
            score = sum(1 for keyword in keywords if keyword.lower() in sent.lower())
            scored_sentences.append((sent, score))
    else:
        # Default to length
        scored_sentences = [(sent, len(sent.split())) for sent in sentences]

    return scored_sentences


def select_smart_precedents(cfg: DictConfig):
    """Select top documents and their most important sentences."""

    top_docs = cfg.get("interpretation", {}).get("top_docs", 3)
    top_sentences_per_doc = cfg.get("interpretation", {}).get("top_sentences_per_doc", 2)
    method = cfg.get("interpretation", {}).get("sentence_scoring", "length")

    project_root = Path(get_original_cwd())
    interpret_dir = project_root / "outputs" / "gcn" / "interpret"

    # Load document influence data (try both IG and SHAP)
    influence_file = None
    influence_method = None

    for method_name in ["ig", "shap"]:
        candidate = interpret_dir / f"document_influence_{method_name}.csv"
        if candidate.exists():
            influence_file = candidate
            influence_method = method_name
            break

    if influence_file is None:
        raise FileNotFoundError("No document influence file found. Run interpret_docs_ig or interpret_docs_shap first.")

    print(f"Loading document influence data from {influence_file} (method: {influence_method})")

    # Load the influence data
    influence_df = pd.read_csv(influence_file)

    # Load processed dataset to get original texts
    from bertgcn.train_gcn import load_processed_dataset
    dataset, label_encoder = load_processed_dataset(cfg)

    # Load tokenizer to decode texts
    model_dir = interpret_dir.parent.parent / "models" / "final_model"  # fallback
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in dataset["input_ids"]
        ]
    except:
        # Fallback: assume texts are stored elsewhere or use placeholder
        print("Warning: Could not load tokenizer, using placeholder texts")
        texts = [f"Document {i}" for i in range(len(dataset))]

    print(f"Processing {len(influence_df)} documents for smart precedent selection...")

    precedent_rows = []

    for idx, row in influence_df.iterrows():
        doc_id = int(row["doc_id"])
        pred_label = row["pred_label"]

        # Get top documents for this target document
        top_neighbor_ids = eval(row["top_neighbors"])  # List of neighbor doc IDs
        top_neighbor_scores = eval(row["neighbor_scores"])  # Corresponding scores

        # Take only the top N documents
        selected_neighbors = top_neighbor_ids[:top_docs]
        selected_scores = top_neighbor_scores[:top_docs]

        # For each top document, extract important sentences
        doc_precedents = []

        for neigh_id, score in zip(selected_neighbors, selected_scores):
            if neigh_id >= len(texts):
                print(f"Warning: Neighbor ID {neigh_id} out of range, skipping")
                continue

            neighbor_text = texts[neigh_id]

            # Compute sentence importance
            sentence_scores = compute_sentence_importance(neighbor_text, method)

            # Get top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:top_sentences_per_doc]

            doc_precedents.append({
                "neighbor_doc_id": neigh_id,
                "neighbor_score": score,
                "neighbor_text_preview": neighbor_text[:100] + "..." if len(neighbor_text) > 100 else neighbor_text,
                "top_sentences": [sent for sent, _ in top_sentences],
                "sentence_scores": [score for _, score in top_sentences]
            })

        precedent_rows.append({
            "target_doc_id": doc_id,
            "target_pred_label": pred_label,
            "target_text_preview": texts[doc_id][:100] + "..." if len(texts[doc_id]) > 100 else texts[doc_id],
            "influence_method": influence_method,
            "precedents": doc_precedents
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed smart precedents for {idx + 1}/{len(influence_df)} documents")

    # Save results
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "smart_precedents.csv"

    # Convert to DataFrame (flatten the nested structure)
    flat_rows = []
    for row in precedent_rows:
        for i, precedent in enumerate(row["precedents"]):
            flat_rows.append({
                "target_doc_id": row["target_doc_id"],
                "target_pred_label": row["target_pred_label"],
                "target_text_preview": row["target_text_preview"],
                "influence_method": row["influence_method"],
                "rank": i + 1,
                "neighbor_doc_id": precedent["neighbor_doc_id"],
                "neighbor_score": precedent["neighbor_score"],
                "neighbor_text_preview": precedent["neighbor_text_preview"],
                "top_sentences": precedent["top_sentences"],
                "sentence_scores": precedent["sentence_scores"]
            })

    pd.DataFrame(flat_rows).to_csv(out_file, index=False)
    print(f"Smart precedents saved to {out_file}")
    print(f"Total precedents generated: {len(flat_rows)}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides
    OmegaConf.set_struct(cfg, False)

    select_smart_precedents(cfg)



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides
    OmegaConf.set_struct(cfg, False)

    select_smart_precedents(cfg)


if __name__ == "__main__":
    main()
