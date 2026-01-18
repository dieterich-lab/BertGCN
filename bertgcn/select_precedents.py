"""Smart precedent selection: Top documents + Top sentences.

Idea: First identify the most influential documents using IG/SHAP, then extract
the most important sentences from those top documents using token-level IG.
This provides focused, clinically relevant precedent information for doctor evaluation.

Usage:
    poetry run python -m bertgcn.select_precedents
Output:
    outputs/gcn/interpret/smart_precedents.csv
Config:
    interpretation.top_docs (default 3), interpretation.top_sentences_per_doc (default 2)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger

logger = get_logger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    # Simple sentence splitting - could be improved with NLTK
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def compute_token_ig_for_document(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    target_class: int,
    device: torch.device,
    top_k_tokens: int = 10,
) -> List[Tuple[str, float]]:
    """Compute token-level IG for a document and return top influential tokens.

    Args:
        model: BertGCN model
        tokenizer: BERT tokenizer
        text: Document text
        target_class: Predicted class for the document
        device: Torch device
        top_k_tokens: Number of top tokens to return

    Returns:
        List of (token, score) tuples for most influential tokens
    """
    # Tokenize the document
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    ).to(device)

    # Compute IG attributions
    baseline_ids = torch.zeros_like(inputs["input_ids"])
    baseline_mask = torch.ones_like(inputs["attention_mask"])

    # Get embeddings
    with torch.no_grad():
        inputs_embeds = model.bert_model.embeddings(inputs["input_ids"])
        baseline_embeds = model.bert_model.embeddings(baseline_ids)

    total_grad = torch.zeros_like(inputs_embeds)

    # IG computation with 16 steps
    for alpha in torch.linspace(0.0, 1.0, 16):
        x_embeds = (
            (baseline_embeds + alpha * (inputs_embeds - baseline_embeds))
            .clone()
            .requires_grad_(True)
        )

        outputs = model.bert_model(
            inputs_embeds=x_embeds, attention_mask=inputs["attention_mask"]
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = model.classifier(cls_embedding)
        logit = logits[0, target_class]

        logit.backward(retain_graph=True)
        total_grad += x_embeds.grad.detach()

    avg_grad = total_grad / 16
    ig = (inputs_embeds - baseline_embeds) * avg_grad
    token_scores = ig.sum(dim=-1).squeeze(0)  # (seq_len,)

    # Get tokens and filter out special tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention_mask = inputs["attention_mask"][0]

    valid_indices = (
        attention_mask.bool()
        & (inputs["input_ids"][0] != tokenizer.cls_token_id)
        & (inputs["input_ids"][0] != tokenizer.sep_token_id)
        & (inputs["input_ids"][0] != tokenizer.pad_token_id)
    )

    valid_tokens = [tokens[j] for j in range(len(tokens)) if valid_indices[j]]
    valid_scores = token_scores[valid_indices]

    # Get top tokens
    if len(valid_scores) > 0:
        top_vals, top_inds = torch.topk(
            valid_scores, k=min(top_k_tokens, len(valid_scores))
        )
        top_token_pairs = [
            (valid_tokens[idx], float(top_vals[i])) for i, idx in enumerate(top_inds)
        ]
        return top_token_pairs

    return []


def compute_sentence_importance_hierarchical(
    text: str, influential_tokens: List[Tuple[str, float]], tokenizer
) -> List[Tuple[str, float]]:
    """Compute sentence importance based on influential tokens from IG.

    Args:
        text: Full document text
        influential_tokens: List of (token, score) from token-level IG
        tokenizer: BERT tokenizer for tokenization

    Returns:
        List of (sentence, score) tuples scored by influential token content
    """
    sentences = split_into_sentences(text)

    if not influential_tokens:
        # Fallback to length-based scoring if no influential tokens
        return [(sent, len(sent.split())) for sent in sentences]

    # Create token->score mapping
    token_score_map = dict(influential_tokens)

    scored_sentences = []
    for sent in sentences:
        # Tokenize sentence and find influential tokens
        sent_tokens = tokenizer.tokenize(sent.lower())
        sent_score = 0.0

        for token, score in influential_tokens:
            # Check if influential token appears in sentence (case-insensitive)
            if token.lower() in sent.lower() or any(
                token.lower() in st for st in sent_tokens
            ):
                sent_score += score

        scored_sentences.append((sent, sent_score))


def compute_sentence_importance(
    text: str, method: str = "length"
) -> List[Tuple[str, float]]:
    """Compute importance scores for sentences in a document (fallback method).

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
        keywords = [
            "diagnosis",
            "treatment",
            "patient",
            "clinical",
            "symptoms",
            "therapy",
        ]
        scored_sentences = []
        for sent in sentences:
            score = sum(1 for keyword in keywords if keyword.lower() in sent.lower())
            scored_sentences.append((sent, score))
    else:
        # Default to length
        scored_sentences = [(sent, len(sent.split())) for sent in sentences]

    return scored_sentences


def select_smart_precedents(cfg: DictConfig):
    """Select top documents and their most important sentences using hierarchical IG."""

    top_docs = cfg.get("interpretation", {}).get("top_docs", 3)
    top_sentences_per_doc = cfg.get("interpretation", {}).get(
        "top_sentences_per_doc", 2
    )
    method = cfg.get("interpretation", {}).get("sentence_scoring", "hierarchical")
    top_tokens_per_doc = cfg.get("interpretation", {}).get("top_tokens_per_doc", 20)

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        # Fallback when not running under Hydra
        project_root = Path.cwd()

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
        raise FileNotFoundError(
            "No document influence file found. Run interpret_docs_ig or interpret_docs_shap first."
        )

    print(
        f"Loading document influence data from {influence_file} (method: {influence_method})"
    )

    # Load the influence data
    influence_df = pd.read_csv(influence_file)

    # Load processed dataset to get original texts
    from bertgcn.train_gcn import load_processed_dataset

    dataset, label_encoder = load_processed_dataset(cfg)

    # Load tokenizer and model for hierarchical IG
    model_dir = project_root / "models" / "final_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    tokenizer = None
    if method == "hierarchical":
        try:
            from transformers import AutoModel, AutoTokenizer

            from bertgcn.train_gcn import BertGCN

            print("Loading BERT+GCN model for hierarchical token-level IG...")

            # Load BERT model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            bert_model = AutoModel.from_pretrained(model_dir)
            feat_dim = bert_model.config.hidden_size

            # Load GCN components
            gcn_checkpoint = torch.load(
                model_dir / "gcn_components.pt", map_location="cpu"
            )
            m = gcn_checkpoint["m"]
            nb_class = gcn_checkpoint["nb_class"]
            # Extract n_hidden from checkpoint (conv1.bias shape gives us n_hidden)
            n_hidden = gcn_checkpoint["gcn"]["conv1.bias"].shape[0]

            # Create model
            model = BertGCN(
                bert_model=bert_model,
                tokenizer=tokenizer,
                feat_dim=feat_dim,
                nb_class=nb_class,
                m=m,
                gcn_layers=1,
                n_hidden=n_hidden,
                dropout=cfg.gcn.dropout,
            )

            # Load weights
            model.classifier.load_state_dict(gcn_checkpoint["classifier"])
            model.gcn.load_state_dict(gcn_checkpoint["gcn"])
            model = model.to(device)
            model.eval()

            # Decode texts
            texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in dataset["input_ids"]
            ]
            print("✓ Model and tokenizer loaded for hierarchical IG")

        except Exception as e:
            print(f"Warning: Could not load model for hierarchical IG: {e}")
            print("Falling back to length-based sentence scoring")
            method = "length"
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                texts = [
                    tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in dataset["input_ids"]
                ]
            except:
                texts = [f"Document {i}" for i in range(len(dataset))]
    else:
        # For non-hierarchical methods, just load tokenizer for text decoding
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in dataset["input_ids"]
            ]
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            texts = [f"Document {i}" for i in range(len(dataset))]

    print(f"Processing {len(influence_df)} documents for smart precedent selection...")

    precedent_rows = []

    for idx, row in influence_df.iterrows():
        try:
            doc_id = int(row["doc_id"])
            pred_label = row["pred_label"]

            # Get top documents for this target document
            top_neighbor_ids = eval(row["top_neighbors"])  # List of neighbor doc IDs
            top_neighbor_scores = eval(row["neighbor_scores"])  # Corresponding scores

            # Take only the top N documents
            selected_neighbors = top_neighbor_ids[:top_docs]
            selected_scores = top_neighbor_scores[:top_docs]

        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            print(f"  doc_id: {row.get('doc_id', 'N/A')}")
            print(f"  top_neighbors: {row.get('top_neighbors', 'N/A')[:100]}...")
            print(f"  neighbor_scores: {row.get('neighbor_scores', 'N/A')[:100]}...")
            continue

        # For each top document, extract important sentences
        doc_precedents = []

        for neigh_id, score in zip(selected_neighbors, selected_scores):
            if neigh_id >= len(texts):
                print(f"Warning: Neighbor ID {neigh_id} out of range, skipping")
                continue

            neighbor_text = texts[neigh_id]

            # Compute sentence importance using hierarchical IG
            if method == "hierarchical" and model is not None and tokenizer is not None:
                # Get the predicted class for this neighbor document
                # We need to find what class this document was predicted as
                neighbor_pred_class = None
                for _, inf_row in influence_df.iterrows():
                    if int(inf_row["doc_id"]) == neigh_id:
                        neighbor_pred_class = label_encoder.transform(
                            [inf_row["pred_label"]]
                        )[0]
                        break

                if neighbor_pred_class is not None:
                    # Compute token-level IG for this document
                    influential_tokens = compute_token_ig_for_document(
                        model,
                        tokenizer,
                        neighbor_text,
                        neighbor_pred_class,
                        device,
                        top_tokens_per_doc,
                    )

                    # Score sentences based on influential tokens
                    sentence_scores = compute_sentence_importance_hierarchical(
                        neighbor_text, influential_tokens, tokenizer
                    )
                else:
                    # Fallback if we can't find the predicted class
                    sentence_scores = [
                        (sent, len(sent.split()))
                        for sent in split_into_sentences(neighbor_text)
                    ]
            else:
                # Fallback to traditional methods
                sentence_scores = compute_sentence_importance(neighbor_text, method)

            # Get top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:top_sentences_per_doc]

            doc_precedents.append(
                {
                    "neighbor_doc_id": neigh_id,
                    "neighbor_score": score,
                    "neighbor_text_preview": (
                        neighbor_text[:100] + "..."
                        if len(neighbor_text) > 100
                        else neighbor_text
                    ),
                    "top_sentences": [sent for sent, _ in top_sentences],
                    "sentence_scores": [score for _, score in top_sentences],
                }
            )

        precedent_rows.append(
            {
                "target_doc_id": doc_id,
                "target_pred_label": pred_label,
                "target_text_preview": (
                    texts[doc_id][:100] + "..."
                    if len(texts[doc_id]) > 100
                    else texts[doc_id]
                ),
                "influence_method": influence_method,
                "precedents": doc_precedents,
            }
        )

        if (idx + 1) % 100 == 0:
            print(
                f"Processed smart precedents for {idx + 1}/{len(influence_df)} documents"
            )

    # Save results
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "smart_precedents.csv"

    # Convert to DataFrame (flatten the nested structure)
    flat_rows = []
    for row in precedent_rows:
        try:
            if not row["precedents"]:
                print(
                    f"Warning: No precedents for document {row['target_doc_id']}, skipping"
                )
                continue

            for i, precedent in enumerate(row["precedents"]):
                flat_rows.append(
                    {
                        "target_doc_id": row["target_doc_id"],
                        "target_pred_label": row["target_pred_label"],
                        "target_text_preview": row["target_text_preview"],
                        "influence_method": row["influence_method"],
                        "rank": i + 1,
                        "neighbor_doc_id": precedent["neighbor_doc_id"],
                        "neighbor_score": precedent["neighbor_score"],
                        "neighbor_text_preview": precedent["neighbor_text_preview"],
                        "top_sentences": precedent["top_sentences"],
                        "sentence_scores": precedent["sentence_scores"],
                    }
                )
        except Exception as e:
            print(
                f"Warning: Error processing precedents for document {row.get('target_doc_id', 'unknown')}: {e}"
            )
            continue

    pd.DataFrame(flat_rows).to_csv(out_file, index=False)
    print(f"Smart precedents saved to {out_file}")
    print(f"Total precedents generated: {len(flat_rows)}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides
    OmegaConf.set_struct(cfg, False)

    select_smart_precedents(cfg)


if __name__ == "__main__":
    main()
