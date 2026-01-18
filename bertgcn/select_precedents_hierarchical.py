"""Hierarchical IG/SHAP Precedent Selection.

Two-pass algorithm:
1. First pass: Graph-level IG to find top k most influential documents
2. Second pass: Word-level IG on those top k documents to find most important words,
   then extract sentences containing those words.

This provides truly hierarchical attribution: best documents → best words → best sentences.

Usage:
    poetry run python -m bertgcn.select_precedents_hierarchical
Output:
    outputs/gcn/interpret/hierarchical_precedents.csv
Config:
    interpretation.top_docs (default 3), interpretation.top_words_per_doc (default 5),
    interpretation.top_sentences_per_doc (default 2)
"""

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger

logger = get_logger(__name__)
from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def integrated_gradients_tokens(
    model: BertGCN,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int,
    steps: int = 16,
    debug: bool = False,
) -> torch.Tensor:
    """Compute IG attributions over token embeddings for a single document.

    For BertGCN, we compute attribution at the BERT level since the GCN operates
    on document-level embeddings, not individual tokens. We use the full BERT
    forward pass and proper baseline tokens.

    Returns: attribution scores for each token (seq_len,).
    """
    device = input_ids.device

    # Create proper baseline: use PAD tokens instead of zeros
    pad_token_id = (
        model.tokenizer.pad_token_id
        if hasattr(model, "tokenizer") and model.tokenizer.pad_token_id is not None
        else 0
    )
    baseline_ids = torch.full_like(input_ids, pad_token_id)
    baseline_mask = torch.zeros_like(attention_mask)  # No attention for baseline

    if debug:
        print(f"Debug: input_ids shape={input_ids.shape}, target_class={target_class}")
        print(f"Debug: baseline_ids[:10] = {baseline_ids[0][:10].tolist()}")
        print(f"Debug: pad_token_id = {pad_token_id}")

    # Get full BERT outputs for actual and baseline inputs
    with torch.no_grad():
        actual_outputs = model.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        baseline_outputs = model.bert_model(
            input_ids=baseline_ids, attention_mask=baseline_mask
        )

    total_grad = torch.zeros_like(actual_outputs.last_hidden_state)

    for i, alpha in enumerate(torch.linspace(0.0, 1.0, steps)):
        # Interpolate between baseline and actual hidden states at token level
        interpolated_hidden = (
            (
                baseline_outputs.last_hidden_state
                + alpha
                * (
                    actual_outputs.last_hidden_state
                    - baseline_outputs.last_hidden_state
                )
            )
            .clone()
            .requires_grad_(True)
        )

        # Forward pass: use interpolated [CLS] token through classifier
        # Note: This is an approximation since we skip the GCN graph processing
        cls_embedding = interpolated_hidden[:, 0, :]
        logits = model.classifier(cls_embedding)
        logit = logits[0, target_class]

        logit.backward(retain_graph=True)
        total_grad += interpolated_hidden.grad.detach()

        if debug and i == 0:
            print(f"Debug: alpha={alpha.item():.3f}, logit={logit.item():.6f}")

    avg_grad = total_grad / steps
    ig_attributions = (
        actual_outputs.last_hidden_state - baseline_outputs.last_hidden_state
    ) * avg_grad

    # Sum over embedding dimensions to get token-level scores
    token_scores = ig_attributions.sum(dim=-1).squeeze(0)  # (seq_len,)

    if debug:
        print(
            f"Debug: token_scores shape={token_scores.shape}, mean={token_scores.mean().item():.6f}"
        )

    return token_scores


def select_hierarchical_precedents(cfg: DictConfig):
    """Two-pass hierarchical precedent selection using IG."""

    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting hierarchical precedent selection...")
    sys.stdout.flush()

    top_docs = cfg.get("interpretation", {}).get("top_docs", 3)
    top_words_per_doc = cfg.get("interpretation", {}).get("top_words_per_doc", 5)
    top_sentences_per_doc = cfg.get("interpretation", {}).get(
        "top_sentences_per_doc", 2
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] Configuration: top_docs={top_docs}, top_words_per_doc={top_words_per_doc}, top_sentences_per_doc={top_sentences_per_doc}"
    )
    sys.stdout.flush()

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        project_root = Path.cwd()

    print(f"[{time.strftime('%H:%M:%S')}] Project root: {project_root}")
    sys.stdout.flush()

    interpret_dir = project_root / "outputs" / "gcn" / "interpret"
    print(f"[{time.strftime('%H:%M:%S')}] Interpret directory: {interpret_dir}")
    sys.stdout.flush()

    # Load document influence data (try both IG and SHAP)
    influence_file = None
    influence_method = None

    print(f"[{time.strftime('%H:%M:%S')}] Looking for document influence files...")
    sys.stdout.flush()

    for method_name in ["ig", "shap"]:
        candidate = interpret_dir / f"document_influence_{method_name}.csv"
        print(f"[{time.strftime('%H:%M:%S')}] Checking {candidate}...")
        sys.stdout.flush()
        if candidate.exists():
            influence_file = candidate
            influence_method = method_name
            print(
                f"[{time.strftime('%H:%M:%S')}] Found influence file: {influence_file}"
            )
            sys.stdout.flush()
            break

    if influence_file is None:
        raise FileNotFoundError(
            "No document influence file found. Run interpret_docs_ig or interpret_docs_shap first."
        )

    print(
        f"[{time.strftime('%H:%M:%S')}] Loading document influence data from {influence_file} (method: {influence_method})"
    )
    sys.stdout.flush()

    # Load the influence data
    influence_df = pd.read_csv(influence_file)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loaded influence data: {len(influence_df)} documents"
    )
    sys.stdout.flush()

    # Load processed dataset to get original texts
    print(f"[{time.strftime('%H:%M:%S')}] Loading processed dataset...")
    sys.stdout.flush()
    from bertgcn.train_gcn import load_processed_dataset

    dataset, label_encoder = load_processed_dataset(cfg)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded dataset with {len(dataset)} samples")
    sys.stdout.flush()

    # Load tokenizer to decode texts
    model_dir = project_root / "models" / "final_model"
    print(f"[{time.strftime('%H:%M:%S')}] Loading tokenizer from {model_dir}...")
    sys.stdout.flush()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"[{time.strftime('%H:%M:%S')}] Loaded tokenizer")
        sys.stdout.flush()
        texts = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in dataset["input_ids"]
        ]
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_dir}: {e}")
        texts = [f"Document {i}" for i in range(len(dataset))]

    # Load model for token-level IG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        from bertgcn.interpret_docs_ig import _load_model

        model = _load_model(
            cfg, n_classes=len(label_encoder.classes_), n_features=768
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load model for token-level IG: {e}")

    print(
        f"[{time.strftime('%H:%M:%S')}] Model loading completed (with or without success)"
    )
    sys.stdout.flush()

    print(
        f"[{time.strftime('%H:%M:%S')}] Processing {len(influence_df)} documents for hierarchical precedent selection..."
    )
    sys.stdout.flush()

    precedent_rows = []
    processed_count = 0

    for idx, row in influence_df.iterrows():
        if processed_count % 50 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Processed {processed_count}/{len(influence_df)} documents"
            )
            sys.stdout.flush()
        try:
            doc_id = int(row["doc_id"])
            pred_label = row["pred_label"]
            pred_class = label_encoder.transform([pred_label])[0]

            if processed_count % 100 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Processing document {doc_id} (idx {idx})"
                )
                sys.stdout.flush()

            # Get top documents for this target document (Pass 1 result)
            top_neighbor_ids = eval(row["top_neighbors"])
            top_neighbor_scores = eval(row["neighbor_scores"])

            # Take only the top N documents
            selected_neighbors = top_neighbor_ids[:top_docs]
            selected_scores = top_neighbor_scores[:top_docs]

        except Exception as e:
            print(
                f"[{time.strftime('%H:%M:%S')}] Warning: Error processing row {idx}: {e}"
            )
            sys.stdout.flush()
            continue

        # Pass 2: For each top document, compute token-level IG and extract sentences
        doc_precedents = []

        for neigh_id, score in zip(selected_neighbors, selected_scores):
            if neigh_id >= len(texts):
                print(f"Warning: Neighbor ID {neigh_id} out of range, skipping")
                continue

            neighbor_text = texts[neigh_id]

            # Token-level IG computation
            important_sentences = []
            if model is not None and tokenizer is not None:
                try:
                    # Tokenize the document
                    inputs = tokenizer(
                        neighbor_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                    ).to(device)

                    # Compute token-level IG
                    token_scores = integrated_gradients_tokens(
                        model,
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        pred_class,
                        debug=(idx < 1 and len(doc_precedents) < 1),
                    )

                    # Get token information
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                    attention_mask = inputs["attention_mask"][0]

                    # Filter out padding tokens and special tokens
                    valid_indices = (
                        attention_mask.bool()
                        & (inputs["input_ids"][0] != tokenizer.cls_token_id)
                        & (inputs["input_ids"][0] != tokenizer.sep_token_id)
                        & (inputs["input_ids"][0] != tokenizer.pad_token_id)
                    )

                    valid_tokens = [
                        tokens[j] for j in range(len(tokens)) if valid_indices[j]
                    ]
                    valid_scores = token_scores[valid_indices]

                    # Get top tokens
                    if len(valid_scores) > 0:
                        top_vals, top_inds = torch.topk(
                            valid_scores, k=min(top_words_per_doc, len(valid_scores))
                        )
                        top_tokens = [valid_tokens[idx] for idx in top_inds.tolist()]

                        # Find sentences containing the most influential words
                        sentences = split_into_sentences(neighbor_text)
                        sentence_scores = []

                        for sent_idx, sentence in enumerate(sentences):
                            sent_tokens = tokenizer.tokenize(sentence.lower())
                            sent_score = 0.0
                            for token, score in zip(
                                valid_tokens, valid_scores.tolist()
                            ):
                                # Check if the token appears in this sentence
                                if token.lower().strip("##") in [
                                    t.lower().strip("##") for t in sent_tokens
                                ]:
                                    sent_score += score
                            if (
                                sent_score > 0
                            ):  # Only include sentences with influential words
                                sentence_scores.append((sent_idx, sent_score, sentence))

                        # Get top sentences by influential word score
                        sentence_scores.sort(key=lambda x: x[1], reverse=True)
                        top_sentences = sentence_scores[:top_sentences_per_doc]

                        important_sentences = [sent for _, _, sent in top_sentences]
                        sentence_scores_list = [score for _, score, _ in top_sentences]

                        if idx < 1:  # Debug for first document
                            print(f"Debug: Top tokens for doc {neigh_id}: {top_tokens}")
                            print(
                                f"Debug: Found {len(important_sentences)} important sentences"
                            )
                    else:
                        # Fallback: use simple sentence selection
                        sentences = split_into_sentences(neighbor_text)
                        important_sentences = (
                            sentences[:top_sentences_per_doc] if sentences else []
                        )
                        sentence_scores_list = [
                            len(sent.split()) for sent in important_sentences
                        ]

                except Exception as e:
                    print(f"Warning: Error computing token IG for doc {neigh_id}: {e}")
                    # Fallback to simple sentence selection
                    sentences = split_into_sentences(neighbor_text)
                    important_sentences = (
                        sentences[:top_sentences_per_doc] if sentences else []
                    )
                    sentence_scores_list = [
                        len(sent.split()) for sent in important_sentences
                    ]
            else:
                # Fallback when model/tokenizer not available
                sentences = split_into_sentences(neighbor_text)
                important_sentences = (
                    sentences[:top_sentences_per_doc] if sentences else []
                )
                sentence_scores_list = [
                    len(sent.split()) for sent in important_sentences
                ]

            doc_precedents.append(
                {
                    "neighbor_doc_id": neigh_id,
                    "neighbor_score": score,
                    "neighbor_text_preview": (
                        neighbor_text[:100] + "..."
                        if len(neighbor_text) > 100
                        else neighbor_text
                    ),
                    "top_sentences": important_sentences,
                    "sentence_scores": sentence_scores_list,
                    "influential_tokens": (
                        top_tokens if "top_tokens" in locals() else []
                    ),
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

        if (idx + 1) % 50 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Processed hierarchical precedents for {idx + 1}/{len(influence_df)} documents"
            )
            sys.stdout.flush()

        processed_count += 1

    # Save results
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hierarchical_precedents.csv"

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
                        "influential_tokens": precedent.get("influential_tokens", []),
                    }
                )
        except Exception as e:
            print(
                f"Warning: Error processing precedents for document {row.get('target_doc_id', 'unknown')}: {e}"
            )
            continue

    pd.DataFrame(flat_rows).to_csv(out_file, index=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Hierarchical precedents saved to {out_file}")
    print(
        f"[{time.strftime('%H:%M:%S')}] Total hierarchical precedents generated: {len(flat_rows)}"
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] Total processing time: {elapsed_time:.2f} seconds"
    )
    sys.stdout.flush()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides
    OmegaConf.set_struct(cfg, False)

    select_hierarchical_precedents(cfg)


if __name__ == "__main__":
    main()
