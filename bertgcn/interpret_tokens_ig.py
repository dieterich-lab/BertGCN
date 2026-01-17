"""Token-level influence via Integrated Gradients.

Idea: For each document, compute IG attributions at the token level to identify
which words/tokens are most important for the classification decision.

Usage:
    poetry run python -m bertgcn.interpret_tokens_ig
Output:
    outputs/gcn/interpret/token_influence_ig.csv
Config:
    interpretation.top_k (default 5), interpretation.max_docs (optional)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import hydra
from bertgcn.core import get_logger

logger = get_logger(__name__)
from bertgcn.train_gcn import BertGCN, load_graph_data_from_disk, load_processed_dataset


def _resolve_model_dir(cfg: DictConfig) -> Path:
    """Pick model_dir in priority: cfg.interpretation.model_dir -> MLflow artifacts -> hydra/gcn/**/final_model -> models/final_model."""

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
    for cand in candidates + [fallback]:
        if cand.is_dir():
            logger.info(f"Using fallback model directory: {cand}")
            return cand
    logger.warning(f"Using final fallback: {fallback}")
    return fallback


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


def integrated_gradients_tokens(
    model: BertGCN,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int,
    steps: int = 16,
    debug: bool = False,
) -> torch.Tensor:
    """Compute IG attributions over token embeddings for a single document.

    Returns: attribution tensor shaped like input embeddings (seq_len, hidden_dim).
    """

    # Get baseline (zero embeddings)
    baseline_ids = torch.zeros_like(input_ids)
    baseline_mask = torch.ones_like(attention_mask)  # Keep attention mask for baseline

    if debug:
        print(f"Debug: input_ids shape={input_ids.shape}, target_class={target_class}")

    # Get BERT embeddings for the actual input
    with torch.no_grad():
        inputs_embeds = model.bert_model.embeddings(input_ids)
        baseline_embeds = model.bert_model.embeddings(baseline_ids)

    total_grad = torch.zeros_like(inputs_embeds)

    for i, alpha in enumerate(torch.linspace(0.0, 1.0, steps)):
        # Interpolate between baseline and actual embeddings
        x_embeds = (baseline_embeds + alpha * (inputs_embeds - baseline_embeds)).clone().requires_grad_(True)

        # Forward pass through BERT
        outputs = model.bert_model(
            inputs_embeds=x_embeds,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Forward through GCN (but we need to handle this differently since we're doing token-level)
        # For now, we'll just use the BERT output directly for classification
        # This is a simplification - ideally we'd need to integrate with the full graph
        logits = model.classifier(cls_embedding)
        logit = logits[0, target_class]

        logit.backward(retain_graph=True)
        total_grad += x_embeds.grad.detach()

        if debug and i == 0:
            print(f"Debug: alpha={alpha.item():.3f}, logit={logit.item():.6f}")

    avg_grad = total_grad / steps
    ig = (inputs_embeds - baseline_embeds) * avg_grad

    # Sum over embedding dimensions to get token-level scores
    token_scores = ig.sum(dim=-1).squeeze(0)  # (seq_len,)

    if debug:
        print(f"Debug: token_scores shape={token_scores.shape}, mean={token_scores.mean().item():.6f}")

    return token_scores


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    # Simple sentence splitting - could be improved with NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def run_token_ig(cfg: DictConfig):
    top_k = cfg.get("interpretation", {}).get("top_k", 5)
    max_docs = cfg.get("interpretation", {}).get("max_docs", None)

    # Data
    dataset, label_encoder = load_processed_dataset(cfg)
    n_classes = len(label_encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer to decode texts
    model_dir = _resolve_model_dir(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Decode texts from input_ids
    texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in dataset["input_ids"]
    ]

    # Model
    model = _load_model(cfg, n_classes=n_classes, n_features=768).to(device)

    # Get predictions for all documents
    print("Computing predictions for all documents...", flush=True)
    predictions = []
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
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(cls_embedding)
            pred_class = logits.argmax(dim=1).item()

        predictions.append(pred_class)

        if (i + 1) % 100 == 0:
            print(f"Processed predictions for {i + 1}/{len(texts)} documents", flush=True)

    n_docs = len(dataset)
    target_docs = range(n_docs if max_docs is None else min(max_docs, n_docs))

    rows: List[Dict] = []
    for i, doc_idx in enumerate(target_docs):
        text = texts[doc_idx]
        pred_class = predictions[doc_idx]

        # Tokenize for IG
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(device)

        # Compute token-level IG
        token_scores = integrated_gradients_tokens(
            model, inputs["input_ids"], inputs["attention_mask"],
            pred_class, debug=(doc_idx < 3)
        )

        # Get token information
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_mask = inputs["attention_mask"][0]

        # Filter out padding tokens and special tokens
        valid_indices = attention_mask.bool() & \
                       (inputs["input_ids"][0] != tokenizer.cls_token_id) & \
                       (inputs["input_ids"][0] != tokenizer.sep_token_id) & \
                       (inputs["input_ids"][0] != tokenizer.pad_token_id)

        valid_tokens = [tokens[j] for j in range(len(tokens)) if valid_indices[j]]
        valid_scores = token_scores[valid_indices]

        # Get top tokens
        if len(valid_scores) > 0:
            top_vals, top_inds = torch.topk(valid_scores, k=min(top_k, len(valid_scores)))
            top_tokens = [valid_tokens[idx] for idx in top_inds.tolist()]

            # Split text into sentences and find which sentences contain top tokens
            sentences = split_into_sentences(text)
            sentence_scores = []

            for sent_idx, sentence in enumerate(sentences):
                sent_tokens = tokenizer.tokenize(sentence)
                sent_score = 0.0
                for token, score in zip(valid_tokens, valid_scores.tolist()):
                    if token in sent_tokens:
                        sent_score += score
                sentence_scores.append((sent_idx, sent_score, sentence))

            # Get top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:min(3, len(sentence_scores))]

            rows.append({
                "doc_id": doc_idx,
                "text": text[:200] + "..." if len(text) > 200 else text,
                "pred_label": label_encoder.inverse_transform([pred_class])[0],
                "top_tokens": top_tokens,
                "token_scores": [float(v) for v in top_vals],
                "top_sentences": [sent for _, _, sent in top_sentences],
                "sentence_scores": [float(score) for _, score, _ in top_sentences],
            })

        if (i + 1) % 50 == 0:
            print(f"Processed token IG for {i + 1}/{len(target_docs)} documents", flush=True)

    import pandas as pd

    project_root = Path(get_original_cwd())
    out_dir = project_root / "outputs" / "gcn" / "interpret"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "token_influence_ig.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Token-level IG influence saved to {out_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_token_ig(cfg)



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    run_token_ig(cfg)


if __name__ == "__main__":
    main()
