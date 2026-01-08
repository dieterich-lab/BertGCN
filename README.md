# BertGCN: BERT-based Graph Convolutional Network with Document-Level Interpretability for Precedents Detection

**Major Extension:** BertGCN now includes advanced document-level interpretability capabilities, enabling the identification of influential documents (precedents) that contribute to a model's predictions. This opens up a wide range of use cases in explainable AI, legal analysis, medical diagnostics, and any domain requiring transparent document classification with precedent-based reasoning.

A BERT-based Graph Convolutional Network for document classification, combining the strengths of transformer-based language models with graph neural networks for enhanced text understanding.

---

## Table of Contents

- [Installation](#installation)
- [Data Format](#data-format)
- [Modes Overview](#modes-overview)
- [Usage](#usage)
- [Configuration Management](#configuration-management)
- [Output Directory Structure](#output-directory-structure)
- [MLflow Tracking](#mlflow-tracking)
- [Testing](#testing)
- [Contributing](#contributing)

---

## 🚀 Installation

**Requirements:**

- Python 3.10+
- Poetry ([install guide](https://python-poetry.org/docs/#installation))

**Installation:**

```bash
git clone https://github.com/dieterich-lab/BertGCN.git
cd BertGCN
./install.sh
```

## 📊 Data Format

Input files must be CSV format with columns for diagnosis, anamnesis, risk_factor, discharge_letter, medication_type. The preprocessing step handles cleaning and tokenization.

## ⚡ Modes Overview

| Mode         | Description                                                                 | Typical Use/Plugin         |
|--------------|-----------------------------------------------------------------------------|---------------------------|
| preprocess   | Clean and tokenize the dataset.                                            | Data preparation          |
| build-graph  | Build the document-word graph via PMI and TF-IDF edges.                    | Graph construction       |
| train_bert   | Fine-tune BERT for document classification.                                 | Baseline model            |
| train_gcn    | Train the BertGCN ensemble model.                                          | Main model                |
| predict      | Run inference/prediction on new data.                                       | Evaluation                |
| interpret    | Feature importance/interpretation using SHAP.                              | Model explanation         |

**Notes:**

- Preprocessing and graph building are prerequisites for training.
- BERT fine-tuning provides a strong baseline for comparison with BertGCN.

### Mode Quickstart

**Preprocess**

```bash
poetry run bertgcn preprocess
```

- Output: `data/processed/tokenized_dataset` plus label encoders.

**Build-graph**

```bash
poetry run bertgcn build-graph
```

- Output: Graph components in `data/ind.*` files.

**Train BERT**

```bash
poetry run bertgcn finetune
```

- Output: Fine-tuned BERT model in `models/finetuned/`.

**Train GCN**

```bash
poetry run bertgcn train
```

- Output: Trained BertGCN model in `models/final_model/`.

**Predict**

```bash
poetry run bertgcn predict
```

- Requires trained model; outputs predictions CSV.

**Interpret**

```bash
poetry run bertgcn interpret
```

- Outputs SHAP-based feature importance.

---

## 🛠️ Usage

Run any mode with:

```bash
poetry run bertgcn {preprocess | build-graph | finetune | train | predict | interpret}
```

### Quickstart Commands

With default configs, run the pipeline sequentially:

```bash
poetry run bertgcn preprocess
poetry run bertgcn build-graph
poetry run bertgcn finetune
poetry run bertgcn train
poetry run bertgcn predict
```

### Runtime Overrides

Pass overrides like `hparams.learning_rate=3e-5` or `training.epochs=10` after the command to tweak parameters without editing YAML.

---

## ⚙️ Configuration Management

BertGCN uses Hydra to compose configurations. The framework supports mode-based overrides.

### Config Structure

- `conf/config.yaml` - Shared defaults and mode selection
- `conf/mode/` - Mode-specific configurations
- `conf/hparams.yaml` - Hyperparameters
- `conf/dataset.yaml` - Dataset settings

### Custom Experiments

Create experiment configs in `./experiments/`:

```
experiments/
└── my_experiment/
    └── config.yaml
```

With mode selection:

```yaml
defaults:
  - mode: train_bert
  - _self_

hparams:
  learning_rate: 2e-5
```

Run with:

```bash
poetry run bertgcn finetune --config-path experiments/my_experiment
```

---

## 📂 Output Directory Structure

All outputs are organized under the `outputs/` directory with mode-specific subfolders:

```
outputs/
├── train_bert/              # BERT fine-tuning outputs
│   ├── hydra/               # Hydra run directories (one per run)
│   │   └── <timestamp>/     # e.g., 2025-08-28_12-39-43/
│   │       ├── config.yaml  # Resolved configuration
│   │       ├── overrides.yaml
│   │       └── .hydra/
│   └── mlruns/              # MLflow tracking data for BERT runs
│       └── <experiment_id>/ # e.g., 142214766644063952/
│           ├── <run_id>/    # Individual run data
│           │   ├── metrics/
│           │   ├── params/
│           │   ├── tags/
│           │   ├── artifacts/
│           │   └── meta.yaml
│           └── meta.yaml
└── train_gcn/               # BertGCN training outputs
    ├── hydra/               # Hydra run directories (one per run)
    │   └── <timestamp>/     # e.g., 2025-08-29_00-08-27/
    │       ├── config.yaml  # Resolved configuration
    │       ├── overrides.yaml
    │       └── .hydra/
    └── mlruns/              # MLflow tracking data for GCN runs
        └── <experiment_id>/ # e.g., 273766186618787412/
            ├── <run_id>/    # Individual run data
            │   ├── metrics/
            │   ├── params/
            │   ├── tags/
            │   ├── artifacts/
            │   └── meta.yaml
            └── meta.yaml

models/                      # Final trained models (legacy location)
├── finetuned/               # BERT fine-tuning results
│   └── <timestamp>/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── label_encoder.joblib
└── final_model/             # BertGCN final model
    ├── model.pt
    └── config.json

data/                        # Processed data and graphs
├── processed/
│   ├── tokenized_dataset/
│   ├── label_encoder.joblib
│   └── meds_label_encoder.joblib
└── ind.*                    # Graph components (PMI/TF-IDF edges)
```

### Artifact Contents

- **Models**: Saved in `models/` with timestamps for versioning
- **Datasets**: Cached processed data for reproducibility
- **Graphs**: Sparse matrices for document-word relationships
- **Predictions**: CSV files with class probabilities
- **Interpretations**: SHAP values for feature importance

---

## 📈 MLflow Tracking

MLflow tracking is enabled by default and writes to mode-specific directories under `outputs/`:

```bash
# View BERT fine-tuning experiments
poetry run mlflow ui --backend-store-uri outputs/train_bert/mlruns

# View BertGCN training experiments  
poetry run mlflow ui --backend-store-uri outputs/train_gcn/mlruns
```

Tracks parameters, metrics, and artifacts automatically for each run.

---

## 🧭 Document-Level Interpretability for Precedents Detection (Major Extension)

**Revolutionary Feature:** BertGCN introduces groundbreaking document-level interpretability, allowing users to explain predictions by identifying influential documents (precedents) that shaped the model's decision. This transforms opaque black-box models into transparent, precedent-aware systems, enabling:

- **Legal Analysis:** Identify case precedents influencing judicial decisions.
- **Medical Diagnostics:** Highlight similar patient records affecting diagnoses.
- **Regulatory Compliance:** Trace decision influences for auditing and transparency.
- **Research & Education:** Understand model reasoning for scientific insights.

Three complementary approaches explain a document's prediction by citing influential *documents* (precedents). All assume a trained BertGCN and the graph with edge weights.

- **A. Neighbor scoring (fast, edge-weighted GCN probs)**
  - Script: `poetry run python -m bertgcn.interpret_docs_neighbors`
  - How it works: For a target doc *t* with predicted class *c*, each incoming
    neighbor *j* gets a score `edge_weight(j→t) * P_c(j)` using the GCN class
    probability of *j*. Top-k neighbors are returned.
  - Output: `outputs/train_gcn/interpret/document_influence.csv` with
    `top_neighbors` and `neighbor_scores`.

- **B. Integrated Gradients over document features (gradient-based)**
  - Script: `poetry run python -m bertgcn.interpret_docs_ig`
  - How it works: Runs IG on the full document feature matrix for the target
    doc and class; sums attributions per document to produce a ranked list of
    influential documents.
  - Output: `outputs/train_gcn/interpret/document_influence_ig.csv`.

- **C. SHAP-style neighbor perturbation (leave-one-out edges)**
  - Script: `poetry run python -m bertgcn.interpret_docs_shap`
  - How it works: For each incoming neighbor edge to the target doc, removes
    the edge and measures the drop in the target class probability. The delta
    serves as the neighbor's importance (SHAP-like intuition without full
    kernel sampling).
  - Output: `outputs/train_gcn/interpret/document_influence_shap.csv`.

**Config knobs (Hydra via mode/train_gcn):** `interpretation.top_k` (default 5),
`interpretation.max_docs` (optional), plus model/graph settings. Models load
from `outputs/train_gcn/final_model/pytorch_model.bin` when present.

---

## 🧪 Testing

Run tests with:

```bash
poetry run pytest tests/
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and open Pull Request

---

## 📝 Citation

```bibtex
@software{bertgcn2024,
  title = {BertGCN: BERT-based Graph Convolutional Network for Document Classification},
  author = {Philipp Wiesenbach},
  year = {2024},
  url = {https://github.com/dieterich-lab/BertGCN}
}
```