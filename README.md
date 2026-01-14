# BertGCN: BERT-based Graph Convolutional Network with Document-Level Interpretability for Precedents Detection

**Major Extension:** BertGCN now includes advanced document-level interpretability capabilities, enabling the identification of influential documents (precedents) that contribute to a model's predictions. This opens up a wide range of use cases in explainable AI, legal analysis, medical diagnostics, and any domain requiring transparent document classification with precedent-based reasoning.

A BERT-based Graph Convolutional Network for document classification, combining the strengths of transformer-based language models with graph neural networks for enhanced text understanding.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [Configuration Management](#configuration-management)
- [Running Experiments and Sweeps](#running-experiments-and-sweeps)
- [Model Organization](#model-organization)
- [MLflow Tracking](#mlflow-tracking)
- [Interpretability](#interpretability)
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

This installs all dependencies via Poetry and sets up the environment.

---

## ⚡ Quick Start

Run the full pipeline with default settings:

```bash
# 1. Preprocess data (run once)
poetry run preprocess

# 2. Build document-word graph
poetry run build-graph

# 3. Fine-tune BERT baseline
poetry run finetune-bert

# 4. Train BertGCN model
poetry run train-bertgcn

# 5. Run predictions (optional)
poetry run predict

# 6. Interpret results (optional)
poetry run interpret-docs-shap
```

Models are saved in `outputs/`, Hydra configs in `hydra/`, and experiments tracked in `mlruns/`. View all experiments in the single MLflow UI.

---

## 🔄 Workflow Overview

BertGCN follows a sequential pipeline:

### 1. **Preprocess** (`poetry run preprocess`)
   - Cleans and tokenizes `data/med_indication_all_RF_diag.csv`
   - Saves processed dataset to `data/processed/tokenized_dataset`
   - Outputs label encoders for downstream use
   - **Run once** after data changes

### 2. **Build Graph** (`poetry run build-graph`)
   - Constructs document-word graph using sliding windows, PMI, and TF-IDF edges
   - Saves graph components (`adj`, `x`, `tx`, etc.) to `data/ind.*` files
   - Configurable via `bertgcn/params.py` (window size, BERT model, etc.)

### 3. **Fine-tune BERT** (`poetry run finetune-bert`)
   - Trains plain BERT for document classification as baseline
   - Uses Hydra config from `conf/config.yaml`
   - Outputs model to `outputs/bert/final_model`
   - Logs to MLflow in the canonical `mlruns` store (experiment: bertgcn_finetuning)

### 4. **Train BertGCN** (`poetry run train-bertgcn`)
   - Trains the full BertGCN ensemble (BERT + GCN)
   - Supports hyperparameter sweeps via Optuna
   - Saves models to `outputs/gcn/final_model`
   - Logs Hydra configs to `hydra/gcn/runs/<timestamp>/` (single runs) or `hydra/gcn/sweeps/<timestamp>/` (sweeps)
   - Logs experiments to MLflow in the canonical `mlruns` store (experiment: bertgcn_training)

### 5. **Predict** (`poetry run predict`)
   - Runs inference on new data using trained models
   - Requires model path and input data

### 6. **Interpret** (`poetry run interpret` or `poetry run interpret-docs-*`)
   - Provides feature-level or document-level explanations
   - See [Interpretability](#interpretability) section

---

## ⚙️ Configuration Management

BertGCN uses Hydra for flexible configuration. All configs are in `conf/`:

- `conf/config.yaml` - Main config composing defaults
- `conf/main.yaml` - Global settings (MLflow URI, experiment names)
- `conf/hparams.yaml` - Hyperparameters (learning rate, dropout, etc.)
- `conf/gcn.yaml` - GCN-specific settings (mix_factor, layers, zero_word_features)
- `conf/dataset.yaml` - Dataset splits and paths

### Runtime Overrides

Override any parameter at runtime:

```bash
# Change learning rate
poetry run train-bertgcn hparams.learning_rate=3e-5

# Modify GCN settings
poetry run train-bertgcn gcn.mix_factor=0.7 gcn.zero_word_features=true

# Adjust training
poetry run train-bertgcn training.epochs=50 training.batch_size=32
```



### Custom Configs

Create experiment-specific configs in `experiments/my_experiment/config.yaml`:

```yaml
defaults:
  - _self_
  - main
  - hparams
  - gcn
  - dataset

hparams:
  learning_rate: 2e-5

gcn:
  mix_factor: 0.8
```

Run with:

```bash
poetry run train-bertgcn --config-path experiments/my_experiment
```

---

## 🔍 Running Experiments and Sweeps

### Single Runs

Run with custom parameters:

```bash
poetry run train-bertgcn hparams.learning_rate=1e-5 gcn.mix_factor=0.5
```

### Hyperparameter Sweeps

Use Optuna for automated sweeps:

```bash
# Sweep learning rate and mix_factor
poetry run train-bertgcn --multirun \
  hparams.learning_rate=1e-5,3e-5,5e-5 \
  gcn.mix_factor=0.3,0.5,0.7 \
  gcn.zero_word_features=true,false
```

- Each combination runs as a separate job
- Results logged to MLflow for comparison
- Hydra configs saved in `hydra/gcn/sweeps/<timestamp>/`, models in `outputs/gcn/`



---

## 📂 Model Organization

Models are saved in `outputs/` directories, Hydra configs in `hydra/`, and experiments tracked in `mlruns/`:

### Directory Structure

```
outputs/
├── bert/
│   └── final_model/          # BERT baseline model
└── gcn/
    ├── final_model/          # Best GCN model
    ├── interpret/            # Document-level interpretability outputs
    └── multirun/<timestamp>/ # Legacy Hydra logs for sweeps (to be migrated)

hydra/
├── bert/
│   └── runs/<timestamp>/     # Hydra logs for BERT single runs
└── gcn/
    ├── runs/<timestamp>/     # Hydra logs for GCN single runs
    └── sweeps/<timestamp>/   # Hydra logs for GCN sweeps

mlruns/
├── 1/                        # bertgcn_finetuning (BERT experiments)
├── 2/                        # bertgcn_training (GCN experiments)
└── ...                       # Other experiments
```

### Naming Convention

Model directories include key varying parameters: `m{mix_factor}` (e.g., `m0.25`, `m0.5`, `m0.75`)

- `m`: GCN mix factor (BERT-GCN blend weight)

This keeps names concise while differentiating runs by the swept parameter.

### For Sweeps

Each sweep job creates its own `<timestamp>/` directory under `hydra/gcn/sweeps/` with parameter-specific subdirs.

---

## 📈 MLflow Tracking

All experiments are tracked in a single canonical MLflow store at `mlruns/` in the project root:

```bash
# View all experiments (BERT and GCN)
poetry run mlflow ui --backend-store-uri mlruns
```

### Store Layout

```
mlruns/
├── 0/                          # Default experiment (.trash)
├── 1/                          # bertgcn_finetuning (BERT experiments)
│   ├── <run_id>/
│   │   ├── artifacts/          # Models, configs, metrics
│   │   ├── meta.yaml           # Run metadata
│   │   ├── params/             # Logged parameters
│   │   └── tags/               # Run tags
└── 2/                          # bertgcn_training (GCN experiments)
    ├── <run_id>/
    │   ├── artifacts/
    │   ├── meta.yaml
    │   └── ...
```

### What Gets Logged

- **Parameters**: All Hydra config values (hparams, gcn, dataset, etc.)
- **Metrics**: Training/validation loss, accuracy, F1, precision, recall per epoch
- **Artifacts**: Final models, label encoders, config files
- **Tags**: Experiment type, timestamp, hyperparameters summary

Compare runs across BERT and GCN experiments, download models, and analyze performance.

---

## 🧭 Document-Level Interpretability for Precedents Detection

BertGCN provides three approaches to explain predictions by identifying influential documents (precedents):

### A. Neighbor Scoring (`poetry run interpret-docs-neighbors`)
- Fast method using edge weights and GCN probabilities
- Output: `outputs/gcn/interpret/document_influence.csv`

### B. Integrated Gradients (`poetry run interpret-docs-ig`)
- Gradient-based attribution over document features
- Output: `outputs/gcn/interpret/document_influence_ig.csv`

### C. SHAP-style Perturbation (`poetry run interpret-docs-shap`)
- Leave-one-out edge removal analysis
- Output: `outputs/gcn/interpret/document_influence_shap.csv`

Configure via `interpretation.top_k` (default 5) and other settings.

---

## 🧪 Testing

Run the test suite:

```bash
poetry run pytest tests/
```

Tests cover config behavior, graph building, and training stability.

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