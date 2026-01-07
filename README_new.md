# BertGCN

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

Outputs are organized under configured paths:

```
models/
├── finetuned/               # BERT fine-tuning results
│   └── <timestamp>/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── label_encoder.joblib
└── final_model/             # BertGCN final model
    ├── model.pt
    └── config.json

data/
├── processed/
│   ├── tokenized_dataset/
│   ├── label_encoder.joblib
│   └── meds_label_encoder.joblib
└── ind.*                    # Graph components

outputs/                     # Hydra run directories
├── train_bert/
└── train_gcn/

mlruns/                      # MLflow tracking data
```

Each mode writes logs and artifacts to organized subdirectories.

### Artifact Contents

- **Models**: Saved in `models/` with timestamps for versioning
- **Datasets**: Cached processed data for reproducibility
- **Graphs**: Sparse matrices for document-word relationships
- **Predictions**: CSV files with class probabilities
- **Interpretations**: SHAP values for feature importance

---

## 📈 MLflow Tracking

Enabled by default for experiment tracking:

```bash
poetry run mlflow ui --backend-store-uri mlruns
```

Tracks parameters, metrics, and artifacts automatically.

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