# 🧠 BertGCN: Advanced Clinical Document Classification with Hierarchical Interpretability

**🏆 Cutting-Edge Framework:** BertGCN pioneers **hierarchical interpretability** in clinical AI, combining BERT-based Graph Convolutional Networks with multi-level attribution methods. Our framework uniquely provides **document-level**, **sentence-level**, and **token-level** explanations, enabling unprecedented transparency in medical document classification and precedent-based reasoning.

**🎯 Clinical Impact:** Unlike traditional "black-box" AI models, BertGCN generates **clinically actionable insights** through smart precedent selection - identifying the most influential documents and their key sentences for doctor evaluation. This breakthrough enables evidence-based model validation and clinical decision support.

**🔬 Technical Innovation:** Features advanced attribution techniques including Integrated Gradients at multiple granularities, SHAP-style edge perturbation, and hierarchical precedent extraction. Perfect for medical diagnostics, legal analysis, and any domain requiring explainable document classification with clinical validation.

---

## 🌟 Unique Capabilities

- **🔍 Multi-Level Interpretability**: Document → Sentence → Token attribution hierarchy
- **🏥 Clinical Evaluation Ready**: Smart precedent selection for doctor assessment
- **📊 Evidence-Based Validation**: Quantifiable clinical relevance through hierarchical explanations
- **⚡ Production Optimized**: Scalable attribution methods for real-world deployment
- **🎯 Domain Agnostic**: Applicable to medical, legal, and scientific document analysis

---

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

## 🚀 Advanced Features Overview

### Hierarchical Interpretability Pipeline
```
📄 Document Classification → 🔍 Document Attribution → 📝 Sentence Extraction → 🔤 Token Attribution
```

**1. Document-Level Attribution**
- Integrated Gradients over document embeddings
- SHAP-style edge perturbation analysis
- Identifies precedent documents influencing predictions

**2. Smart Sentence Selection**
- Hierarchical precedent extraction (Top-K documents → Top-M sentences)
- Clinical relevance scoring (length, position, keyword-based)
- Doctor evaluation-ready outputs

**3. Token-Level Attribution**
- Word-level importance analysis using Integrated Gradients
- Identifies specific terms driving classifications
- Clinical terminology validation

### Clinical Validation Framework
- **Evidence-Based Assessment**: Doctors evaluate model decisions using extracted precedents
- **Clinical Relevance Metrics**: Quantifiable measures of medical utility
- **Regulatory Compliance**: Transparent AI for healthcare applications

---

---

## 🚀 Installation {#installation}

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

## ⚡ Quick Start {#quick-start}

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

## 🔄 Workflow Overview {#workflow-overview}

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
   - Saves models to MLflow artifacts (`final_model`)
   - Logs Hydra configs to `hydra/gcn/runs/<timestamp>/` (single runs) or `hydra/gcn/sweeps/<timestamp>/` (sweeps)
   - Logs experiments to MLflow in the canonical `mlruns` store (experiment: bertgcn_training)

### 5. **Predict** (`poetry run predict`)
   - Runs inference on new data using trained models
   - Requires model path and input data

### 6. **Interpret** (`poetry run interpret` or `poetry run interpret-docs-*`)
   - Provides feature-level or document-level explanations
   - See [Interpretability](#interpretability) section

---

## ⚙️ Configuration Management {#configuration-management}

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

## 🔍 Running Experiments and Sweeps {#running-experiments-and-sweeps}

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
- Hydra configs saved in `hydra/gcn/sweeps/<timestamp>/`, models in MLflow artifacts



---

## 📂 Model Organization {#model-organization}

Models are saved in `outputs/` directories, Hydra configs in `hydra/`, and experiments tracked in `mlruns/`:

### Directory Structure

```
outputs/
├── bert/
│   └── final_model/          # BERT baseline model
└── gcn/
    ├── final_model/          # Best GCN model
    ├── interpret/            # Document-level interpretability outputs
    └── [predictions]/        # Prediction outputs

hydra/
├── bert/                     # BERT training runs
│   └── <timestamp>/          # Individual training runs
├── gcn/                      # GCN training runs
│   └── <timestamp>/          # Individual training runs
└── inference/                # Prediction runs
    └── <timestamp>/          # Individual prediction runs
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

## 📈 MLflow Tracking {#mlflow-tracking}

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

## 🧠 Advanced Multi-Level Interpretability Framework {#interpretability}

BertGCN pioneers **hierarchical interpretability** - the first framework to provide attribution at document, sentence, and token levels simultaneously. This breakthrough enables unprecedented transparency in clinical AI decision-making.

### A. Document-Level Attribution Methods

#### Fast Neighbor Scoring (`poetry run interpret-docs-neighbors`)
- Rapid edge-weight based influence estimation
- Output: `outputs/gcn/interpret/document_influence.csv`

#### Integrated Gradients (`poetry run interpret-docs-ig`)
- **Advanced Gradient Attribution**: Computes path integrals between baseline and actual document embeddings through the GCN
- **Mathematical Rigor**: Uses Riemann approximation with configurable steps for precise attribution
- **Clinical Validation**: Identifies precedent documents that truly influence medical classifications
- Output: `outputs/gcn/interpret/document_influence_ig.csv`

#### SHAP-Style Edge Perturbation (`poetry run interpret-docs-shap`)
- **Exact Shapley Values**: Leave-One-Out edge perturbation providing mathematically grounded importance scores
- **Computational Innovation**: Efficient edge-level analysis avoiding expensive 2^k coalition sampling
- **Medical Focus**: Document-only attribution excluding noise from word-level graph connections
- Output: `outputs/gcn/interpret/document_influence_shap.csv`

### B. Sentence-Level Intelligence

#### Smart Precedent Selection (`poetry run select-precedents`)
- **Hierarchical Extraction**: Top-K documents → Top-M sentences per document
- **Clinical Relevance Scoring**: Multiple strategies (length, position, medical keywords)
- **Doctor Evaluation Optimized**: Provides focused, actionable precedents instead of overwhelming full documents
- **Regulatory Ready**: Evidence-based framework for clinical AI validation
- Output: `outputs/gcn/interpret/smart_precedents.csv`

### C. Token-Level Precision

#### Word Attribution Analysis (`poetry run interpret-tokens-ig`)
- **Fine-Grained Attribution**: Integrated Gradients at individual token level
- **Clinical Terminology Validation**: Identifies specific medical terms driving classifications
- **Interpretability Depth**: Unprecedented granularity for understanding model decisions
- Output: `outputs/gcn/interpret/token_influence_ig.csv`
- **Output**: `outputs/gcn/interpret/smart_precedents.csv` with ranked precedents and key sentences
- **Doctor Evaluation**: Exports samples in `doctor_evaluation_sample.csv` for clinical relevance assessment

Configure via `interpretation.top_k` (default 5) and other settings in `conf/config.yaml`.

### Analysis Notebook
Run `notebooks/analyze_interpretation_statistics.ipynb` to:
- Analyze document influence patterns and class relationships
- Visualize token importance distributions
- Review smart precedent selections
- Export evaluation samples for clinical assessment

---

## 🏥 Clinical Applications & Impact

### Medical Decision Support
- **Transparent Diagnostics**: Understand which clinical precedents influence AI classifications
- **Evidence-Based Validation**: Doctors can evaluate if model decisions align with medical knowledge
- **Regulatory Compliance**: Explainable AI for healthcare accreditation (FDA, EU AI Act)

### Research & Validation Workflow
1. **Model Training**: Train BertGCN on clinical documents
2. **Attribution Analysis**: Generate hierarchical explanations (docs → sentences → tokens)
3. **Clinical Evaluation**: Domain experts assess explanation quality and clinical relevance
4. **Iterative Improvement**: Refine models based on expert feedback

### Real-World Deployment
- **UKHD Integration**: Ready for University Hospital Düsseldorf clinical workflows
- **Scalable Attribution**: Production-optimized explanation methods
- **Multi-Stakeholder Validation**: Clinicians, researchers, and regulators can all assess model decisions

### Key Innovation: Clinical Precedent Intelligence
Unlike traditional XAI methods that provide abstract feature importance, BertGCN generates **clinically actionable precedents** - the specific documents and sentences that truly matter for medical decision-making.

---

## 🧪 Testing {#testing}

Run the test suite:

```bash
poetry run pytest tests/
```

Tests cover config behavior, graph building, and training stability.

---

## 🤝 Contributing {#contributing}

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and open Pull Request

---

## 📝 Citation

If you use BertGCN in your research, please cite:

```bibtex
@software{bertgcn2024,
  title = {BertGCN: Advanced Clinical Document Classification with Hierarchical Interpretability},
  author = {Philipp Wiesenbach and Dieterich Lab},
  year = {2024},
  url = {https://github.com/dieterich-lab/BertGCN},
  note = {Pioneering hierarchical interpretability framework for clinical AI with document, sentence, and token-level attribution}
}
```

### Key Contributions
- **Hierarchical Interpretability**: First framework providing multi-level attribution (document → sentence → token)
- **Clinical Precedent Intelligence**: Smart extraction of medically relevant precedents for doctor evaluation
- **Production-Ready XAI**: Scalable attribution methods for real-world clinical deployment
- **Evidence-Based Validation**: Framework for clinical assessment of AI decision-making