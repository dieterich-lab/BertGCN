# Corpus-Level Interpretability for Precedent Detection in Medical Document Classification using BERT-GCN

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
- [Interpretability](#document-level-interpretability-for-precedents-detection)
- [Results](#results)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Installation

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

## Quick Start

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

## Workflow Overview

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

## Configuration Management

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

## Running Experiments and Sweeps

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

### SLURM-Based Distributed Sweeps

For large-scale hyperparameter sweeps on GPU clusters, use the provided SLURM scripts:

```bash
# Phase 1: Initial exploration (24 configs, ~7.6 hours)
sbatch slurm/train_gcn_sweep.sh

# Phase 2: Extended sweep with larger models (144 configs, ~45.6 hours)
sbatch slurm/train_gcn_sweep_p2.sh
```

- **Phase 1**: Explores core hyperparameters (mix factor, hidden dims, dropout)
- **Phase 2**: Builds on Phase 1 results with larger architectures and A100 GPU
- Jobs run in parallel on SLURM cluster with A100 GPUs (80GB VRAM)
- All results automatically logged to MLflow for analysis

### BERT Baseline Performance
Plain BERT fine-tuning (no GCN) establishes the baseline for comparison with BertGCN models:

- **Test Accuracy**: 91.3%
- **Validation Accuracy**: 93.7% (best epoch)
- **Training**: 100 epochs with early stopping (patience=7), LR=5e-5, batch_size=96
- **Model**: MedBERT (German medical language model) with classification head
- **Dataset**: Same clinical indication classification task

This baseline shows that BertGCN models significantly outperform plain BERT, with improvements of 4-6% in test accuracy.

### Recent Sweep Results

#### Phase 1: Initial Exploration (24 configs)
Analysis of a 24-configuration hyperparameter sweep exploring mix factor (0.4, 0.5, 0.65, 0.8), hidden dimensions (250, 300), and dropout rates (0.2, 0.3, 0.4):

| Rank | Mix Factor | Hidden Dim | Dropout | Test Acc | Notes |
|------|------------|------------|---------|----------|-------|
| 1-2 | 0.8 | 300 | 0.2, 0.3 | 97.4% | **Best performers** |
| 3 | 0.8 | 300 | 0.4 | 97.0% | |
| 4-5 | 0.65 | 300 | 0.3, 0.4 | 96.9% | |
| 6 | 0.8 | 250 | 0.3 | 96.7% | |
| 7-9 | 0.4, 0.65 | 250 | 0.2, 0.4 | 96.5% | |
| 10-16 | Various | 250 | Various | 96.3% | |
| 17-23 | Various | 300 | Various | 95.6-96.3% | Lower hidden dim configs |
| 24 | 0.65 | 300 | 0.2 | 95.0% | **Worst performer** |

**Key Findings:**
- **Hidden dimension 300** achieved best peak performance (97.4% vs 96.7% for 250)
- **Mix factor 0.8** showed most consistent high performance across dropout rates
- **Dropout 0.3** was optimal in most configurations
- **Early stopping** prevented overfitting (13 epochs trained vs 70 max configured)
- **Test accuracy range:** 95.0% - 97.4% (2.4% spread)

#### Phase 2: Targeted Extension (144 configs, Completed)
Building on Phase 1 results, exploring larger architectures with A100 GPU (80GB VRAM):

**Parameter Ranges:**
- **GCN Layers:** 2-3 (vs 2 fixed in Phase 1)
- **Hidden Dimensions:** 400-600 (vs 250-300 in Phase 1)
- **Mix Factor:** 0.85-0.95 (vs 0.4-0.8 in Phase 1)
- **Dropout:** 0.2-0.3 (optimal range from Phase 1)
- **BERT Learning Rate:** 2e-5, 5e-5 (fine-tuning)
- **Batch Size:** 96, 128 (memory-optimized)

**Top Performers (97.0%+ Test Accuracy):**

| Rank | Mix Factor | GCN Layers | Hidden Dim | Dropout | BERT LR | Batch Size | Test Acc | Notes |
|------|------------|------------|------------|---------|---------|------------|----------|-------|
| 1-2 | 0.85 | 2, 3 | 600 | 0.3 | 5e-5 | 96 | 97.6% | **Best performers** - Larger models with higher BERT LR |
| 3-4 | 0.85 | 2, 3 | 600 | 0.2 | 5e-5 | 96 | 97.4% | Strong performance with lower dropout |
| 5-8 | 0.85, 0.9 | 2, 3 | 400-600 | 0.2-0.3 | 2e-5 | 96, 128 | 97.2% | Consistent high performance across configs |
| 9-10 | 0.85, 0.95 | 2 | 600 | 0.3 | 2e-5, 5e-5 | 128 | 97.0% | |

**Key Findings:**
- **Hidden dimension 600** achieved best peak performance (97.6% vs lower for smaller dims)
- **Mix factor 0.85** showed most consistent high performance, slightly better than 0.9-0.95
- **GCN layers 3** achieved comparable performance to 2 layers with larger hidden dims
- **BERT LR 5e-5** outperformed 2e-5 in top configurations
- **Batch size 96** generally better than 128 for convergence
- **Dropout 0.3** remained optimal in most configurations
- **Test accuracy range:** 94.3% - 97.6% (3.3% spread, improved from Phase 1)
- **Runtime:** ~48 hours total (144 jobs × ~20 min avg, sequential on single GPU)

All results logged to MLflow for detailed comparison and model artifact retrieval.

---

## Model Organization

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

## MLflow Tracking

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

## Document-Level Interpretability for Precedents Detection

BertGCN provides multiple approaches to explain predictions by identifying influential documents (precedents):

### Recent Analysis Findings

Recent analysis of hierarchical precedents on a clinical dataset revealed key insights into the interpretability framework:

- **Scale**: Analyzed 8,097 precedents across 2,699 unique target documents, with an average of 3 precedents per document
- **Reliability**: Achieved 98.4% sentence extraction success rate and 100% token extraction success rate
- **Clinical Relevance**: Precedents provide additive information for classification, helping understand disease progression, risk stratification, and treatment decisions
- **Utility**: Precedents demonstrate how similar patient profiles can inform clinical decision-making, such as intensifying therapy or monitoring for disease evolution

### A. Neighbor Scoring (`poetry run interpret-docs-neighbors`)
- Fast method using edge weights and GCN probabilities
- Output: `outputs/gcn/interpret/document_influence.csv`

### B. Integrated Gradients (`poetry run interpret-docs-ig`)
- Gradient-based attribution over document features
- Custom implementation (no external XAI library used)
- Interpolates between zero baseline and actual document embeddings, computing gradients through the GCN
- Output: `outputs/gcn/interpret/document_influence_ig.csv`

### C. SHAP-style Perturbation (`poetry run interpret-docs-shap`)
- **Leave-One-Out (LOO) Edge Perturbation**: For each target document, systematically removes each incoming edge from neighboring documents and measures the change in prediction probability
- **Computational Optimization**: Instead of expensive kernel SHAP (2^k coalitions), uses efficient edge-level perturbations that give exact Shapley values under feature independence assumptions
- **Document-Only Attribution**: Filters to only consider document-to-document edges, excluding word nodes from the analysis
- **Output**: `outputs/gcn/interpret/document_influence_shap.csv` with neighbor importance scores

### D. Token-Level Attribution (`poetry run interpret-tokens-ig`)
- **Word-Level Importance**: Computes Integrated Gradients at the token level to identify which individual words/tokens are most important for classification decisions
- **Clinical Relevance**: Helps understand what specific terms drive medical classifications
- **Output**: `outputs/gcn/interpret/token_influence_ig.csv` with top tokens and their attribution scores per document

### E. Smart Precedent Selection (`poetry run select-precedents`)
- **Hierarchical IG Approach**: First selects top N most influential documents, then uses token-level IG to find the most important sentences within those documents
- **Two-Pass Algorithm**:
  1. **Pass 1**: Graph-level IG/SHAP to identify top precedent documents
  2. **Pass 2**: Token-level IG on top documents to find influential words, then extract sentences containing those words
- **Clinical Evaluation Ready**: Provides focused sentence-level precedents perfect for doctor evaluation
- **Configurable**: Set `interpretation.top_docs` (default 3), `interpretation.top_sentences_per_doc` (default 2), `interpretation.sentence_scoring` (default "hierarchical")
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

## Results

Recent evaluation of the corpus-level interpretability framework on a clinical document classification dataset demonstrates strong performance and clinical utility:

- **Scale and Coverage**: Analyzed 8,097 precedents across 2,699 unique target documents, with an average of 3 precedents per document
- **Extraction Reliability**: Achieved 98.4% success rate for sentence extraction and 100% for token extraction from influential documents
- **Clinical Relevance**: Precedents exhibit low average clinical term density (0.40 per precedent) but provide valuable additive information for classification tasks
- **Practical Impact**: Precedents enable understanding of disease progression, risk stratification, and treatment decision-making by highlighting similar patient profiles and outcomes

These results validate the framework's effectiveness in providing transparent, precedent-based explanations for medical document classification.

---

## Testing

Run the test suite:

```bash
poetry run pytest tests/
```

Tests cover config behavior, graph building, and training stability.

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and open Pull Request

---

## 📝 Citation

```bibtex
@software{bertgcn2024,
  title = {Corpus-Level Interpretability for Precedent Detection in Medical Document Classification using BERT-GCN},
  author = {Philipp Wiesenbach},
  year = {2024},
  url = {https://github.com/dieterich-lab/BertGCN}
}
```