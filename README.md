# BertGCN Project Structure

This document describes the organized directory structure for the BertGCN project.

## Directory Structure

```
BertGCN/
├── outputs/                    # All generated outputs
│   ├── data/                   # Data storage
│   │   ├── datasets/           # Processed datasets (.pkl files)
│   │   ├── graphs/             # Graph adjacency matrices and features
│   │   └── features/           # Extracted features
│   ├── models/                 # Model storage
│   │   ├── checkpoints/        # Training checkpoints
│   │   ├── finetuned/          # Fine-tuned BERT models
│   │   └── gcn/               # GCN models
│   ├── cache/                  # Temporary cache files
│   └── logs/                   # Log files
├── *.py                       # Source code files
└── README.md                  # This file
```

## File Naming Conventions

### Datasets
- `medindcls_medbert_{doclevel}_clean.pkl` - Cleaned datasets
- `medindcls_medbert_{doclevel}_nomeds_clean.pkl` - Without medications

### Graph Files
- `ind.{dataset}_{doclevel}.{suffix}` - Graph components
- Suffixes: `x`, `y`, `vx`, `vy`, `tx`, `ty`, `allx`, `ally`, `adj`
- `_testunklar` suffix for special test configurations

### Models
- `finetuned/{doclevel}/` - BERT models by document level
- `gcn/{doclevel}/` - GCN models by document level

## Usage

The `config.py` module automatically creates this structure when first imported.
All scripts use centralized path management through `get_paths()`.