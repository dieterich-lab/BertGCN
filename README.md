# BertGCN

BertGCN: Research prototype for document/word graph construction with BERT and GCN.

This project builds a document/word graph with document-word (TF-IDF) and word-word edges (PMI) for text classification using BERT and Graph Convolutional Networks.

## Setup

Install dependencies using Poetry:

```bash
poetry install
```

## Usage

To build the document/word graph:

```bash
poetry run python build_graph.py
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- SciPy
- scikit-learn
- NetworkX
- NLTK
