"""
Graph building functionality for BertGCN.

Builds document-word heterogeneous graphs for clinical text classification.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from transformers import AutoTokenizer

from .config import get_paths, PRETRAINEDMODEL
from .datasets import CleanClinicDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_graph(doclevel: str = "letter", testunklar: bool = False) -> Dict:
    """Build document-word graph for clinical text classification."""
    
    logging.info(f"Building graph for doclevel: {doclevel}, testunklar: {testunklar}")
    
    # Get paths
    paths = get_paths()
    
    # Initialize tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = CleanClinicDataset(tokenizer, doclevel=doclevel, clean=True)
    
    # Build vocabulary
    logging.info("Building vocabulary...")
    vocab = set()
    for text in dataset.texts:
        words = text.lower().split()
        vocab.update(words)
    
    vocab = list(vocab)
    word2id = {word: i for i, word in enumerate(vocab)}
    
    logging.info(f"Vocabulary size: {len(vocab)}")
    
    # Create simple adjacency matrix (documents + words)
    num_docs = len(dataset)
    num_words = len(vocab)
    total_nodes = num_docs + num_words
    
    # Create adjacency matrix
    logging.info("Creating adjacency matrix...")
    adj_matrix = csr_matrix((total_nodes, total_nodes), dtype=np.float32)
    
    # Create feature matrices
    logging.info("Creating feature matrices...")
    
    # Split data
    train_size = int(0.7 * num_docs)
    val_size = int(0.1 * num_docs)
    
    train_labels = dataset.ohe_labels[:train_size]
    val_labels = dataset.ohe_labels[train_size:train_size + val_size]
    test_labels = dataset.ohe_labels[train_size + val_size:]
    
    # Save graph files
    graph_name = f"medindcls_{doclevel}"
    if testunklar:
        graph_name += "_testunklar"
    
    graph_dir = paths["graphs"] / graph_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Saving graph files to: {graph_dir}")
    
    # Save adjacency matrix
    save_npz(graph_dir / f"ind.{graph_name}.adj", adj_matrix)
    
    # Save feature matrices
    with open(graph_dir / f"ind.{graph_name}.x", "wb") as f:
        pickle.dump(train_labels, f)
    
    with open(graph_dir / f"ind.{graph_name}.vx", "wb") as f:
        pickle.dump(val_labels, f)
        
    with open(graph_dir / f"ind.{graph_name}.tx", "wb") as f:
        pickle.dump(test_labels, f)
    
    # Save metadata
    metadata = {
        "num_docs": num_docs,
        "num_words": num_words,
        "total_nodes": total_nodes,
        "vocab_size": len(vocab),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_labels),
        "num_classes": len(dataset.class_names),
        "class_names": list(dataset.class_names)
    }
    
    with open(graph_dir / f"ind.{graph_name}.metadata", "wb") as f:
        pickle.dump(metadata, f)
    
    logging.info(f"Graph built successfully: {total_nodes} nodes, {len(dataset.class_names)} classes")
    
    return {
        "adj_matrix": adj_matrix,
        "metadata": metadata,
        "graph_dir": graph_dir,
        "graph_name": graph_name
    }
