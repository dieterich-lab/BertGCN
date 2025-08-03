"""
Graph utilities for BertGCN framework.

Essential graph processing functions moved from legacy utils.py.
These functions handle adjacency matrix operations and data loading.
"""

import pickle as pkl
import sys
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp


def sample_mask(idx: List[int], length: int) -> np.ndarray:
    """Create boolean mask array."""
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_corpus(
    dataset_path: str,
) -> Tuple[
    sp.spmatrix,
    sp.spmatrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """
    Load graph corpus data from pickle files.

    Returns:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size
    """
    names = ["x", "y", "vx", "vy", "tx", "ty", "allx", "ally", "adj"]
    objects = []

    for name in names:
        file_path = f"{dataset_path}.{name}"
        with open(file_path, "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, vx, vy, tx, ty, allx, ally, adj = tuple(objects)

    # Convert sparse matrices to dense for labels
    y = y.toarray()
    vy = vy.toarray()
    ty = ty.toarray()

    # Stack features and labels
    features = sp.vstack((allx, vx, tx)).tolil()
    labels = np.vstack((ally, vy, ty))

    # Calculate sizes
    train_size = x.shape[0]
    val_size = vx.shape[0]
    test_size = tx.shape[0]

    # Create indices
    idx_train = list(range(len(y)))
    idx_val = list(range(allx.shape[0], allx.shape[0] + val_size))
    idx_test = list(
        range(allx.shape[0] + val_size, allx.shape[0] + val_size + test_size)
    )

    # Create masks
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # Create label matrices
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # Make adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return (
        adj,
        features,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
        train_size,
        test_size,
    )
