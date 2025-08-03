"""
Core graph construction algorithms for document-word graphs.

This module contains the essential algorithms for building heterogeneous graphs
with TF-IDF and PMI edge weights, extracted for reusability and simplicity.
"""

import logging
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_vocabulary(
    tokenized_texts: List[List[str]], min_freq: int = 1
) -> Tuple[List[str], Dict[str, int]]:
    """Build vocabulary from tokenized texts."""
    word_counter = Counter(
        word for text_tokens in tokenized_texts for word in text_tokens
    )

    if min_freq > 1:
        word_counter = {
            word: count for word, count in word_counter.items() if count >= min_freq
        }

    vocab = list(word_counter.keys())
    word2id = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2id


def calculate_word_doc_counts(
    tokenized_texts: List[List[str]], vocab: List[str]
) -> Dict[str, int]:
    """Calculate document frequency for each word."""
    vocab_set = set(vocab)
    word_in_docs = defaultdict(set)

    for doc_idx, doc_tokens in enumerate(tokenized_texts):
        doc_vocab = vocab_set.intersection(doc_tokens)
        for word in doc_vocab:
            word_in_docs[word].add(doc_idx)

    return {word: len(docs) for word, docs in word_in_docs.items()}


def generate_windows(
    tokenized_texts: List[List[str]], window_size: int
) -> List[List[str]]:
    """Generate sliding windows for PMI calculation."""
    windows = []
    for words in tokenized_texts:
        if len(words) > 0:
            windows.append(words[:window_size])
        for j in range(1, len(words) - window_size + 1):
            windows.append(words[j : j + window_size])
    return windows


def calculate_pmi_edges(
    windows: List[List[str]], word2id: Dict[str, int], train_size: int
) -> Tuple[List[int], List[int], List[float]]:
    """Calculate PMI weights for word-word edges."""
    # Count word occurrences in windows
    word_counts = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word in appeared or word not in word2id:
                continue
            word_counts[word] += 1
            appeared.add(word)

    # Count word pair co-occurrences
    pair_counts = defaultdict(int)
    for window in windows:
        for i in range(1, len(window)):
            for j in range(i):
                word_i, word_j = window[i], window[j]
                if word_i not in word2id or word_j not in word2id or word_i == word_j:
                    continue

                id_i, id_j = word2id[word_i], word2id[word_j]
                pair_counts[f"{id_i},{id_j}"] += 1
                pair_counts[f"{id_j},{id_i}"] += 1

    # Calculate PMI and build edges
    rows, cols, weights = [], [], []
    num_windows = len(windows)
    vocab = list(word2id.keys())

    for pair_key, count in pair_counts.items():
        id_i, id_j = map(int, pair_key.split(","))
        word_i, word_j = vocab[id_i], vocab[id_j]

        freq_i = word_counts[word_i]
        freq_j = word_counts[word_j]

        pmi = log((count / num_windows) / (freq_i * freq_j / (num_windows**2)))

        if pmi > 0:
            rows.append(train_size + id_i)
            cols.append(train_size + id_j)
            weights.append(pmi)

    return rows, cols, weights


def calculate_tfidf_edges(
    tokenized_texts: List[List[str]],
    word2id: Dict[str, int],
    word_doc_counts: Dict[str, int],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    dataset_size: int,
    vocab_size: int,
) -> Tuple[List[int], List[int], List[float]]:
    """Calculate TF-IDF weights for document-word edges."""
    # Calculate document-word frequencies
    doc_word_freq = defaultdict(int)
    all_indices = train_indices + val_indices + test_indices

    for doc_id in all_indices:
        words = tokenized_texts[doc_id]
        for word in words:
            if word in word2id:
                doc_word_freq[f"{doc_id},{word2id[word]}"] += 1

    rows, cols, weights = [], [], []
    train_size = len(train_indices)

    # Process each split
    splits = [
        (train_indices, 0),
        (val_indices, train_size + vocab_size),
        (test_indices, train_size + vocab_size + len(val_indices)),
    ]

    for indices, base_offset in splits:
        for split_idx, doc_id in enumerate(indices):
            words = tokenized_texts[doc_id]
            seen_words = set()

            for word in words:
                if word in seen_words or word not in word2id:
                    continue

                word_id = word2id[word]
                freq = doc_word_freq[f"{doc_id},{word_id}"]
                idf = log(dataset_size / word_doc_counts[word])

                rows.append(base_offset + split_idx)
                cols.append(train_size + word_id)
                weights.append(freq * idf)
                seen_words.add(word)

    return rows, cols, weights


def create_splits(
    dataset_size: int, test_split: float, val_split: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Create train/validation/test splits."""
    indices = np.arange(dataset_size)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(dataset_size * (1 - test_split - val_split))
    val_size = int(dataset_size * val_split)

    return (
        indices[:train_size].tolist(),
        indices[train_size : train_size + val_size].tolist(),
        indices[train_size + val_size :].tolist(),
    )


def build_adjacency_matrix(
    pmi_edges: Tuple[List[int], List[int], List[float]],
    tfidf_edges: Tuple[List[int], List[int], List[float]],
    node_size: int,
) -> csr_matrix:
    """Build sparse adjacency matrix from edge lists."""
    all_rows = pmi_edges[0] + tfidf_edges[0]
    all_cols = pmi_edges[1] + tfidf_edges[1]
    all_weights = pmi_edges[2] + tfidf_edges[2]

    return csr_matrix((all_weights, (all_rows, all_cols)), shape=(node_size, node_size))
