"""
Text processing utilities for clinical document analysis.

This module provides common text processing functions that can be reused
across different components of the clinical text analysis pipeline.
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def tokenize_texts_batch(texts: List[str]) -> List[List[str]]:
    """
    Efficiently tokenize a batch of texts by splitting on whitespace.

    Args:
        texts: List of text strings to tokenize

    Returns:
        List of tokenized texts (list of word lists)
    """
    return [text.split() for text in texts]


def build_vocabulary_from_tokens(
    tokenized_texts: List[List[str]], min_freq: int = 1
) -> Tuple[List[str], Dict[str, int]]:
    """
    Build vocabulary from pre-tokenized texts.

    Args:
        tokenized_texts: List of tokenized documents
        min_freq: Minimum frequency threshold for vocabulary inclusion

    Returns:
        Tuple of (vocabulary list, word to ID mapping)
    """
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


def calculate_word_document_frequencies(
    tokenized_texts: List[List[str]], vocab: List[str]
) -> Dict[str, int]:
    """
    Calculate document frequency for each vocabulary word.

    Args:
        tokenized_texts: List of tokenized documents
        vocab: Vocabulary list

    Returns:
        Dictionary mapping words to their document frequencies
    """
    vocab_set = set(vocab)
    word_doc_counts = defaultdict(set)

    for doc_idx, doc_tokens in enumerate(tokenized_texts):
        doc_vocab = vocab_set.intersection(doc_tokens)
        for word in doc_vocab:
            word_doc_counts[word].add(doc_idx)

    return {word: len(docs) for word, docs in word_doc_counts.items()}


def generate_sliding_windows(
    tokenized_texts: List[List[str]], window_size: int = 20
) -> List[List[str]]:
    """
    Generate sliding windows from tokenized texts for co-occurrence analysis.

    Args:
        tokenized_texts: List of tokenized documents
        window_size: Size of sliding window

    Returns:
        List of sliding windows (each window is a list of words)
    """
    windows = []

    for words in tokenized_texts:
        length = len(words)

        # Add first window
        if length > 0:
            windows.append(words[:window_size])

        # Add sliding windows
        for j in range(1, length - window_size + 1):
            windows.append(words[j : j + window_size])

    return windows


def log_processing_stats(
    vocab_size: int, num_documents: int, num_windows: int, logger: logging.Logger = None
) -> None:
    """
    Log statistics about text processing.

    Args:
        vocab_size: Size of vocabulary
        num_documents: Number of documents processed
        num_windows: Number of sliding windows generated
        logger: Logger instance to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Text processing statistics:")
    logger.info(f"  Vocabulary size: {vocab_size:,}")
    logger.info(f"  Documents processed: {num_documents:,}")
    logger.info(f"  Sliding windows generated: {num_windows:,}")
    logger.info(f"  Avg windows per document: {num_windows/max(num_documents, 1):.1f}")
