"""
Modern Document-Word Graph Builder for Clinical Text Classification

This module builds heterogeneous graphs from clinical text datasets where nodes represent
both documents and vocabulary words, with edges weighted by TF-IDF (doc-word) and
PMI (word-word) relationships.
"""

import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from config import get_paths
from entry import PRETRAINEDMODEL
from text_utils import (
    build_vocabulary_from_tokens,
    calculate_word_document_frequencies,
    generate_sliding_windows,
    log_processing_stats,
    tokenize_texts_batch,
)


@dataclass
class GraphConfig:
    """Configuration for graph building parameters."""

    window_size: int = 20
    min_word_freq: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    random_seed: int = 0


class DocumentWordGraphBuilder:
    """
    Builds heterogeneous document-word graphs for clinical text classification.

    The resulting graph has:
    - Document nodes connected to word nodes via TF-IDF weights
    - Word nodes connected to other word nodes via PMI weights
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: GraphConfig = None,
        doclevel: str = "letter",
        clean: bool = True,
    ):
        self.tokenizer = tokenizer
        self.config = config or GraphConfig()
        self.doclevel = doclevel
        self.clean = clean
        self.logger = logging.getLogger(__name__)

        # Will be set during processing
        self.dataset: Optional[CleanClinicDataset] = None
        self.vocab: List[str] = []
        self.word2id: Dict[str, int] = {}
        self.vocab_size: int = 0
        self.embed_dim: int = 0

        # Cache for performance optimization
        self._tokenized_texts: Optional[List[List[str]]] = None

    def load_or_create_dataset(
        self, dataset_file: Path, force_recreate: bool = False
    ) -> CleanClinicDataset:
        """Load existing dataset or create new one."""
        if dataset_file.exists() and not force_recreate:
            self.logger.info(f"Loading dataset from: {dataset_file}")
            with open(dataset_file, "rb") as f:
                dataset = pickle.load(f)
        else:
            self.logger.info("Creating new dataset")
            dataset = CleanClinicDataset(
                tokenizer=self.tokenizer, doclevel=self.doclevel, clean=self.clean
            )
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_file, "wb") as f:
                self.logger.info(f"Saving dataset to: {dataset_file}")
                pickle.dump(dataset, f)

        self.dataset = dataset
        return dataset

    def _get_tokenized_texts(self) -> List[List[str]]:
        """Get cached tokenized texts for performance."""
        if self._tokenized_texts is None:
            self.logger.info("Tokenizing texts for graph building")
            self._tokenized_texts = [text.split() for text in self.dataset.texts]
        return self._tokenized_texts

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from pretrained model."""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.tokenizer.name_or_path
            )
            embed_dim = model.bert.embeddings.word_embeddings.embedding_dim
            del model  # Free memory
            return embed_dim
        except Exception as e:
            self.logger.error(f"Failed to load model for embedding dimension: {e}")
            # Fallback to common BERT dimension
            return 768

    def _build_vocabulary(self) -> Tuple[List[str], Dict[str, int]]:
        """Build vocabulary from dataset texts."""
        self.logger.info("Building vocabulary")

        # Use cached tokenized texts for better performance
        tokenized_texts = self._get_tokenized_texts()
        word_counter = Counter(
            word for text_tokens in tokenized_texts for word in text_tokens
        )

        # Filter by minimum frequency if needed
        if self.config.min_word_freq > 1:
            word_counter = {
                word: count
                for word, count in word_counter.items()
                if count >= self.config.min_word_freq
            }

        vocab = list(word_counter.keys())
        word2id = {word: idx for idx, word in enumerate(vocab)}

        self.vocab = vocab
        self.word2id = word2id
        self.vocab_size = len(vocab)

        return vocab, word2id

    def _calculate_word_document_stats(self) -> Dict[str, int]:
        """Calculate how many documents each word appears in."""
        self.logger.info("Calculating word-document statistics")

        # Use cached tokenized texts and set operations for better performance
        tokenized_texts = self._get_tokenized_texts()
        vocab_set = set(self.vocab)

        word_in_docs = defaultdict(set)
        for doc_idx, doc_tokens in enumerate(tokenized_texts):
            doc_vocab = vocab_set.intersection(doc_tokens)
            for word in doc_vocab:
                word_in_docs[word].add(doc_idx)

        return {word: len(docs) for word, docs in word_in_docs.items()}

    def _generate_sliding_windows(self) -> List[List[str]]:
        """Generate sliding windows for PMI calculation."""
        self.logger.info("Generating sliding windows")

        windows = []
        tokenized_texts = self._get_tokenized_texts()

        for words in tokenized_texts:
            length = len(words)

            # Add first window
            if length > 0:
                windows.append(words[: self.config.window_size])

            # Add sliding windows
            for j in range(1, length - self.config.window_size + 1):
                windows.append(words[j : j + self.config.window_size])

        return windows

    def _calculate_pmi_weights(
        self, windows: List[List[str]]
    ) -> Tuple[List[int], List[int], List[float]]:
        """Calculate PMI weights for word-word connections."""
        self.logger.info("Calculating PMI weights")

        # Count word occurrences in windows
        word_window_counts = defaultdict(int)
        for window in windows:
            appeared = set()
            for word in window:
                if word in appeared or word not in self.word2id:
                    continue
                word_window_counts[word] += 1
                appeared.add(word)

        # Count word pair co-occurrences
        word_pair_counts = defaultdict(int)
        for window in windows:
            for i in range(1, len(window)):
                for j in range(i):
                    word_i, word_j = window[i], window[j]

                    # Skip if words not in vocabulary or identical
                    if (
                        word_i not in self.word2id
                        or word_j not in self.word2id
                        or word_i == word_j
                    ):
                        continue

                    word_i_id = self.word2id[word_i]
                    word_j_id = self.word2id[word_j]

                    # Add both directions
                    pair_key_ij = f"{word_i_id},{word_j_id}"
                    pair_key_ji = f"{word_j_id},{word_i_id}"
                    word_pair_counts[pair_key_ij] += 1
                    word_pair_counts[pair_key_ji] += 1

        # Calculate PMI and build edge lists
        row, col, weights = [], [], []
        num_windows = len(windows)
        train_data_size = len(self.train_indices)

        for pair_key, count in word_pair_counts.items():
            word_i_id, word_j_id = map(int, pair_key.split(","))
            word_i, word_j = self.vocab[word_i_id], self.vocab[word_j_id]

            word_freq_i = word_window_counts[word_i]
            word_freq_j = word_window_counts[word_j]

            # Calculate PMI
            pmi = log(
                (count / num_windows) / (word_freq_i * word_freq_j / (num_windows**2))
            )

            if pmi > 0:  # Only keep positive PMI
                row.append(train_data_size + word_i_id)
                col.append(train_data_size + word_j_id)
                weights.append(pmi)

        return row, col, weights

    def _calculate_tfidf_weights(
        self, word_in_doc_counts: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[float]]:
        """Calculate TF-IDF weights for document-word connections."""
        self.logger.info("Calculating TF-IDF weights")

        # Use cached tokenized texts for better performance
        tokenized_texts = self._get_tokenized_texts()

        # Calculate document-word frequencies for all splits
        doc_word_freq = defaultdict(int)
        all_indices = self.train_indices + self.val_indices + self.test_indices

        for doc_id in all_indices:
            words = tokenized_texts[doc_id]
            for word in words:
                if word in self.word2id:
                    word_id = self.word2id[word]
                    key = f"{doc_id},{word_id}"
                    doc_word_freq[key] += 1

        row, col, weights = [], [], []
        train_data_size = len(self.train_indices)

        # Process each split
        splits = [
            (self.train_indices, 0),
            (self.val_indices, train_data_size + self.vocab_size),
            (
                self.test_indices,
                train_data_size + self.vocab_size + len(self.val_indices),
            ),
        ]

        for indices, base_offset in splits:
            for split_idx, doc_id in enumerate(indices):
                words = tokenized_texts[doc_id]
                doc_word_set = set()

                for word in words:
                    if word in doc_word_set or word not in self.word2id:
                        continue

                    word_id = self.word2id[word]
                    key = f"{doc_id},{word_id}"
                    freq = doc_word_freq[key]

                    # Calculate TF-IDF
                    idf = log(len(self.dataset) / word_in_doc_counts[word])
                    tfidf = freq * idf

                    row.append(base_offset + split_idx)
                    col.append(train_data_size + word_id)
                    weights.append(tfidf)
                    doc_word_set.add(word)

        return row, col, weights

    def _create_data_splits(self, testunklar: bool = False) -> None:
        """Create train/validation/test splits."""
        import random

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        if not testunklar:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)

            train_size = int(
                len(indices) * (1 - self.config.test_split - self.config.val_split)
            )
            val_size = int(len(indices) * self.config.val_split)

            self.train_indices = indices[:train_size].tolist()
            self.val_indices = indices[train_size : train_size + val_size].tolist()
            self.test_indices = indices[train_size + val_size :].tolist()
        else:
            # Special handling for "unklar" labels
            train_val_indices, test_indices = [], []

            for i, example in enumerate(self.dataset):
                if "unklar" in self.dataset.LE.classes_[example["labels"]]:
                    test_indices.append(i)
                else:
                    train_val_indices.append(i)

            np.random.shuffle(train_val_indices)
            val_size = int(len(train_val_indices) * 0.1)

            self.train_indices = train_val_indices[val_size:]
            self.val_indices = train_val_indices[:val_size]
            self.test_indices = test_indices

        self.logger.info(
            f"Dataset splits - Train: {len(self.train_indices)}, "
            f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}"
        )

    def build_graph(
        self, dataset_file: Path, testunklar: bool = False
    ) -> Tuple[csr_matrix, Dict]:
        """
        Build the complete document-word graph.

        Returns:
            Sparse adjacency matrix and metadata dictionary
        """
        # Load dataset and get embedding dimension
        self.load_or_create_dataset(dataset_file)
        self.embed_dim = self._get_embedding_dimension()

        # Create data splits
        self._create_data_splits(testunklar)

        # Build vocabulary and calculate statistics
        self._build_vocabulary()
        word_in_doc_counts = self._calculate_word_document_stats()

        # Generate sliding windows for PMI
        windows = self._generate_sliding_windows()

        # Calculate edge weights
        pmi_row, pmi_col, pmi_weights = self._calculate_pmi_weights(windows)
        tfidf_row, tfidf_col, tfidf_weights = self._calculate_tfidf_weights(
            word_in_doc_counts
        )

        # Combine all edges
        all_rows = pmi_row + tfidf_row
        all_cols = pmi_col + tfidf_col
        all_weights = pmi_weights + tfidf_weights

        # Create adjacency matrix
        node_size = len(self.dataset) + self.vocab_size
        adj_matrix = csr_matrix(
            (all_weights, (all_rows, all_cols)), shape=(node_size, node_size)
        )

        # Create metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "node_size": node_size,
            "vocab": self.vocab,
            "word2id": self.word2id,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "label_classes": self.dataset.LE.classes_,
        }

        self.logger.info(
            f"Graph built successfully: {node_size} nodes, {len(all_weights)} edges"
        )
        return adj_matrix, metadata

    def save_graph_data(
        self,
        adj_matrix: csr_matrix,
        metadata: Dict,
        dataset_name: str,
        testunklar: bool = False,
    ) -> None:
        """Save all graph-related data files."""
        paths = get_paths()
        output_dir = paths.get_graph_path(dataset_name, self.doclevel, testunklar)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"ind.{dataset_name}_{self.doclevel}"
        if testunklar:
            base_name += "_testunklar"

        # Create data matrices
        train_size = metadata["train_size"]
        val_size = metadata["val_size"]
        test_size = metadata["test_size"]
        embed_dim = metadata["embed_dim"]
        vocab_size = metadata["vocab_size"]

        # Training data features and labels
        x = csr_matrix((train_size, embed_dim), dtype=np.float32)
        y = self.dataset.ohe_labels[metadata["train_indices"]]

        # Validation data
        vx = csr_matrix((val_size, embed_dim), dtype=np.float32)
        vy = self.dataset.ohe_labels[metadata["val_indices"]]

        # Test data
        tx = csr_matrix((test_size, embed_dim), dtype=np.float32)
        ty = self.dataset.ohe_labels[metadata["test_indices"]]

        # All training data (train + vocab)
        allx = csr_matrix((train_size + vocab_size, embed_dim), dtype=np.float32)
        ally = np.concatenate(
            [y.toarray(), np.zeros((vocab_size, len(metadata["label_classes"])))]
        )

        # Save all files
        files_to_save = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "tx": tx,
            "ty": ty,
            "allx": allx,
            "ally": ally,
            "adj": adj_matrix,
        }

        for file_suffix, data in files_to_save.items():
            file_path = output_dir / f"{base_name}.{file_suffix}"
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

        # Save metadata
        metadata_path = output_dir / f"{base_name}.metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Graph data saved to {output_dir}")


def main():
    from params import parse_args

    args = parse_args()

    """Example usage of the DocumentWordGraphBuilder."""
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    doclevel = args.doclevel
    testunklar = args.testunklar

    # Initialize tokenizer and graph builder
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    config = GraphConfig(window_size=20, random_seed=0)

    builder = DocumentWordGraphBuilder(
        tokenizer=tokenizer, config=config, doclevel=doclevel, clean=True
    )

    # Build graph
    paths = get_paths()
    dataset_file = paths.get_dataset_path(doclevel, "medbert", clean=True)
    adj_matrix, metadata = builder.build_graph(dataset_file, testunklar=testunklar)

    # Save graph data
    dataset_name = f"medindcls_{doclevel}"
    builder.save_graph_data(adj_matrix, metadata, dataset_name, testunklar=testunklar)

    print(f"Graph built successfully!")
    print(f"Nodes: {metadata['node_size']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(
        f"Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}"
    )


if __name__ == "__main__":
    main()
