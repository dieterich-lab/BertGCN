"""
Build a document-word graph with TF-IDF and PMI edges for a BertGCN model.

This script creates a graph where nodes represent documents and words, and edges
represent their relationships using PMI (Pointwise Mutual Information) for word-word
edges and TF-IDF for document-word edges.

When installed in development mode, changes to this file will be immediately reflected
without needing to reinstall the package.
"""

import logging
import os
import pickle
import pickle as pkl
import random
import time  # Used in log_step
from collections import Counter, defaultdict
from contextlib import contextmanager
from math import log
from pathlib import Path
from typing import (
    Any,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

# Try to import required packages, but handle gracefully if they're not available
import numpy as np
import torch
import typer
from scipy.sparse import csr_matrix, lil_matrix
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bertgcn.clinic_datasets import CleanClinicDataset
from bertgcn.config import DEFAULT_MODEL_PATH
from bertgcn.core import get_logger
from bertgcn.params import BertGCNParameters
from bertgcn.utils import *

# Get logger
logger = get_logger(__name__)


# Set seeds for reproducibility
def set_seed(seed: int = 0) -> None:
    """Set seeds for reproducibility."""
    from bertgcn.core import setup_environment

    setup_environment(seed)
    # Additional CUDA-specific settings, only if torch.cuda is available
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        except (AttributeError, ImportError):
            logger.warning(
                "CUDA support unavailable in PyTorch. Running in CPU-only mode."
            )


@contextmanager
def log_step(message: str) -> Generator[None, None, None]:
    """Context manager to log the start and end of a step with timing information."""
    import time

    start_time = time.time()
    logger.info(f"Starting: {message}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed: {message} in {elapsed:.2f}s")


class GraphBuilder:
    """Class for building a document-word graph with TF-IDF and PMI edges."""

    def __init__(
        self,
        dataset,
        embed_dim: int,
        dataname: str,
        window_size: int = 20,
        batch_size: int = 1000,
        bidirectional_tfidf: bool = True,
        min_pmi_threshold: float = 0,
    ):
        """
        Initialize the GraphBuilder.

        Args:
            dataset: The dataset containing documents and labels
            embed_dim: Embedding dimension for the sparse matrices
            dataname: Name of the dataset for saving files
            window_size: Size of sliding window for word co-occurrence
            batch_size: Number of documents to process at once
            bidirectional_tfidf: Whether to add bidirectional TF-IDF edges
            min_pmi_threshold: Minimum PMI value threshold (default: 0)
        """
        self.dataset = dataset

        def main(
            doclevel: str = typer.Option("letter", help="Document level"),
            bertmodel: str = typer.Option("medbert", help="BERT model"),
            window_size: int = typer.Option(20, help="Window size"),
            batch_size: int = typer.Option(1000, help="Batch size"),
            bidirectional_tfidf: bool = typer.Option(True, help="Bidirectional TF-IDF"),
            min_pmi: float = typer.Option(0.0, help="Minimum PMI"),
            seed: int = typer.Option(42, help="Random seed"),
            testunklar: bool = typer.Option(False, help="Test unclear samples"),
        ) -> None:
            """
            Main function to build and save the document-word graph.
            Parses CLI arguments, loads model and data, builds graph, and saves results.
            """
            params = BertGCNParameters(
                doclevel=doclevel,
                bertmodel=bertmodel,
                window_size=window_size,
                batch_size=batch_size,
                bidirectional_tfidf=bidirectional_tfidf,
                min_pmi=min_pmi,
                seed=seed,
                testunklar=testunklar,
            )
            set_seed(params.seed)

            dataname = f"medindcls_{params.doclevel}"
            logger.info(f"Building graph for dataset {dataname}")
            logger.info(f"Arguments: {params}")

            # Model path selection (fallback to DEFAULT_MODEL_PATH if needed)
            try:
                from bertgcn.config import MODEL_PATHS, DEFAULT_MODEL_PATH
                model_path = MODEL_PATHS.get(params.bertmodel, DEFAULT_MODEL_PATH)
            except ImportError:
                logger.error("Could not import MODEL_PATHS or DEFAULT_MODEL_PATH from bertgcn.config. Using default path.")
                model_path = "bert-base-uncased"

            try:
                with log_step("Loading BERT model and tokenizer"):
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    embed_dim = model.bert.embeddings.word_embeddings.embedding_dim
                    logger.info(f"Embedding dimension: {embed_dim}")
                    del model  # Free memory
            except Exception as e:
                logger.error(f"Failed to load model or tokenizer: {e}")
                raise SystemExit(1)

            dataset = load_or_create_dataset(tokenizer, params.doclevel)
            train_idx, val_idx, test_idx, train_dataset, val_dataset, test_dataset = (
                split_dataset(dataset, params)
            )

            graph_builder = GraphBuilder(
                dataset=dataset,
                embed_dim=embed_dim,
                dataname=dataname,
                window_size=params.window_size,
                batch_size=params.batch_size,
                bidirectional_tfidf=params.bidirectional_tfidf,
                min_pmi_threshold=params.min_pmi,
            )

            graph_components = graph_builder.build_graph(
                train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx
            )

            save_graph_components(
                graph_components, dataname, params.doclevel, params.testunklar
            )

            logger.info("Graph building complete!")
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, num_docs)

                # Process this batch of documents
                for doc_idx in range(start_idx, end_idx):
                    words = set(self.dataset.texts[doc_idx].split())
                    word_ids = [
                        self.word2id[word] for word in words if word in self.word2id
                    ]

                    if word_ids:
                        self.word_doc_matrix[word_ids, doc_idx] = 1

                if batch_idx % 10 == 0 or batch_idx == num_doc_batches - 1:
                    logger.info(
                        f"Processed document batch {batch_idx+1}/{num_doc_batches}"
                    )

            # Convert to more efficient format
            logger.info("Converting to CSR format for efficiency...")
            self.word_doc_matrix = self.word_doc_matrix.tocsr()

            # Extract word-in-docs information efficiently
            self.word_in_doc_counts = {}

            for word_idx, word in enumerate(self.vocab):
                # Get all documents containing this word using sparse matrix slicing
                doc_indices = self.word_doc_matrix[word_idx].nonzero()[1]
                self.word_in_doc_counts[word] = len(doc_indices)

            return self.word_doc_matrix, self.word_in_doc_counts

    def create_windows(self) -> List[List[str]]:
        """
        Create sliding windows of words from each document.

        Returns:
            List of word windows
        """
        with log_step(f"Creating sliding windows with size {self.window_size}"):
            windows = []

            for doc_words in self.dataset.texts:
                words = doc_words.split()
                length = len(words)

                # Add first window
                windows.append(words[: min(self.window_size, length)])

                # Add sliding windows
                for j in range(1, max(1, length - self.window_size + 1)):
                    window = words[j : j + self.window_size]
                    windows.append(window)

            logger.info(f"Created {len(windows)} windows")
            return windows

    def process_windows(self, windows: List[List[str]]) -> Tuple[Any, Counter]:
        """
        Process word windows to get word frequencies and word pair counts.

        Args:
            windows: List of word windows

        Returns:
            Tuple containing word window counts and word pair counts
        """
        with log_step("Processing windows for word frequencies and pair counts"):
            # Convert windows to word IDs
            all_window_word_ids = []
            all_word_ids_for_counting = []
            word2id_get = self.word2id.get

            # Process windows in batches
            batch_size = min(10000, len(windows))
            num_batches = (len(windows) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(windows))

                for window in windows[start_idx:end_idx]:
                    # Convert words to IDs, filtering out words not in vocabulary
                    word_ids = [
                        word2id_get(word) for word in window if word in self.word2id
                    ]

                    if word_ids:
                        # Remove duplicates while preserving order
                        unique_word_ids = list(dict.fromkeys(word_ids))
                        all_window_word_ids.append(unique_word_ids)
                        all_word_ids_for_counting.extend(unique_word_ids)

                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    logger.info(f"Processed window batch {batch_idx+1}/{num_batches}")

            logger.info(f"Processed {len(all_window_word_ids)} valid windows")

            # Count word frequencies using numpy's bincount for efficiency
            logger.info("Counting word frequencies...")
            if all_word_ids_for_counting:
                all_word_ids_array = np.array(all_word_ids_for_counting, dtype=np.int32)
                word_window_count = np.bincount(
                    all_word_ids_array, minlength=self.vocab_size
                ).astype(np.int32)
            else:
                word_window_count = np.zeros(self.vocab_size, dtype=np.int32)

            # Count word pairs using vectorized operations
            logger.info("Counting word pairs...")
            all_i_indices = []
            all_j_indices = []

            for window_ids in all_window_word_ids:
                if len(window_ids) > 1:
                    ids_array = np.array(window_ids, dtype=np.int32)
                    n = len(ids_array)

                    # Generate upper triangle indices for all pairs
                    i_idx, j_idx = np.triu_indices(n, k=1)

                    # Store actual word IDs
                    all_i_indices.extend(ids_array[i_idx])
                    all_j_indices.extend(ids_array[j_idx])

            # Create pair tuples and count them efficiently
            pair_tuples = list(zip(all_i_indices, all_j_indices))
            word_pair_count = Counter(pair_tuples)

            logger.info(f"Found {len(word_pair_count)} unique word pairs")
            return word_window_count, word_pair_count

    def calculate_pmi_edges(
        self,
        word_window_count: Any,
        word_pair_count: Counter,
        num_window: int,
        train_size: int,
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Calculate PMI (Pointwise Mutual Information) for word-word edges.

        Args:
            word_window_count: Count of each word in all windows
            word_pair_count: Count of each word pair
            num_window: Total number of windows
            train_size: Size of the training set

        Returns:
            Tuple containing row indices, column indices, and edge weights
        """
        with log_step("Calculating PMI for word-word edges"):
            row = []
            col = []
            weight = []

            # Avoid division by zero by adding a small epsilon
            eps = 1e-10
            n_windows = max(num_window, 1)  # Avoid division by zero

            for (i, j), count in word_pair_count.items():
                word_freq_i = max(word_window_count[i], 1)  # Avoid division by zero
                word_freq_j = max(word_window_count[j], 1)  # Avoid division by zero

                # PMI formula with smoothing
                try:
                    pmi = log(
                        (count / n_windows)
                        / (
                            ((word_freq_i / n_windows) * (word_freq_j / n_windows))
                            + eps
                        )
                    )
                except (ValueError, ZeroDivisionError):
                    # Skip if calculation fails due to numerical issues
                    continue

                # Only add edges with PMI above threshold
                if pmi > self.min_pmi_threshold:
                    # Add both directions for undirected graph
                    row.extend([train_size + i, train_size + j])
                    col.extend([train_size + j, train_size + i])
                    weight.extend([pmi, pmi])

            logger.info(
                f"Created {len(weight)} PMI edges ({len(weight)//2} unique pairs)"
            )
            return row, col, weight

    def calculate_doc_word_freq(
        self, dataset_indices: List[int]
    ) -> Dict[Tuple[int, int], int]:
        """
        Calculate word frequency for each document.

        Args:
            dataset_indices: List of document indices

        Returns:
            Dictionary mapping (doc_id, word_id) to frequency
        """
        with log_step("Calculating document-word frequencies"):
            doc_word_freq = {}

            # Process documents in batches
            batch_size = min(5000, len(dataset_indices))
            num_batches = (len(dataset_indices) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(dataset_indices))

                for doc_id in dataset_indices[start_idx:end_idx]:
                    words = self.dataset.texts[doc_id].split()
                    word_counts = Counter(
                        word for word in words if word in self.word2id
                    )

                    for word, count in word_counts.items():
                        word_id = self.word2id[word]
                        doc_word_freq[(doc_id, word_id)] = count

                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    logger.info(f"Processed document batch {batch_idx+1}/{num_batches}")

            return doc_word_freq

    def calculate_tfidf_edges(
        self,
        doc_word_freq: Dict[Tuple[int, int], int],
        datasets_info: List[Tuple[List[int], int, str]],
        train_size: int,
        row: List[int],
        col: List[int],
        weight: List[float],
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Calculate TF-IDF for document-word edges.

        Args:
            doc_word_freq: Document-word frequency dictionary
            datasets_info: List of tuples containing (indices, offset, name) for each dataset split
            train_size: Size of training dataset
            row, col, weight: Existing edge information from PMI calculation

        Returns:
            Updated row, col, weight lists with TF-IDF edges added
        """
        with log_step("Calculating TF-IDF for document-word edges"):
            # Copy existing edges
            new_row = row.copy()
            new_col = col.copy()
            new_weight = weight.copy()
            tfidf_edge_count = 0

            # Process datasets

            for indices, offset, dataset_type in datasets_info:
                logger.info(f"Processing {dataset_type} documents...")

                batch_size = min(1000, len(indices))
                num_batches = (len(indices) + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(indices))

                    for batch_pos, doc_id in enumerate(indices[start_idx:end_idx]):
                        global_pos = start_idx + batch_pos
                        words = self.dataset.texts[doc_id].split()
                        total_words_in_doc = len(words) if len(words) > 0 else 1
                        unique_words = set(words) & set(self.word2id.keys())

                        for word in unique_words:
                            word_id = self.word2id[word]
                            freq = doc_word_freq.get((doc_id, word_id), 0)

                            # Normalized term frequency
                            tf = freq / total_words_in_doc

                            if freq > 0:
                                doc_node_id = offset + global_pos
                                word_node_id = train_size + word_id

                                # Calculate IDF with smoothing
                                idf = log(
                                    max(
                                        len(self.dataset)
                                        / self.word_in_doc_counts[word],
                                        1.0,
                                    )
                                )
                                tfidf = tf * idf

                                # Add document -> word edge
                                new_row.append(doc_node_id)
                                new_col.append(word_node_id)
                                new_weight.append(tfidf)
                                tfidf_edge_count += 1

                                # Add word -> document edge if bidirectional
                                if self.bidirectional_tfidf:
                                    new_row.append(word_node_id)
                                    new_col.append(doc_node_id)
                                    new_weight.append(tfidf)
                                    tfidf_edge_count += 1

                    if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                        logger.info(
                            f"Processed {dataset_type} batch {batch_idx+1}/{num_batches}"
                        )

            logger.info(f"Created {tfidf_edge_count} TF-IDF edges")
            logger.info(f"Total edges: {len(new_weight)}")
            return new_row, new_col, new_weight

    def build_graph(
        self,
        train_dataset: Subset,
        val_dataset: Subset,
        test_dataset: Subset,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
    ) -> Dict[str, Any]:
        """
        Build the complete document-word graph.

        Args:
            train_dataset: Training dataset subset
            val_dataset: Validation dataset subset
            test_dataset: Test dataset subset
            train_idx, val_idx, test_idx: Indices for each split

        Returns:
            Dictionary containing graph components
        """
        # Build vocabulary
        self.build_vocab()

        # Build word-document matrix
        self.build_word_doc_matrix()

        # Get label information
        label_list = self.dataset.LE.classes_

        # Initialize sparse matrices for each dataset split
        train_size = len(train_dataset)
        x = csr_matrix((train_size, self.embed_dim), dtype=np.float64)
        y = self.dataset.ohe_labels[train_dataset.indices]

        val_size = len(val_dataset)
        vx = csr_matrix((val_size, self.embed_dim), dtype=np.float64)
        vy = self.dataset.ohe_labels[val_dataset.indices]

        test_size = len(test_dataset)
        tx = csr_matrix((test_size, self.embed_dim), dtype=np.float64)
        ty = self.dataset.ohe_labels[test_dataset.indices]

        allx = csr_matrix(
            (train_size + self.vocab_size, self.embed_dim), dtype=np.float64
        )
        ally = np.concatenate((y, np.zeros((self.vocab_size, len(label_list)))))

        logger.info(
            f"Matrix shapes: x={x.shape}, y={y.shape}, tx={tx.shape}, ty={ty.shape}, "
            f"allx={allx.shape}, ally={ally.shape}, vx={vx.shape}, vy={vy.shape}"
        )

        # Create word windows
        windows = self.create_windows()
        num_window = len(windows)

        # Process windows to get word frequencies and pair counts
        word_window_count, word_pair_count = self.process_windows(windows)

        # Calculate PMI edges
        row, col, weight = self.calculate_pmi_edges(
            word_window_count, word_pair_count, num_window, train_size
        )

        # Calculate document-word frequencies
        all_doc_indices = (
            list(train_dataset.indices)
            + list(val_dataset.indices)
            + list(test_dataset.indices)
        )
        doc_word_freq = self.calculate_doc_word_freq(all_doc_indices)

        # Calculate TF-IDF edges
        datasets_info = [
            (train_dataset.indices, 0, "train"),
            (val_dataset.indices, train_size + self.vocab_size, "val"),
            (test_dataset.indices, train_size + self.vocab_size + val_size, "test"),
        ]

        row, col, weight = self.calculate_tfidf_edges(
            doc_word_freq, datasets_info, train_size, row, col, weight
        )

        # Create adjacency matrix
        node_size = len(self.dataset) + self.vocab_size
        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))

        # Check if matrix is symmetric
        is_symmetric = (adj != adj.T).nnz == 0
        logger.info(f"Adjacency matrix is symmetric: {is_symmetric}")

        if not is_symmetric:
            if self.bidirectional_tfidf:
                logger.warning(
                    "Adjacency matrix is not symmetric despite bidirectional TF-IDF. "
                    "This might indicate an issue in edge construction."
                )

            # Enforce symmetry by averaging with transpose
            logger.info("Enforcing symmetry in the adjacency matrix...")
            adj = 0.5 * (adj + adj.T)
            # Verify symmetry after fixing
            is_symmetric = (adj != adj.T).nnz == 0
            logger.info(f"Adjacency matrix is now symmetric: {is_symmetric}")

        # Validate graph structure before returning
        self._validate_graph_structure(adj, node_size, train_size, val_size, test_size)

        # Return graph components
        graph_data = {
            "adj": adj,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "tx": tx,
            "ty": ty,
            "allx": allx,
            "ally": ally,
            "vocab_size": self.vocab_size,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }

        return graph_data

    def _validate_graph_structure(
        self, adj, node_size, train_size, val_size, test_size
    ):
        """
        Validate the graph structure to ensure it meets requirements.

        Args:
            adj: The adjacency matrix
            node_size: Total number of nodes
            train_size, val_size, test_size: Sizes of dataset splits
        """
        with log_step("Validating graph structure"):
            # Check 1: Verify matrix dimensions match expected node count
            expected_nodes = len(self.dataset) + self.vocab_size
            assert adj.shape == (node_size, node_size), (
                f"Adjacency matrix shape {adj.shape} doesn't match expected "
                f"dimensions ({node_size}, {node_size})"
            )
            assert (
                node_size == expected_nodes
            ), f"Node size {node_size} doesn't match expected count {expected_nodes}"

            # Check 2: Verify no out-of-bound indices in the adjacency matrix
            row_indices, col_indices = adj.nonzero()
            max_idx = max(
                row_indices.max() if len(row_indices) > 0 else 0,
                col_indices.max() if len(col_indices) > 0 else 0,
            )
            assert (
                max_idx < node_size
            ), f"Found edge with index {max_idx} which is outside valid range [0, {node_size-1}]"

            # Check 3: Verify symmetry - already done in build_graph, but double check
            is_symmetric = (adj != adj.T).nnz == 0
            assert is_symmetric, "Adjacency matrix must be symmetric for GCN"

            # Check 4: Verify we have connections for each dataset split
            doc_range = range(len(self.dataset))
            word_range = range(train_size, train_size + self.vocab_size)

            # Validate train, val, test document node connections
            for split_name, split_size, offset in [
                ("train", train_size, 0),
                ("val", val_size, train_size + self.vocab_size),
                ("test", test_size, train_size + self.vocab_size + val_size),
            ]:
                if split_size > 0:  # Skip empty splits
                    split_nodes = range(offset, offset + split_size)
                    # Get subset of adjacency matrix for this split
                    connections = adj[list(split_nodes)].sum()
                    assert (
                        connections > 0
                    ), f"No connections found for {split_name} documents"

            logger.info("Graph structure validation complete - all checks passed")


def load_or_create_dataset(tokenizer, doclevel: str, clean: bool = True) -> Any:
    """
    Load dataset from disk or create a new one if it doesn't exist.

    Args:
        tokenizer: The tokenizer to use for the dataset
        doclevel: Document level to use (e.g., 'letter')
        clean: Whether to use cleaned data

    Returns:
        Dataset object
    """
    with log_step(f"Loading or creating dataset (doclevel={doclevel}, clean={clean})"):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        dataset_file = data_dir / f"medindcls_medbert_{doclevel}_clean.pkl"

        if not dataset_file.exists():
            logger.info("Creating new dataset")
            dataset = CleanClinicDataset(
                tokenizer=tokenizer, doclevel=doclevel, clean=clean
            )
            with open(dataset_file, "wb") as f:
                logger.info(f"Saving dataset under {dataset_file}")
                pickle.dump(dataset, f)
        else:
            logger.info(f"Loading existing dataset from: {dataset_file}")
            with open(dataset_file, "rb") as f:
                dataset = pickle.load(f)

        logger.info(f"Dataset loaded with {len(dataset)} samples")
        return dataset


def split_dataset(
    dataset, params, train_ratio: float = 0.7, val_ratio: float = 0.1
) -> Tuple[List[int], List[int], List[int], Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: The dataset to split
        params: BertGCNParameters object containing splitting parameters
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.1)

    Returns:
        Tuple containing indices and dataset subsets
    """
    with log_step("Splitting dataset"):
        if not params.testunklar:
            # Standard split
            idx = np.arange(len(dataset))
            random.shuffle(idx)

            train_end = int(len(idx) * train_ratio)
            val_end = int(len(idx) * (train_ratio + val_ratio))

            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            test_idx = idx[val_end:]
        else:
            # Split based on 'unklar' label
            train_candidates, test_idx = [], []
            for i, x in enumerate(dataset):
                if "unklar" in dataset.LE.classes_[x["labels"]]:
                    test_idx.append(i)
                else:
                    train_candidates.append(i)

            # Split remaining data into train/val
            random.shuffle(train_candidates)
            train_ratio_adjusted = 0.9  # 90% for training, 10% for validation
            train_split = int(len(train_candidates) * train_ratio_adjusted)

            train_idx = train_candidates[:train_split]
            val_idx = train_candidates[train_split:]

            # Verify split
            assert len(train_idx) + len(val_idx) == len(train_candidates)

        # Create dataset subsets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        # Log split information
        logger.info(
            f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/len(dataset):.1%}), "
            f"Val={len(val_dataset)} ({len(val_dataset)/len(dataset):.1%}), "
            f"Test={len(test_dataset)} ({len(test_dataset)/len(dataset):.1%})"
        )

        return train_idx, val_idx, test_idx, train_dataset, val_dataset, test_dataset


def save_graph_components(
    graph_components: Dict[str, Any], dataname: str, doclevel: str, testunklar: bool
) -> None:
    """
    Save graph components to disk.

    Args:
        graph_components: Dictionary containing graph components
        dataname: Name of the dataset
        doclevel: Document level (e.g., 'letter')
        testunklar: Whether using special 'unklar' test set
    """
    with log_step("Saving graph components to disk"):
        # Extract components
        adj = graph_components["adj"]
        x = graph_components["x"]
        y = graph_components["y"]
        vx = graph_components["vx"]
        vy = graph_components["vy"]
        tx = graph_components["tx"]
        ty = graph_components["ty"]
        allx = graph_components["allx"]
        ally = graph_components["ally"]

        # Prepare output directory
        data_path = Path("data")
        data_path.mkdir(exist_ok=True)

        # Define suffix based on testunklar flag
        suffix = f"_{doclevel}_testunklar" if testunklar else f"_{doclevel}"

        # Save all components
        components = {
            "x": x,
            "y": y,
            "tx": tx,
            "ty": ty,
            "allx": allx,
            "ally": ally,
            "adj": adj,
            "vx": vx,
            "vy": vy,
        }

        for name, component in components.items():
            filename = data_path / f"ind.{dataname}{suffix}.{name}"
            with open(filename, "wb") as f:
                pkl.dump(component, f)
            logger.info(f"Saved {filename}")


def main(
    doclevel: str = typer.Option("letter", help="Document level"),
    bertmodel: str = typer.Option("medbert", help="BERT model"),
    window_size: int = typer.Option(20, help="Window size"),
    batch_size: int = typer.Option(1000, help="Batch size"),
    bidirectional_tfidf: bool = typer.Option(True, help="Bidirectional TF-IDF"),
    min_pmi: float = typer.Option(0.0, help="Minimum PMI"),
    seed: int = typer.Option(42, help="Random seed"),
    testunklar: bool = typer.Option(False, help="Test unclear samples"),
):
    """Main function to build and save the document-word graph."""
    params = BertGCNParameters(
        doclevel=doclevel,
        bertmodel=bertmodel,
        window_size=window_size,
        batch_size=batch_size,
        bidirectional_tfidf=bidirectional_tfidf,
        min_pmi=min_pmi,
        seed=seed,
        testunklar=testunklar,
    )
    set_seed(params.seed)

    dataname = f"medindcls_{params.doclevel}"
    logger.info(f"Building graph for dataset {dataname}")
    logger.info(f"Arguments: {params}")

    # Model path selection (fallback to DEFAULT_MODEL_PATH if needed)
    try:
        from bertgcn.config import DEFAULT_MODEL_PATH, MODEL_PATHS

        model_path = MODEL_PATHS.get(params.bertmodel, DEFAULT_MODEL_PATH)
    except ImportError:
        model_path = DEFAULT_MODEL_PATH

    with log_step("Loading BERT model and tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        embed_dim = model.bert.embeddings.word_embeddings.embedding_dim
        logger.info(f"Embedding dimension: {embed_dim}")
        del model  # Free memory

    dataset = load_or_create_dataset(tokenizer, params.doclevel)
    train_idx, val_idx, test_idx, train_dataset, val_dataset, test_dataset = (
        split_dataset(dataset, params)
    )

    graph_builder = GraphBuilder(
        dataset=dataset,
        embed_dim=embed_dim,
        dataname=dataname,
        window_size=params.window_size,
        batch_size=params.batch_size,
        bidirectional_tfidf=params.bidirectional_tfidf,
        min_pmi_threshold=params.min_pmi,
    )

    graph_components = graph_builder.build_graph(
        train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx
    )

    save_graph_components(
        graph_components, dataname, params.doclevel, params.testunklar
    )

    logger.info("Graph building complete!")


if __name__ == "__main__":
    typer.run(main)
