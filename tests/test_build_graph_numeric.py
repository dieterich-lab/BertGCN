from math import log
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.sparse import csr_matrix

from bertgcn.build_graph import GraphBuilder


class DummyDataset:
    def __init__(self, texts, ohe_labels=None):
        self.texts = texts
        self.LE = type("LE", (), {"classes_": []})()
        if ohe_labels is None:
            import numpy as _np

            self.ohe_labels = _np.zeros((len(texts), 0))
        else:
            self.ohe_labels = ohe_labels

    def __len__(self):
        return len(self.texts)


def test_pmi_numeric_small():
    # Controlled windows: two words co-occur twice out of 4 windows
    texts = [
        "a b c",
        "a b",
        "b c",
        "d e",
    ]
    ds = DummyDataset(texts)
    gb = GraphBuilder(ds, embed_dim=10, dataname="numtest")

    # Build vocab from texts
    vocab, w2id = gb.build_vocab()

    # Manually create windows similar to previous tests
    windows = [["a", "b"], ["a", "b"], ["b", "c"], ["d", "e"]]

    # Ensure vocab contains our words
    for w in ["a", "b", "c", "d", "e"]:
        assert w in w2id

    gb.vocab_size = len(gb.vocab)

    word_window_count, word_pair_count = gb.process_windows(windows)

    # Check counts
    a_id = gb.word2id["a"]
    b_id = gb.word2id["b"]
    c_id = gb.word2id["c"]

    assert word_window_count[a_id] == 2
    assert word_window_count[b_id] == 3
    assert word_window_count[c_id] == 1

    # Now compute PMI edges
    row, col, weight = gb.calculate_pmi_edges(
        word_window_count, word_pair_count, len(windows), train_size=3
    )

    # compute expected PMI for pair (a,b)
    count_ab = word_pair_count.get((a_id, b_id), word_pair_count.get((b_id, a_id), 0))
    N = len(windows)
    pmi_ab = log(
        (count_ab / N)
        / ((word_window_count[a_id] / N) * (word_window_count[b_id] / N) + 1e-10)
    )

    # find edge weight corresponding to a<->b (two directed edges should have same weight)
    found = False
    for w in weight:
        if abs(w - pmi_ab) < 1e-6:
            found = True
            break
    assert found, "Expected PMI weight for (a,b) not found"


def test_tfidf_numeric_small():
    # Create tiny dataset where TF-IDF is predictable
    texts = ["apple banana apple", "banana apple", "cherry"]
    ds = DummyDataset(texts)
    gb = GraphBuilder(ds, embed_dim=10, dataname="numtest")
    vocab, w2id = gb.build_vocab()
    gb.vocab_size = len(gb.vocab)

    # manual indices: train=first two, val=third, no test
    train_idx = [0, 1]
    val_idx = [2]
    test_idx = []

    from torch.utils.data import Subset

    train_dataset = Subset(ds, train_idx)
    val_dataset = Subset(ds, val_idx)
    test_dataset = Subset(ds, test_idx)

    # Build word-doc matrix and windows
    gb.build_word_doc_matrix()
    windows = gb.create_windows()
    word_window_count, word_pair_count = gb.process_windows(windows)

    # compute doc_word_freq across all indices
    all_indices = train_idx + val_idx
    doc_id_map = {doc_id: new_id for new_id, doc_id in enumerate(all_indices)}
    doc_word_freq = gb.calculate_doc_word_freq(all_indices, doc_id_map)

    # compute tfidf edges
    datasets_info = [
        (train_dataset.indices, "train"),
        (val_dataset.indices, "val"),
    ]
    row, col, weight = gb.calculate_tfidf_edges(
        doc_word_freq,
        datasets_info,
        total_docs=len(all_indices),
        row=[],
        col=[],
        weight=[],
        doc_id_map=doc_id_map,
    )

    # We'll check that TF-IDF for 'apple' in doc 0 > 0 and matches manual computation
    apple_id = gb.word2id.get("apple")
    assert apple_id is not None

    # find edges where doc 0 -> word apple
    doc0_node = 0
    word_node = len(train_idx) + apple_id
    tfidf_vals = [
        w for r, c, w in zip(row, col, weight) if r == doc0_node and c == word_node
    ]
    assert len(tfidf_vals) > 0

    # manual tfidf: tf = freq / total_words_in_doc; idf = log(N / df)
    df = gb.word_in_doc_counts.get("apple", 1)
    N = len(all_indices)
    # find freq of apple in doc0
    freq = doc_word_freq.get((0, apple_id), 0)
    tf = freq / (len(ds.texts[0].split()) if len(ds.texts[0].split()) > 0 else 1)
    idf = log(max(len(ds) / df, 1.0))
    expected = tf * idf

    assert any(
        abs(v - expected) < 1e-6 for v in tfidf_vals
    ), f"TF-IDF mismatch: expected {expected} got {tfidf_vals}"
