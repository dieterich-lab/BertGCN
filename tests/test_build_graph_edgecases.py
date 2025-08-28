import pytest

from bertgcn.build_graph import GraphBuilder


class DummyDataset:
    def __init__(self, texts):
        self.texts = texts
        self.LE = type("LE", (), {"classes_": []})()
        import numpy as np

        self.ohe_labels = np.zeros((len(texts), 0))

    def __len__(self):
        return len(self.texts)


def test_build_vocab_with_empty_and_nonstr():
    ds = DummyDataset(["", None, "   ", 123, "Word A", "word a"])
    gb = GraphBuilder(ds, embed_dim=10, dataname="edgecase", min_token_freq=1)
    vocab, w2id = gb.build_vocab()

    # tokens should be normalized (lowercased) and numeric/non-str ignored
    assert "word" in w2id
    assert "a" in w2id or "word a" in w2id


def test_build_vocab_min_freq():
    ds = DummyDataset(["apple banana", "apple", "banana", "cherry"])
    gb = GraphBuilder(ds, embed_dim=10, dataname="edgecase", min_token_freq=2)
    vocab, w2id = gb.build_vocab()

    # only 'apple' and 'banana' meet min_freq=2
    assert "apple" in w2id
    assert "banana" in w2id
    assert "cherry" not in w2id


def test_create_windows_empty_and_small():
    ds = DummyDataset(["", "singleword", "two words here"])
    gb = GraphBuilder(ds, embed_dim=10, dataname="edgecase", window_size=3)
    windows = gb.create_windows()

    # Ensure windows created for non-empty docs
    assert any(len(w) > 0 for w in windows)
    # Ensure no windows for empty/non-str entries
    assert all(isinstance(w, list) for w in windows)
