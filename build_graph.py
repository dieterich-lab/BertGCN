import logging
import os
import pickle
import pickle as pkl
import random
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Subset
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from entry import *
from params import parse_args
from utils import *

args = parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(
    format=f"%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ],
)

DATANAME = f"medindcls_{args.doclevel}" if args.data == "MIC" else "csc"

tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINEDMODEL)
embed_dim = model.bert.embeddings.word_embeddings.embedding_dim
del model

dataset_file = Path("data") / f"medindcls_medbert_{args.doclevel}_clean.pkl"
if not dataset_file.exists():
    logging.info("Creating dataset")
    dataset = CleanClinicDataset(
        tokenizer=tokenizer, doclevel=args.doclevel, clean=True
    )
    with open(dataset_file, "wb") as f:
        logging.info(f"Saving dataset under {dataset_file}")
        pickle.dump(dataset, f)
else:
    logging.info(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

if args.data == "MIC":
    if not args.testunklar:
        if not args.cv:
            idx = np.arange(len(dataset))
            random.shuffle(idx)
            train_idx, val_idx, test_idx = (
                idx[: int(len(idx) * 0.7)],
                idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
                idx[int(len(idx) * 0.8) :],
            )
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            test_dataset = Subset(dataset, test_idx)
        else:
            train_datasets, val_datasets, test_datasets = list(), list(), list()
            for i in range(10):
                idx = np.arange(len(dataset))
                random.shuffle(idx)
                train_idx, val_idx, test_idx = (
                    idx[: int(len(idx) * 0.7)],
                    idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
                    idx[int(len(idx) * 0.8) :],
                )
                train_datasets.append(Subset(dataset, train_idx))
                val_datasets.append(Subset(dataset, val_idx))
                test_datasets.append(Subset(dataset, test_idx))
    else:
        _train_idx, test_idx = list(), list()
        for i, x in enumerate(dataset):
            if "unklar" in dataset.LE.classes_[x["labels"]]:
                test_idx.append(i)
            else:
                _train_idx.append(i)
        test_dataset = Subset(dataset, test_idx)
        random.shuffle(_train_idx)
        train_idx, val_idx = (
            _train_idx[: int(len(_train_idx) * 0.9)],
            _train_idx[int(len(_train_idx) * 0.9) :],
        )
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        assert len(train_idx) + len(val_idx) == len(_train_idx)
elif args.data == "CSC":
    idx = np.arange(len(train_dataset))
    random.shuffle(idx)
    train_idx, val_idx = idx[: int(len(idx) * 0.9)], idx[int(len(idx) * 0.9) :]
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)


if not args.cv:
    train_datasets = [train_dataset]
    val_datasets = [val_dataset]
    test_datasets = [test_dataset]

for r in range(10):
    train_dataset = train_datasets[r]
    val_dataset = val_datasets[r]
    test_dataset = test_datasets[r]
    logging.info(
        f"Len datasets: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}"
    )

    # build vocab
    logging.info("Build vocab")
    if args.data == "MIC":
        word_counter = Counter(
            [word for sents in dataset.texts for word in sents.split()]
        )
    elif args.data == "CSC":
        word_counter = Counter(
            [word for sents in dataset.texts for word in sents.split()]
            + [word for sents in test_dataset.texts for word in sents.split()]
        )

    vocab = list(word_counter.keys())
    print("\n".join(vocab), file=open(Path("data") / f"{DATANAME}_vocab.txt", "w"))
    vocab_size = len(vocab)

    if args.data == "MIC":
        words_per_text_lists = [words.split() for words in dataset.texts]
    elif args.data == "CSC":
        words_per_text_lists = [words.split() for words in dataset.texts] + [
            words.split() for words in test_dataset.texts
        ]

    words_per_text_sets = [set(x) for x in words_per_text_lists]

    logging.info("Getting word doc list")
    word_in_docs_dict = defaultdict(set)
    for word in vocab:
        for i, doc in enumerate(words_per_text_sets):
            if word in doc:
                word_in_docs_dict[word].add(i)

    word_in_doc_counts = {w: len(l) for w, l in word_in_docs_dict.items()}

    word2id = dict(reversed(x) for x in enumerate(vocab))

    label_list = dataset.LE.classes_

    train_size = len(train_dataset)
    x = csr_matrix((train_size, embed_dim), dtype=np.float)
    y = dataset.ohe_labels[train_dataset.indices]

    val_size = len(val_dataset)
    vx = csr_matrix((val_size, embed_dim), dtype=np.float)
    vy = dataset.ohe_labels[val_dataset.indices]

    test_size = len(test_dataset)
    tx = csr_matrix((test_size, embed_dim), dtype=np.float)
    if args.data == "MIC":
        ty = dataset.ohe_labels[test_dataset.indices]
    elif args.data == "CSC":
        ty = test_dataset.ohe_labels
    elif args.data == "Patho":
        pass

    allx = csr_matrix((train_size + vocab_size, embed_dim), dtype=np.float)
    ally = np.concatenate((y.toarray(), np.zeros((vocab_size, len(label_list)))))

    logging.info(
        (
            x.shape,
            y.shape,
            tx.shape,
            ty.shape,
            allx.shape,
            ally.shape,
            vx.shape,
            vy.shape,
        )
    )
    logging.info(
        (
            x.shape,
            y.shape,
            tx.shape,
            ty.shape,
            allx.shape,
            ally.shape,
            vx.shape,
            vy.shape,
        )
    )

    window_size = 20
    windows = []

    logging.info("Getting windows")
    for doc_words in dataset.texts:
        words = doc_words.split()
        length = len(words)
        windows.append(words[:window_size])
        for j in range(1, length - window_size + 1):
            window = words[j : j + window_size]
            windows.append(window)

    logging.info("Getting word window freq")
    word_window_count_dict = defaultdict(int)
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            word_window_count_dict[window[i]] += 1
            appeared.add(window[i])

    logging.info("Getting word pair counts")
    word_pair_count_dict = defaultdict(int)
    for window in windows:
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_i_id = word2id[word_i]
                word_j = window[j]
                word_j_id = word2id[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + "," + str(word_j_id)
                word_pair_count_dict[word_pair_str] += 1
                word_pair_str = str(word_j_id) + "," + str(word_i_id)
                word_pair_count_dict[word_pair_str] += 1

    row = []
    col = []
    weight = []

    # pmi as weights
    num_window = len(windows)

    logging.info("Calculating PMI")
    for key, count in word_pair_count_dict.items():
        temp = key.split(",")
        i = int(temp[0])
        j = int(temp[1])
        word_freq_i = word_window_count_dict[vocab[i]]
        word_freq_j = word_window_count_dict[vocab[j]]
        pmi = log(
            (1.0 * count / num_window)
            / (word_freq_i * word_freq_j / (num_window * num_window))
        )
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # doc word frequency
    doc_word_freq = defaultdict(int)

    # term frequency
    logging.info("Calculating tf-idf")
    for doc_id in train_dataset.indices:
        doc_words = dataset.texts[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word2id[word]
            doc_word_str = str(doc_id) + "," + str(word_id)
            doc_word_freq[doc_word_str] += 1

    for doc_id in val_dataset.indices:
        doc_words = dataset.texts[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word2id[word]
            doc_word_str = str(doc_id) + "," + str(word_id)
            doc_word_freq[doc_word_str] += 1

    if args.data == "MIC":
        for doc_id in test_dataset.indices:
            doc_words = dataset.texts[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = word2id[word]
                doc_word_str = str(doc_id) + "," + str(word_id)
                doc_word_freq[doc_word_str] += 1
    elif args.data == "CSC":
        for c, doc_words in enumerate(test_dataset.texts):
            words = doc_words.split()
            for word in words:
                word_id = word2id[word]
                doc_word_str = str(c + len(dataset)) + "," + str(word_id)
                doc_word_freq[doc_word_str] += 1
    elif args.data == "Patho":
        pass

    for c, i in enumerate(train_dataset.indices):
        doc_words = dataset.texts[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word2id[word]
            key = str(i) + "," + str(j)
            freq = doc_word_freq[key]
            row.append(c)
            col.append(train_size + j)
            idf = log(1.0 * len(dataset) / word_in_doc_counts[word])
            weight.append(freq * idf)
            doc_word_set.add(word)

    for c, i in enumerate(val_dataset.indices):
        doc_words = dataset.texts[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word2id[word]
            key = str(i) + "," + str(j)
            freq = doc_word_freq[key]
            row.append(train_size + vocab_size + c)
            col.append(train_size + j)
            idf = log(1.0 * len(dataset) / word_in_doc_counts[word])
            weight.append(freq * idf)
            doc_word_set.add(word)

    if args.data == "MIC":
        for c, i in enumerate(test_dataset.indices):
            doc_words = dataset.texts[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word2id[word]
                key = str(i) + "," + str(j)
                freq = doc_word_freq[key]
                row.append(train_size + vocab_size + val_size + c)
                col.append(train_size + j)
                idf = log(1.0 * len(dataset) / word_in_doc_counts[vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)
    elif args.data == "CSC":
        for c, doc_words in enumerate(test_dataset.texts):
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word2id[word]
                key = str(c + len(dataset)) + "," + str(j)
                freq = doc_word_freq[key]
                row.append(train_size + vocab_size + val_size + c)
                col.append(train_size + j)
                idf = log(1.0 * len(dataset) / word_in_doc_counts[vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)
    elif args.data == "Patho":
        pass

    if args.data == "MIC":
        node_size = len(dataset) + vocab_size
    elif args.data == "CSC":
        node_size = len(dataset) + vocab_size + len(test_dataset)
    elif args.data == "Patho":
        pass

    adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # dump objects
    logging.info("Dumping objects")
    if not args.cv:
        data_path = "data"
    else:
        data_path = f"data/{r}"

    os.makedirs(data_path, exist_ok=True)

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.x", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.x", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.x", "wb")
    pkl.dump(x, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.y", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.y", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.y", "wb")
    pkl.dump(y, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.tx", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.tx", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.tx", "wb")
    pkl.dump(tx, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.ty", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.ty", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.ty", "wb")
    pkl.dump(ty, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.allx", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.allx", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.allx", "wb")
    pkl.dump(allx, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.ally", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.ally", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.ally", "wb")
    pkl.dump(ally, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.adj", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.adj", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.adj", "wb")
    pkl.dump(adj, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.vx", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.vx", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.vx", "wb")
    pkl.dump(vx, f)
    f.close()

    if args.data == "MIC":
        if args.testunklar:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}_testunklar.vy", "wb")
        else:
            f = open(f"{data_path}/ind.{dataset}_{args.doclevel}.vy", "wb")
    else:
        f = open(f"{data_path}/ind.{dataset}.vy", "wb")
    pkl.dump(vy, f)
    f.close()
