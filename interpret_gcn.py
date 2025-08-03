import datetime
import logging
import pickle
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import dgl
import numpy as np
import scipy as sp
import shap
import torch
import torch.utils.data as Data
from captum.attr import IntegratedGradients
from torch.utils.data import Subset
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from entry import *
from model import BertGCN
from params import parse_args
from utils import *

logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    format=f"%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ],
)

args = parse_args()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

BATCHSIZE = 8

MODELNAME = Path(PRETRAINEDMODEL).stem
if args.data == "MIC":
    DATASET = "med_indication_all_RF_diag"
    DATASETPATH = Path("data") / f"ind.{DATASET}_{args.doclevel}"
    if args.testunklar:
        DATASETPATH = Path("data") / f"ind.{DATASET}_{args.doclevel}_testunklar"
    MAXEVALS = 5399
elif args.data == "CSC":
    DATASET = "CARDIODE400_main"
    DATASETPATH = Path("data") / f"ind.{DATASET}"
    MAXEVALS = 233743

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)

if args.data == "MIC":
    dataset_file = Path("data") / f"medindcls_{args.bertmodel}_{args.doclevel}.json"
    if not dataset_file.exists():
        print("Creating dataset")
        dataset = CleanClinicDataset(
            tokenizer=tokenizer, task="MIC", doclevel=args.doclevel, clean=False
        )
        with open(dataset_file, "wb") as f:
            print(f"Saving dataset under {dataset_file}")
            pickle.dump(dataset, f)
    else:
        print(f"Loading dataset from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
elif args.data == "CSC":
    train_dataset_file = Path("data") / "csc_train_bert.json"
    test_dataset_file = Path("data") / "csc_test_bert.json"

    if not train_dataset_file.exists():
        print("Creating train dataset")
        train_dataset = CleanClinicDataset(
            tokenizer=tokenizer, task="CSC", clean=False, mode="train"
        )
        with open(train_dataset_file, "wb") as f:
            print(f"Saving dataset under {train_dataset_file}")
            pickle.dump(train_dataset, f)
    else:
        print(f"Loading train dataset from: {train_dataset_file}")
        with open(train_dataset_file, "rb") as f:
            train_dataset = pickle.load(f)
    if not test_dataset_file.exists():
        print("Creating test dataset")
        test_dataset = CleanClinicDataset(
            tokenizer=tokenizer, task="CSC", clean=False, mode="test"
        )
        with open(test_dataset_file, "wb") as f:
            print(f"Saving dataset under {test_dataset_file}")
            pickle.dump(test_dataset, f)
    else:
        print(f"Loading test dataset from: {test_dataset_file}")
        with open(test_dataset_file, "rb") as f:
            test_dataset = pickle.load(f)
    dataset = train_dataset

GCNNAME = f"{MODELNAME}_{dataset}.pt"
SAVEDIR = Path(f"models/gcn/{args.mixfactor}/{args.doclevel}")
GCNPATH = SAVEDIR / GCNNAME

IGFILE = SAVEDIR / f"ig_attrs_gcn_{MODELNAME}_{args.data}"
SHAPFILE = SAVEDIR / f"shap_values_gcn_{MODELNAME}_{args.data}"
if args.testunklar:
    IGFILE = SAVEDIR / f"ig_attrs_gcn_{MODELNAME}_{args.data}_testunklar"
    SHAPFILE = SAVEDIR / f"shap_values_gcn_{MODELNAME}_{args.data}_testunklar"

if args.data == "MIC":
    if not args.testunklar:
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

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, _, _ = (
    load_corpus(DATASETPATH)
)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

model = BertGCN(
    nb_class=nb_class,
    pretrained_model=PRETRAINEDMODEL,
    mix_factor=args.mixfactor,
    gcn_layers=2,
    n_hidden=200,
    dropout=0.5,
)

model = model.to(device)
model = model.eval()
model = model.to(device)
model = model.eval()

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

if args.data == "MIC":
    input_ids = torch.cat(
        [
            torch.tensor(
                np.array(
                    [x["input_ids"] for x in np.array(dataset.examples)[train_idx]]
                )
            ),
            torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
            torch.tensor(
                np.array([x["input_ids"] for x in np.array(dataset.examples)[val_idx]])
            ),
            torch.tensor(
                np.array([x["input_ids"] for x in np.array(dataset.examples)[test_idx]])
            ),
        ]
    )
elif args.data == "CSC":
    input_ids = torch.cat(
        [
            torch.tensor(
                np.array(
                    [x["input_ids"] for x in np.array(dataset.examples)[train_idx]]
                )
            ),
            torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
            torch.tensor(
                np.array([x["input_ids"] for x in np.array(dataset.examples)[val_idx]])
            ),
            torch.tensor(np.array([x["input_ids"] for x in test_dataset.examples])),
        ]
    )

assert np.array_equal(y[:nb_train], dataset.labels[train_idx])
assert np.array_equal(
    y[nb_train + nb_word : nb_train + nb_word + nb_val], dataset.labels[val_idx]
)
if args.data == "MIC":
    assert np.array_equal(y[-nb_test:], dataset.labels[test_idx])
elif args.data == "CSC":
    assert np.array_equal(y[-nb_test:], test_dataset.labels)

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

train_idx_dataset = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
val_idx_dataset = Data.TensorDataset(
    torch.arange(nb_train + nb_word, nb_train + nb_word + nb_val, dtype=torch.long)
)
test_idx_dataset = Data.TensorDataset(
    torch.arange(nb_node - nb_test, nb_node, dtype=torch.long)
)

idx_loader_train = Data.DataLoader(train_idx_dataset, batch_size=BATCHSIZE)
idx_loader_val = Data.DataLoader(val_idx_dataset, batch_size=BATCHSIZE)
idx_loader_test = Data.DataLoader(test_idx_dataset, batch_size=BATCHSIZE)

graph = dgl.from_scipy(adj_norm.astype("float32"), eweight_name="edge_weight")
graph.ndata["input_ids"] = input_ids
graph.ndata["label"], graph.ndata["train"], graph.ndata["val"], graph.ndata["test"] = (
    torch.LongTensor(y),
    torch.FloatTensor(train_mask),
    torch.FloatTensor(val_mask),
    torch.FloatTensor(test_mask),
)
graph.ndata["label_train"] = torch.LongTensor(y_train)
graph.ndata["cls_feats"] = torch.zeros((nb_node, model.feat_dim))


def update_feature():
    global graph, model
    dataloader = Data.DataLoader(
        Data.TensorDataset(graph.ndata["input_ids"][doc_mask]), batch_size=64
    )
    with torch.no_grad():
        model.eval()
        cls_list = []
        logging.info("Udating features...")
        for batch in dataloader:
            input_ids = [x.to(device) for x in batch][0]
            output = model.bert_model(input_ids=input_ids)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = torch.cat(cls_list, axis=0)
    graph = graph.to("cpu")
    graph.ndata["cls_feats"][doc_mask] = cls_feat


logging.info(
    f"Loading best gcn model from {GCNPATH} saved on {datetime.datetime.fromtimestamp(GCNPATH.stat().st_ctime)}"
)
model.load_state_dict(torch.load(GCNPATH, map_location="cpu"))

if not args.suppressupdates:
    update_feature()


def zero_masker(node_mask, node_ids):
    return node_mask.reshape(1, len(node_mask))


def shap_forward(node_mask, graph, target_id2, doc_feats):
    with torch.no_grad():
        graph = graph.to(device)
        doc_feats = [doc_feats.detach().cpu().numpy() * m[:, None] for m in node_mask]
        doc_feats = [torch.tensor(x) for x in doc_feats]
        doc_feats = torch.cat(doc_feats).to(device)
        outputs = model.explain_forward(
            doc_feats, graph, target_id2, doc_mask, args.interpret_mode
        )
        return torch.softmax(outputs, dim=-1).detach().cpu().numpy()


def ig_forward(doc_feats, graph, target_id2):
    graph = graph.to(device)
    return model.explain_forward(
        doc_feats, graph, target_id2, doc_mask, args.interpret_mode
    )


doc_feats = graph.ndata["cls_feats"][doc_mask].requires_grad_().to(device)
node_ids = np.array([np.arange(doc_feats.size(0))], dtype=str)


ig_attr_dict = defaultdict(lambda: defaultdict(list))
ig_attr_list = list()
shap_value_list = list()

logging.info(f"Test data size: {test_mask.sum()}")
# for target_id1 in range(test_mask.sum())[:1]:
for target_id1 in range(test_mask.sum()):
    target_id2 = test_mask.nonzero()[0][target_id1]
    logging.info((target_id1, target_id2))

    target_label = graph.ndata["label"][target_id2].item()
    target_cls = dataset.LE.classes_[target_label]

    logging.info("Computing IG attributions ...")
    ig_explainer = IntegratedGradients(
        partial(ig_forward, graph=graph, target_id2=target_id2)
    )
    ig_attr, delta = ig_explainer.attribute(
        doc_feats,
        target=target_label,
        internal_batch_size=graph.num_nodes(),
        return_convergence_delta=True,
    )
    ig_attr = ig_attr.sum(dim=-1)
    ig_attr = ig_attr / torch.norm(ig_attr)
    ig_attr = ig_attr.cpu().detach().numpy()
    ig_attr_list.append(ig_attr)

    if args.data != "CSC":
        logging.info("Computing SHAP values ...")
        shap_explainer = shap.explainers.Permutation(
            partial(
                shap_forward, graph=graph, target_id2=target_id2, doc_feats=doc_feats
            ),
            zero_masker,
            max_evals=MAXEVALS,
        )
        shap_values = shap_explainer(node_ids, silent=True)
        shap_value_list.append(shap_values[0, :, target_label].values)

logging.info("Saving IG values ...")
np.savez_compressed(IGFILE, np.array(ig_attr_list))

if args.data != "CSC":
    logging.info("Saving SHAP values ...")
    np.savez_compressed(SHAPFILE, np.array(shap_value_list))
