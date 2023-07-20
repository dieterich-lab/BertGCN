import datetime
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from utils import *
from model import BertGCN
import torch.utils.data as Data
import dgl

import logging

import shap

import dgl
import torch
from captum.attr import IntegratedGradients
from functools import partial
from collections import defaultdict
from params import parse_args

logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

args = parse_args()

MODELTYPE = "deepset/gbert-base"
BATCHSIZE = 8
DATASET = "med_indication_all_RF_diag"
DATASETPATH =  Path("data") / f"ind.{DATASET}"
DATASETFILE = Path("data") / "medindcls_bert.json"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODELTYPE)

if not DATASETFILE.exists():
    print("Creating dataset")
    dataset = CleanClinicDataset(tokenizer=tokenizer, clean=False)
    with open(DATASETFILE, "wb") as f:
        print(f"Saving dataset under {DATASETFILE}")
        pickle.dump(dataset, f)
else:
    print(f"Loading dataset from: {DATASETFILE}")
    with open(DATASETFILE, "rb") as f:
        dataset = pickle.load(f)

GCNNAME = f"{Path(MODELTYPE).stem}_{dataset}.pt"
SAVEPATH = Path(f"models/gcn/{args.mixfactor}")
GCNPATH= SAVEPATH / GCNNAME
if args.interpret_mode == "gcn_only":
    IGFILE = SAVEPATH / "_ig_attrs_gcn_only.json"
    SHAPFILE = SAVEPATH / "_shap_values_gcn_only.json"
elif args.interpret_mode == "gcn_bert":
    IGFILE = SAVEPATH / "_ig_attrs_gcn_bert.json"
    SHAPFILE = SAVEPATH / "_shap_values_gcn_bert.json"

idx = np.arange(len(dataset))
random.shuffle(idx)

train_idx, val_idx, test_idx = (
    idx[: int(len(idx) * 0.7)],
    idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
    idx[int(len(idx) * 0.8) :],
)

test_dataset = Subset(dataset, test_idx)

# train_idx, val_idx, test_idx = (
#     idx[: int(len(idx) * 0.7)],
#     idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
#     idx[int(len(idx) * 0.8) :],
# )


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, _, _ = load_corpus(DATASETPATH)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# if gcn_model == "gcn":
gcn = BertGCN(
    nb_class=nb_class,
    pretrained_model="deepset/gbert-base",
    mix_factor=args.mixfactor,
    gcn_layers=2,
    n_hidden=200,
    dropout=0.5,
)

gcn = gcn.to(device)
gcn = gcn.eval()

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

tokenizer = AutoTokenizer.from_pretrained(MODELTYPE)


input_ids = torch.cat(
    [
        torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[train_idx]])),
        torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
        torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[val_idx]])),
        torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[test_idx]])),
    ]
)

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

train_idx_dataset = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
val_idx_dataset = Data.TensorDataset(torch.arange(nb_train + nb_word, nb_train + nb_word + nb_val, dtype=torch.long))
test_idx_dataset = Data.TensorDataset(torch.arange(nb_node - nb_test, nb_node, dtype=torch.long))

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
graph.ndata["cls_feats"] = torch.zeros((nb_node, gcn.feat_dim))


def update_feature():
    global graph, gcn
    dataloader = Data.DataLoader(Data.TensorDataset(graph.ndata["input_ids"][doc_mask]), batch_size=64)
    with torch.no_grad():
        # gcn = gcn.to(device)
        gcn.eval()
        cls_list = []
        print("Udating features...")
        for batch in dataloader:
            input_ids = [x.to(device) for x in batch][0]
            output = gcn.bert_model(input_ids=input_ids)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = torch.cat(cls_list, axis=0)
    graph = graph.to("cpu")
    graph.ndata["cls_feats"][doc_mask] = cls_feat


print(f"Loading best gcn model from {GCNPATH} saved on {datetime.datetime.fromtimestamp(GCNPATH.stat().st_ctime)}")
gcn.load_state_dict(torch.load(GCNPATH, map_location="cpu"))

if not args.debug:
    update_feature()


def zero_masker(node_mask, node_ids):
    return node_mask.reshape(1, len(node_mask))


def shap_forward(node_mask, graph, target_id2, doc_feats):
    with torch.no_grad():
        graph = graph.to(device)
        doc_feats = [doc_feats.detach().cpu().numpy() * m[:, None] for m in node_mask]
        doc_feats = [torch.tensor(x) for x in doc_feats]
        doc_feats = torch.cat(doc_feats).to(device)
        outputs = gcn.explain_forward(doc_feats, graph, target_id2, doc_mask, args.interpret_mode)
        return torch.softmax(outputs, dim=-1).detach().cpu().numpy()


def ig_forward(doc_feats, graph, target_id2):
    graph = graph.to(device)
    return gcn.explain_forward(doc_feats, graph, target_id2, doc_mask, args.interpret_mode)


doc_feats = graph.ndata["cls_feats"][doc_mask].requires_grad_().to(device)
node_ids = np.array([np.arange(doc_feats.size(0))], dtype=str)


ig_attr_dict = defaultdict(lambda: defaultdict(list))
ig_attr_list = list()
shap_value_dict = defaultdict(lambda: defaultdict(list))
shap_value_list = list()

for target_id1 in range(test_mask.sum()):
    target_id2 = test_mask.nonzero()[0][target_id1]
    print(target_id1, target_id2, test_idx[target_id1])

    target_label = graph.ndata["label"][target_id2].item()
    target_cls = dataset.LE.classes_[target_label]

    ig_explainer = IntegratedGradients(partial(ig_forward, graph=graph, target_id2=target_id2))
    shap_explainer = shap.explainers.Permutation(
        partial(shap_forward, graph=graph, target_id2=target_id2, doc_feats=doc_feats), zero_masker, max_evals=5399
    )

    ig_attr, delta = ig_explainer.attribute(
        doc_feats, target=target_label, internal_batch_size=graph.num_nodes(), return_convergence_delta=True
    )
    ig_attr = ig_attr.sum(dim=-1)
    ig_attr = ig_attr / torch.norm(ig_attr)
    ig_attr = ig_attr.cpu().detach().numpy()

    shap_values = shap_explainer(node_ids)

    # ig_attr_dict[str(target_id2)]["label"] = dataset.LE.classes_[target_label]
    # shap_value_dict[str(target_id2)]["label"] = dataset.LE.classes_[target_label]

    ig_attr_list.append(ig_attr)
    shap_value_list.append(shap_values[0, :, target_label].values)

    # for id1 in np.argpartition(ig_attr, -10)[-10:][::-1]:
    #     id2 = doc_mask.nonzero()[0][id1]
    #     label = graph.ndata["label"][id2].item()
    #     tgt_cls = dataset.LE.classes_[label]
    #     ig_attr_dict[str(target_id2)]["rel_doc_ids"].append(int(id2))
    #     ig_attr_dict[str(target_id2)]["rel_doc_labels"].append(tgt_cls)

    # for id1 in np.argpartition(shap_values[0, :, target_label].values, -10)[-10:][::-1]:
    #     id2 = doc_mask.nonzero()[0][id1]
    #     label = graph.ndata["label"][id2].item()
    #     tgt_cls = dataset.LE.classes_[label]
    #     shap_value_dict[str(target_id2)]["rel_doc_ids"].append(int(id2))
    #     shap_value_dict[str(target_id2)]["rel_doc_labels"].append(tgt_cls)

# with open(IGFILE, "w") as f:
#     json.dump(ig_attr_dict, f, indent=2)

# with open(SHAPFILE, "w") as f:
#     json.dump(shap_value_dict, f, indent=2)

with open(IGFILE, "wb") as f:
    pickle.dump(ig_attr_list, f)

with open(SHAPFILE, "wb") as f:
    pickle.dump(shap_value_list, f)
