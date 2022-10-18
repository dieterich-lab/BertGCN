import datetime
from metrics import SklearnClassificationReport
import logging
import pickle
import random
from pathlib import Path

import dgl
import torch
import torch.utils.data as Data
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.handlers.param_scheduler import LRScheduler, create_lr_scheduler_with_warmup
from ignite.metrics import Accuracy, ClassificationReport, Loss
from ignite.utils import setup_logger
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from metrics import SklearnClassificationReport
from model import BertGAT, BertGCN
from params import parse_args
from utils import *

args = parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

MODELPATH = "deepset/gbert-base"
BERTLR = 5e-5
GCNLR = 3e-5
LR = 4e-5
BATCHSIZE = 8
NEPOCHS = 50
ACCUSTEPS = 8
LOGINTERVALL = 100
DATASET = "med_indication_all_RF_diag"
PRETRAINDCKPT = Path("models/finetuned/gbert-base_med_indication_all_RF_diag_best.pt")

SAVEPATH = f"{Path(MODELPATH).stem}"

tokenizer = AutoTokenizer.from_pretrained(MODELPATH)

logging.basicConfig(
    format=f"%(asctime)s ({args.mixfactor}) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(DATASET)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# if gcn_model == "gcn":
model = BertGCN(
    nb_class=nb_class,
    pretrained_model="deepset/gbert-base",
    mix_factor=args.mixfactor,
    gcn_layers=2,
    n_hidden=200,
    dropout=0.5,
)
# else:
#     model = BertGAT(
#         nb_class=nb_class,
#         pretrained_model="deepset/gbert-base",
#         m=M,
#         gcn_layers=gcn_layers,
#         heads=heads,
#         n_hidden=200,
#         dropout=0.5,
#     )


logging.info(
    f"Loading pretrained bert model from {PRETRAINDCKPT} saved on {datetime.datetime.fromtimestamp(PRETRAINDCKPT.stat().st_ctime)}"
)
ckpt = torch.load(PRETRAINDCKPT, map_location=device)
model.bert_model.load_state_dict(ckpt["bert_model"])
model.classifier.load_state_dict(ckpt["classifier"])


# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

tokenizer = AutoTokenizer.from_pretrained(MODELPATH)

dataset_file = Path("data") / "medindcls_bert.json"
if not dataset_file.exists():
    print("Creating dataset")
    dataset = CleanClinicDataset(tokenizer=tokenizer, clean=False)
    with open(dataset_file, "wb") as f:
        print(f"Saving dataset under {dataset_file}")
        pickle.dump(dataset, f)
else:
    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

idx = np.arange(len(dataset))
random.shuffle(idx)
train_idx, val_idx, test_idx = (
    idx[: int(len(idx) * 0.7)],
    idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
    idx[int(len(idx) * 0.8) :],
)

logging.info(f"Len idx: {len(train_idx)} {len(val_idx)} {len(test_idx)}")

input_ids = torch.cat(
    [
        torch.tensor([x["input_ids"] for x in np.array(dataset.examples)[train_idx]]),
        torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
        torch.tensor([x["input_ids"] for x in np.array(dataset.examples)[val_idx]]),
        torch.tensor([x["input_ids"] for x in np.array(dataset.examples)[test_idx]]),
    ]
)


print(f"First train text examples (first 300 chars): {tokenizer.decode(input_ids[0])[:300]}")
print(f"Label: {y[0]}, {dataset.LE.classes_[0]}")
print(f"First val text examples (first 300 chars): {tokenizer.decode(input_ids[nb_train + nb_word])[:300]}")
print(f"Label: {y[nb_train + nb_word]}, {dataset.LE.classes_[0]}")
print(f"First test text examples (first 300 chars): {tokenizer.decode(input_ids[-nb_test])[:300]}")
print(f"Label: {y[-nb_test]}, {dataset.LE.classes_[0]}")

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
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

logging.info("graph information:")
logging.info(str(graph))

train_idx_dataset = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
val_idx_dataset = Data.TensorDataset(torch.arange(nb_train + nb_word, nb_train + nb_word + nb_val, dtype=torch.long))
test_idx_dataset = Data.TensorDataset(torch.arange(nb_node - nb_test, nb_node, dtype=torch.long))
doc_idx_dataset = Data.ConcatDataset([train_idx_dataset, val_idx_dataset, test_idx_dataset])

idx_loader_train = Data.DataLoader(train_idx_dataset, batch_size=BATCHSIZE)
idx_loader_val = Data.DataLoader(val_idx_dataset, batch_size=BATCHSIZE)
idx_loader_test = Data.DataLoader(test_idx_dataset, batch_size=BATCHSIZE)
idx_loader = Data.DataLoader(doc_idx_dataset, batch_size=BATCHSIZE, shuffle=True)


def update_feature():
    global model, graph, doc_mask
    dataloader = Data.DataLoader(Data.TensorDataset(graph.ndata["input_ids"][doc_mask]), batch_size=64)
    with torch.no_grad():
        model = model.to(device)
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
    return graph


optimizer = torch.optim.Adam(
    [
        {"params": model.bert_model.parameters(), "lr": BERTLR},
        {"params": model.classifier.parameters(), "lr": BERTLR},
        {"params": model.gcn.parameters(), "lr": GCNLR},
    ],
    lr=GCNLR,
)

criterion = torch.nn.CrossEntropyLoss()


def train_step(engine, batch):
    global model, graph, optimizer, criterion
    model.train()
    model = model.to(device)
    graph = graph.to(device)
    (idx,) = [x.to(device) for x in batch]
    train_mask = graph.ndata["train"][idx].type(torch.BoolTensor)
    y_pred = model(graph, idx)[train_mask]
    y_true = graph.ndata["label_train"][idx][train_mask]
    loss = criterion(y_pred, y_true)
    loss.backward()
    if engine.state.iteration % ACCUSTEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
    graph.ndata["cls_feats"].detach_()
    train_loss = loss.item()
    # with torch.no_grad():
    #     if train_mask.sum() > 0:
    #         y_true = y_true.detach().cpu()
    #         y_pred = y_pred.argmax(axis=1).detach().cpu()
    #         train_acc = accuracy_score(y_true, y_pred)
    #     else:
    #         train_acc = 1
    return train_loss  # , train_acc


trainer = Engine(train_step)
trainer.logger = setup_logger(level=30)

# scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

# torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.5)
# scheduler = LRScheduler(torch_lr_scheduler)
# trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

# scheduler = create_lr_scheduler_with_warmup(
#     torch_lr_scheduler, warmup_start_value=0.0, warmup_end_value=LR, warmup_duration=len(idx_loader_train)
# )
# combined_events = Events.ITERATION_STARTED(event_filter=lambda _, __: trainer.state.iteration <= len(idx_loader_train))
# combined_events |= Events.EPOCH_STARTED(event_filter=lambda _, __: trainer.state.epoch > 2)
# trainer.add_event_handler(combined_events, scheduler)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    update_feature()
    torch.cuda.empty_cache()


def eval_step(engine, batch):
    global model, graph
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        graph = graph.to(device)
        (idx,) = [x.to(device) for x in batch]
        y_pred = model(graph, idx)
        y_true = graph.ndata["label"][idx]
        return y_pred, y_true


train_evaluator = Engine(eval_step)
train_evaluator.logger = setup_logger(level=30)
val_evaluator = Engine(eval_step)
val_evaluator.logger = setup_logger(level=30)
test_evaluator = Engine(eval_step)
test_evaluator.logger = setup_logger(level=30)

metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion),
    "cr": SklearnClassificationReport(target_names=[dataset.LE.classes_[x] for x in np.unique(np.array(dataset.labels)[val_idx])])
}

for n, f in metrics.items():
    f.attach(train_evaluator, n)

for n, f in metrics.items():
    f.attach(val_evaluator, n)

for n, f in metrics.items():
    f.attach(test_evaluator, n)


def score_function(engine):
    return -1.0 * engine.state.metrics["nll"]
    # return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "models/gcn",
    n_saved=1,
    filename_pattern=f"{SAVEPATH}_{dataset}_best.pt",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=lambda *_: trainer.state.epoch,
    require_empty=False,
)
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

stopping_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, stopping_handler)


# @trainer.on(Events.ITERATION_COMPLETED(every=LOGINTERVALL))
# def log_training_loss(engine):
#     logging.info(
#         f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output[0]:.2f} Accuracy: {engine.state.output[1]:.2f}"
#     )


# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     train_evaluator.run(idx_loader_train)
#     metrics = train_evaluator.state.metrics
#     logging.info(
#         f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
#     )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(idx_loader_val)
    metrics = val_evaluator.state.metrics
    # scheduler.step(metrics["nll"])
    logging.info(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
    )


@trainer.on(Events.COMPLETED)
def log_test_results(trainer):
    test_evaluator.run(idx_loader_test)
    metrics = test_evaluator.state.metrics
    logging.info(
        f"Test Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
    )
    logging.info(metrics["cr"])


g = update_feature()
trainer.run(idx_loader_train, max_epochs=NEPOCHS)
# trainer.run(idx_loader, max_epochs=NEPOCHS)
