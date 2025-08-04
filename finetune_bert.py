import datetime
import logging
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.metrics import Accuracy, ClassificationReport, Loss, Precision, Recall
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.utils import convert_tensor, setup_logger
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from entry import *
from metrics import SklearnClassificationReport
from params import parse_args
from utils import *

args = parse_args()

# LR = 5e-6
LR = 1e-5
NEPOCHS = args.nepochs
BATCHSIZE = 8
# ACCUSTEPS = 8
ACCUSTEPS = 1

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M")

LOGPATH = Path("logs") / "finetune" / args.data
os.makedirs(LOGPATH, exist_ok=True)

handlers = [
    logging.FileHandler(LOGPATH / f"{now_str}.log", mode="w"),
    logging.StreamHandler(),
]

logging.basicConfig(
    format=f"%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=handlers,
)

logging.info(f"Learning rate {LR}, Batch size {BATCHSIZE} Accu steps {ACCUSTEPS}")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)


dataset_file = Path("data") / f"medindcls_{args.bertmodel}_{args.doclevel}.json"
if not dataset_file.exists():
    logging.info("Creating dataset")
    dataset = CleanClinicDataset(
        tokenizer=tokenizer,
        task="MIC",
        doclevel=args.doclevel,
        clean=False,
    )
    with open(dataset_file, "wb") as f:
        logging.info(f"Saving dataset under {dataset_file}")
        pickle.dump(dataset, f)
else:
    logging.info(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

SAVENAME = Path(PRETRAINEDMODEL).stem
SAVEDIR = Path(f"models/finetuned/{args.doclevel}")
os.makedirs(SAVEDIR, exist_ok=True)
if args.testunklar:
    SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_testunklar_best.pt")
else:
    SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_best.pt")

model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINEDMODEL, num_labels=len(dataset.LE.classes_)
)

optimizer = torch.optim.AdamW(model.parameters(), LR)
scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

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

train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()


def train_step(engine, batch):
    global model, optimizer, criterion
    model = model.to(torch.device("cuda:0"))
    model.train()
    x, y = batch["input_ids"], batch["labels"]
    x, y = (
        convert_tensor(x, torch.device("cuda:0")),
        convert_tensor(y, torch.device("cuda:0")),
    )
    y_pred = model(x).logits
    loss = criterion(y_pred, y)
    loss.backward()
    if engine.state.iteration % ACCUSTEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


trainer = Engine(train_step)
trainer.logger = setup_logger("trainer", level=30)


def eval_step(engine, batch):
    global model
    model = model.to(torch.device("cuda:0"))
    with torch.no_grad():
        model.eval()
        x, y = batch["input_ids"], batch["labels"]
        x, y = (
            convert_tensor(x, torch.device("cuda:0")),
            convert_tensor(y, torch.device("cuda:0")),
        )
        y_pred = model(x).logits
        return y_pred, y


val_evaluator = Engine(eval_step)
test_evaluator = Engine(eval_step)

precision = Precision(average="macro")
recall = Recall(average="macro")

metrics = {
    "accuracy": Accuracy(),
    "f1": Precision(average="macro")
    * Recall(average="macro")
    * 2
    / (Precision(average="macro") + Recall(average="macro")),
    "precision": Precision(average="macro"),
    "recall": Recall(average="macro"),
    "nll": Loss(criterion),
    "cr": SklearnClassificationReport(
        target_names=[
            dataset.LE.classes_[x] for x in np.unique(np.array(dataset.labels))
        ]
    ),
    "cr_dict": SklearnClassificationReport(
        output_dict=True,
        target_names=[
            dataset.LE.classes_[x] for x in np.unique(np.array(dataset.labels))
        ],
    ),
    "cm": ConfusionMatrix(num_classes=len(dataset.LE.classes_)),
}

for n, f in metrics.items():
    f.attach(val_evaluator, n)

for n, f in metrics.items():
    f.attach(test_evaluator, n)


def score_function(engine):
    return -1.0 * engine.state.metrics["nll"]


def f1_function(engine):
    return engine.state.metrics["f1"]


stopping_handler = EarlyStopping(
    patience=args.patience,
    # score_function=f1_function,
    score_function=score_function,
    trainer=trainer,
)
val_evaluator.add_event_handler(Events.COMPLETED, stopping_handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    scheduler.step(metrics["nll"])
    prec, rec, f1, acc = (
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["accuracy"],
    )
    logging.info(
        f"Validation Results - Epoch: {trainer.state.epoch} Prec: {prec:.2f} Rec: {rec:.2f} F1: {f1:.2f} Acc: {acc:.2f}  Avg loss: {metrics['nll']:.2f}"
    )
    if metrics["accuracy"] > log_validation_results.best_val_acc:
        torch.save(
            {
                "bert_model": model.bert.state_dict(),
                "classifier": model.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": trainer.state.epoch,
                "config": model.config,
            },
            SAVEPATH,
        )
        log_validation_results.best_val_acc = metrics["accuracy"]

    log_validation_results.best_val_acc = 0

    if not args.testonly:
        trainer.run(train_loader, max_epochs=NEPOCHS)

    logging.info(
        f"Loading best BERT model from {SAVEPATH} saved on {datetime.datetime.fromtimestamp(SAVEPATH.stat().st_ctime)}"
    )
    ckpt = torch.load(SAVEPATH)
    model.bert.load_state_dict(ckpt["bert_model"])
    model.classifier.load_state_dict(ckpt["classifier"])
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    logging.info(
        f"Test Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
    )

    report = metrics["cr_dict"]
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{SAVEDIR}/{args.bertmodel}_cr.csv")
    logging.info(df.to_latex(index=False, float_format="{:.2f}".format))
    logging.info(metrics["cr"])
    cm = metrics["cm"].detach().cpu().numpy()
    with open(f"{SAVEDIR}/{args.bertmodel}_cm.npy", "wb") as f:
        np.save(f, cm)
