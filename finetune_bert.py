import datetime
import logging
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.metrics import Accuracy, ClassificationReport, Loss, Precision, Recall
from ignite.utils import convert_tensor, setup_logger
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from entry import *
from metrics import SklearnClassificationReport
from params import parse_args
from utils import *

args = parse_args()

LEARNINGRATE = 5e-5
NEPOCHS = 50
BATCHSIZE = 8
ACCUSTEPS = 8

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
tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)

if args.data == "MIC":
    if args.noarznei:
        dataset_file = (
            Path("data") / f"medindcls_{args.bertmodel}_{args.doclevel}_noarznei.json"
        )
    else:
        dataset_file = Path("data") / f"medindcls_{args.bertmodel}_{args.doclevel}.json"
    if not dataset_file.exists():
        logging.info("Creating dataset")
        dataset = CleanClinicDataset(
            tokenizer=tokenizer,
            task="MIC",
            doclevel=args.doclevel,
            clean=False,
            noarznei=args.noarznei,
        )
        with open(dataset_file, "wb") as f:
            logging.info(f"Saving dataset under {dataset_file}")
            pickle.dump(dataset, f)
    else:
        logging.info(f"Loading dataset from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
elif args.data == "CSC":
    train_dataset_file = (
        Path("data") / f"csc_train_{args.bertmodel}_{args.doclevel}.json"
    )
    test_dataset_file = Path("data") / f"csc_test_{args.bertmodel}_{args.doclevel}.json"

    if not train_dataset_file.exists():
        logging.info("Creating train dataset")
        train_dataset = CleanClinicDataset(
            tokenizer=tokenizer, task="CSC", clean=False, mode="train"
        )
        with open(train_dataset_file, "wb") as f:
            logging.info(f"Saving dataset under {train_dataset_file}")
            pickle.dump(train_dataset, f)
    else:
        logging.info(f"Loading train dataset from: {train_dataset_file}")
        with open(train_dataset_file, "rb") as f:
            train_dataset = pickle.load(f)
    if not test_dataset_file.exists():
        logging.info("Creating test dataset")
        test_dataset = CleanClinicDataset(
            tokenizer=tokenizer, task="CSC", clean=False, mode="test"
        )
        with open(test_dataset_file, "wb") as f:
            logging.info(f"Saving dataset under {test_dataset_file}")
            pickle.dump(test_dataset, f)
    else:
        logging.info(f"Loading test dataset from: {test_dataset_file}")
        with open(test_dataset_file, "rb") as f:
            test_dataset = pickle.load(f)
    dataset = train_dataset

SAVENAME = Path(PRETRAINEDMODEL).stem
if args.data == "MIC":
    SAVEDIR = Path(f"models/finetuned/{args.doclevel}")
    os.makedirs(SAVEDIR, exist_ok=True)
    if args.testunklar:
        SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_testunklar_best.pt")
    if args.noarznei:
        SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_noarznei_best.pt")
    else:
        SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_best.pt")
elif args.data == "CSC":
    SAVEDIR = Path(f"models/finetuned")
    os.makedirs(SAVEDIR, exist_ok=True)
    SAVEPATH = Path(f"{SAVEDIR}/{SAVENAME}_{dataset}_best.pt")

if not args.cv: 
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINEDMODEL, num_labels=len(dataset.LE.classes_)
    )

    optimizer = torch.optim.AdamW(model.parameters(), LEARNINGRATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

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
elif args.data == "Patho":
    pass

if not args.cv:
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)
else:
    train_loaders, val_loaders, test_loaders = list(), list(), list()
    for i in range(10):
        train_loaders.append(
            DataLoader(train_datasets[i], batch_size=BATCHSIZE, shuffle=True)
        )
        val_loaders.append(
            DataLoader(val_datasets[i], batch_size=BATCHSIZE, shuffle=False)
        )
        test_loaders.append(
            DataLoader(test_datasets[i], batch_size=BATCHSIZE, shuffle=False)
        )

# print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in train_dataset.indices]))
# print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in val_dataset.indices]))
# if args.data == "MIC":
#     print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in test_dataset.indices]))
# elif args.data == "CSC":
#     print(Counter([dataset.LE.classes_[x["labels"]] for x in test_dataset]))

# print("First train set example:")
# print(f"Text: {dataset.texts[train_dataset.indices[0]]}")
# print(f"Label: {dataset.examples[train_dataset.indices[0]]['labels']}")

# print("First train set example:")
# print(f"Text: {dataset.texts[train_dataset.indices[0]]}")
# print(f"Label: {dataset.examples[train_dataset.indices[0]]['labels']}")

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


# torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.5)
# scheduler = create_lr_scheduler_with_warmup(
#     torch_lr_scheduler, warmup_start_value=0.0, warmup_end_value=LEARNINGRATE, warmup_duration=len(train_loader)
# )
# combined_events = Events.ITERATION_STARTED(event_filter=lambda _, __: trainer.state.iteration <= len(train_loader))
# combined_events |= Events.EPOCH_STARTED(event_filter=lambda _, __: trainer.state.epoch > 2)
# trainer.add_event_handler(combined_events, scheduler)


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
F1 = (precision * recall * 2 / (precision + recall))

metrics = {
    "accuracy": Accuracy(),
    "f1": F1,
    "precision": Precision(average="macro"),
    "recall": Recall(average="macro"),
    "nll": Loss(criterion),
    "cr": SklearnClassificationReport(
        target_names=[
            dataset.LE.classes_[x] for x in np.unique(np.array(dataset.labels))
        ]
    ),
}

for n, f in metrics.items():
    f.attach(val_evaluator, n)

for n, f in metrics.items():
    f.attach(test_evaluator, n)


def score_function(engine):
    return -1.0 * engine.state.metrics["nll"]


stopping_handler = EarlyStopping(
    patience=3, score_function=score_function, trainer=trainer
)
val_evaluator.add_event_handler(Events.COMPLETED, stopping_handler)


# @trainer.on(Events.COMPLETED)
# def log_test_results(trainer):
#     test_evaluator.run(test_loader)
#     metrics = test_evaluator.state.metrics
#     print(
#         f"Test Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
#     )
#     print("Classification report")
#     print(metrics["cr"])


if not args.cv:
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        scheduler.step(metrics["nll"])
        logging.info(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
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
    logging.info(metrics["cr"])
else:
    f1s, accs = list(), list()
    for i in range(10):
        logging.info(f"Training split {i}...")

        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINEDMODEL, num_labels=len(dataset.LE.classes_)
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), LEARNINGRATE)

        SAVEPATH = Path(f"{SAVEDIR}/{i}/{SAVENAME}_{dataset}_best.pt")
        os.makedirs(Path(f"{SAVEDIR}/{i}"), exist_ok=True)

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

        scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

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
        stopping_handler = EarlyStopping(
            patience=3, score_function=score_function, trainer=trainer
        )
        val_evaluator.add_event_handler(Events.COMPLETED, stopping_handler)

        test_evaluator = Engine(eval_step)

        metrics = {
            "accuracy": Accuracy(),
            "f1": F1,
            "precision": Precision(average="macro"),
            "recall": Recall(average="macro"),
            "nll": Loss(criterion),
            "cr": SklearnClassificationReport(
                target_names=[
                    dataset.LE.classes_[x] for x in np.unique(np.array(dataset.labels))
                ]
            ),
        }

        for n, f in metrics.items():
            f.attach(val_evaluator, n)

        for n, f in metrics.items():
            f.attach(test_evaluator, n)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(val_loaders[i])
            metrics = val_evaluator.state.metrics
            prec, rec, f1, acc = metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]
            logging.info(
                f"Validation Results - Epoch: {trainer.state.epoch} Prec: {prec:.2f} Rec: {rec:.2f} Acc: {acc:.2f}  F1: {f1:.2f} Avg accuracy: {acc:.2f} Avg loss: {metrics['nll']:.2f}"
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
            scheduler.step(metrics["nll"])

        log_validation_results.best_val_acc = 0


        trainer.run(train_loaders[i], max_epochs=NEPOCHS)

        logging.info(
            f"Loading best BERT model from {SAVEPATH} saved on {datetime.datetime.fromtimestamp(SAVEPATH.stat().st_ctime)}"
        )
        ckpt = torch.load(SAVEPATH)
        model.bert.load_state_dict(ckpt["bert_model"])
        model.classifier.load_state_dict(ckpt["classifier"])
        test_evaluator.run(test_loaders[i])
        metrics = test_evaluator.state.metrics
        prec, rec, f1, acc = metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]
        logging.info(
            f"Test Results - Epoch[{trainer.state.epoch}] Precison: {prec:.2f} Recall: {rec:.2f} F1: {f1:.2f} Avg accuracy: {acc:.2f} Avg loss: {metrics['nll']:.2f}"
        )
        f1s.append(f1)
        accs.append(acc)
        logging.info(metrics["cr"])
    logging.info(f"Mean F1 scores: {np.mean(f1s)}, Std: {np.std(f1s)}")
    logging.info(f"Mean accuracies: {np.mean(accs)}, Std: {np.std(accs)}")
