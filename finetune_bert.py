import json
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
from ignite.metrics import Accuracy, ClassificationReport, Loss
from ignite.utils import convert_tensor
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from params import parse_args

args = parse_args()

MODELPATH = "deepset/gbert-base"
LEARNINGRATE = 5e-5
NEPOCHS = 50
BATCHSIZE = 8
ACCUSTEPS = 8

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
SAVENAME = Path(MODELPATH).stem

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

model = AutoModelForSequenceClassification.from_pretrained(MODELPATH, num_labels=len(dataset.LE.classes_))

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

train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

# print("First train set example:")
print(f"First train examples (first 300 tokens): {dataset.texts[train_dataset.indices[0]][:300]}")
print(f"Label: {dataset.examples[train_dataset.indices[0]]['labels']}")
print(f"First val examples (first 300 tokens): {dataset.texts[val_dataset.indices[0]][:300]}")
print(f"Label: {dataset.examples[val_dataset.indices[0]]['labels']}")
print(f"First test examples (first 300 tokens): {dataset.texts[test_dataset.indices[0]][:300]}")
print(f"Label: {dataset.examples[test_dataset.indices[0]]['labels']}")
print(f"Len datsets: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")
print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in train_dataset.indices]))
print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in val_dataset.indices]))
print(Counter([dataset.LE.classes_[dataset.labels[x]] for x in test_dataset.indices]))

optimizer = torch.optim.AdamW(model.parameters(), LEARNINGRATE)
criterion = torch.nn.CrossEntropyLoss()


def train_step(engine, batch):
    global model, optimizer, criterion
    model = model.to(torch.device("cuda:0"))
    model.train()
    x, y = batch["input_ids"], batch["labels"]
    x, y = (convert_tensor(x, torch.device("cuda:0")), convert_tensor(y, torch.device("cuda:0")))
    y_pred = model(x).logits
    loss = criterion(y_pred, y)
    loss.backward()
    if engine.state.iteration % ACCUSTEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


trainer = Engine(train_step)

# scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

# torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.5)
# scheduler = create_lr_scheduler_with_warmup(
#     torch_lr_scheduler, warmup_start_value=0.0, warmup_end_value=LEARNINGRATE, warmup_duration=len(train_loader)
# )
# combined_events = Events.ITERATION_STARTED(event_filter=lambda _, __: trainer.state.iteration <= len(train_loader))
# combined_events |= Events.EPOCH_STARTED(event_filter=lambda _, __: trainer.state.epoch > 2)
# trainer.add_event_handler(combined_events, scheduler)


def eval_step(engine, batch):
    global model
    with torch.no_grad():
        model.eval()
        x, y = batch["input_ids"], batch["labels"]
        x, y = (convert_tensor(x, torch.device("cuda:0")), convert_tensor(y, torch.device("cuda:0")))
        y_pred = model(x).logits
        return y_pred, y

train_evaluator = Engine(eval_step)
val_evaluator = Engine(eval_step)
test_evaluator = Engine(eval_step)

metrics = {"accuracy": Accuracy(), "nll": Loss(criterion), "cr": ClassificationReport(output_dict=True, labels=dataset.LE.classes_.tolist())}

for n, f in metrics.items():
    f.attach(train_evaluator, n)

for n, f in metrics.items():
    f.attach(val_evaluator, n)

for n, f in metrics.items():
    f.attach(test_evaluator, n)


def score_function(engine):
    return engine.state.metrics["nll"]
    # return engine.state.metrics["accuracy"]


stopping_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, stopping_handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    # scheduler.step(metrics["nll"])
    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
    )
    if metrics["accuracy"] > log_validation_results.best_val_acc:
        print("New checkpoint")
        torch.save(
            {
                "bert_model": model.bert.state_dict(),
                "classifier": model.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": trainer.state.epoch,
            },
            os.path.join(f"models/finetuned/{SAVENAME}_{dataset}_best.pt"),
        )
        log_validation_results.best_val_acc = metrics["accuracy"]
    # print("Classification report")
    # print(json.dumps(metrics["cr"], indent=4, default=str))


@trainer.on(Events.COMPLETED)
def log_test_results(trainer):
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    print(
        f"Test Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
    )
    print("Classification report")
    print(json.dumps(metrics["cr"], indent=4, default=str))


log_validation_results.best_val_acc = 0
trainer.run(train_loader, max_epochs=NEPOCHS)
