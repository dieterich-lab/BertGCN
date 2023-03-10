import datetime
import shap
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
from ignite.metrics import Accuracy, Loss
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from metrics import SklearnClassificationReport
from ignite.metrics.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import *
from model import BertGAT, BertGCN
import torch.utils.data as Data
import dgl


from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

SAVEPATH = Path(f"models/finetuned/{SAVENAME}_{dataset}_best.pt")

model = AutoModelForSequenceClassification.from_pretrained(MODELPATH, num_labels=len(dataset.LE.classes_))
model = model.to(device)

idx = np.arange(len(dataset))
random.shuffle(idx)

_, _, test_idx = (
    idx[: int(len(idx) * 0.7)],
    idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
    idx[int(len(idx) * 0.8) :],
)

test_dataset = Subset(dataset, test_idx)

def forward(inputs):
    preds = model(inputs)
    print(preds.logits.shape)
    return preds.logits

token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
lig = LayerIntegratedGradients(forward, model.bert.embeddings)


vis_data_records_ig = []

def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                        attributions,
                        pred,
                        dataset.LE.classes_[pred_ind], # predicted
                        dataset.LE.classes_[label], # true
                        dataset.LE.classes_[label], # attributed
                        attributions.sum(),
                        text,
                        delta))

def interpret_sentence(model, indexed, label):
    model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)
    pred = model(input_indices).logits
    pred_ind = torch.argmax(pred).item()
    pred = torch.argmax(torch.softmax(pred, dim=-1)).item()

    reference_indices = token_reference.generate_reference(len(indexed), device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    label = torch.tensor(label)
    attributions_ig, delta = lig.attribute(input_indices, reference_indices, n_steps=50, target=label, return_convergence_delta=True)
    text = tokenizer.convert_ids_to_tokens(indexed)
    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)

for sample in np.array(dataset.examples)[test_idx][:1]:
    interpret_sentence(model, sample["input_ids"], sample["labels"])


def shap_func(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=500, truncation=True) for v in x])
    outputs = model(tv)
    return torch.softmax(outputs.logits, dim=-1).detach().numpy()

explainer = shap.Explainer(shap_func, tokenizer)

for sample in np.array(dataset.texts)[test_idx][24:25]:
    label = np.array(dataset.examples)[test_idx][24]["labels"]
    shap_values = explainer([sample])
    shap.plots.text(shap_values[0, :, label])
    shap.plots.bar(shap_values[0, :, label], order=shap.Explanation.argsort.flip)