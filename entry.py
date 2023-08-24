import logging
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    LongformerForSequenceClassification,
    LongformerForTokenClassification
)

from params import parse_args

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATADICT = {
    "train": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE400_main",
    "test": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE100_heldout",
    "medindcls": "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv",
}

PRETRAINEDPATHDICT = {
    "bert": "deepset/gbert-base",
    "gbert": "deepset/gbert-base",
    "medbert": "/prj/doctoral_letters/PETGUI/med_bert_local",
    "trans": "deepset/gbert-base",
    "long": "tmp/bert-base-german-cased-4096",
}

PRETRAINEDMODEL = PRETRAINEDPATHDICT[args.model] if not args.bertmodel else PRETRAINEDPATHDICT[args.bertmodel]