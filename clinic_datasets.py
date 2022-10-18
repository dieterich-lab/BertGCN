# from entry import *
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset

stop_words = stopwords.words("german")


class CleanClinicDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path="/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv",
        task="medindcls",
        dev=False,
        clean=True,
    ):
        self.file_path = file_path
        self.LE = LabelEncoder()
        self.OHE = OneHotEncoder()
        self.task = task
        lines = open(self.file_path, "r").readlines()
        self.texts, self.labels, diagnoses, anamneses, risk_factors = list(), list(), list(), list(), list()
        for i, line in enumerate(lines):
            (
                _,
                diagnosis,
                anamnesis,
                risk_factor,
                discharge_letter,
                medication_type,
                medication_name,
                label,
            ) = line.split("|||")

            text_list = f"Arznei {medication_name} & {discharge_letter.strip()}".split()
            if clean:
                self.texts.append(" ".join(x for x in text_list if x.lower() not in stop_words))
            else:
                self.texts.append(" ".join(x for x in text_list))
            self.labels.append(f"{medication_type.strip()}_{label.strip()}")
            diagnoses.append(diagnosis.strip())
            anamneses.append(anamnesis.strip())
            risk_factors.append(risk_factor.strip())
            if dev and (i - 1) == dev:
                break

        self.labels = np.array(self.LE.fit_transform(self.labels))
        self.ohe_labels = self.OHE.fit_transform(np.array(self.labels).reshape(-1, 1))

        batch_encoding = tokenizer(
            self.texts, add_special_tokens=True, truncation=True, is_split_into_words=False, padding="max_length"
        )
        self.examples = [{"input_ids": np.array(e)} for e in batch_encoding["input_ids"]]
        for l, e in zip(self.labels, self.examples):
            e.update({"labels": l})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.examples[i]

    def __str__(self):
        return Path(self.file_path).stem
