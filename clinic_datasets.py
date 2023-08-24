# from entry import *
import glob
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
		task,
		doclevel,
		mode=None,
		dev=False,
		clean=True,
		noarznei=False
	):
		if task == "MIC":
			self.file_path = "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv"
		elif task == "CSC":
			if mode == "train":
				self.file_path = "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE400_main"
			elif mode == "test":
				self.file_path = "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE100_heldout"
		elif task == "Patho":
			pass

		self.LE = LabelEncoder()
		self.arzneiLE = LabelEncoder()
		self.OHE = OneHotEncoder()
		if task == "MIC":
			lines = open(self.file_path, "r").readlines()
			self.texts, self.labels, diagnoses, anamneses, risk_factors = list(), list(), list(), list(), list()
			self.label2id = dict()
			self.arznei = list()
			enum = 0
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
				if not medication_name in self.label2id:
					self.label2id[medication_name] = enum
					enum += 1
				arznei = "" if noarznei else f"Arznei {medication_name} & "
				self.arznei.append(medication_name)
				if doclevel=="letter":
					text_list = f"{arznei}{discharge_letter.strip()}".split()
				elif doclevel=="diagnosis":
					text_list = f"{arznei}{diagnosis.strip()}".split()
				elif doclevel=="riskfactor":
					text_list = f"{arznei}{risk_factor.strip()}".split()
				elif doclevel=="anamnesis":
					text_list = f"{arznei}{anamnesis.strip()}".split()

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
			self.arznei = np.array(self.arzneiLE.fit_transform(self.arznei))
			self.ohe_labels = self.OHE.fit_transform(np.array(self.labels).reshape(-1, 1))

			batch_encoding = tokenizer(
				self.texts, add_special_tokens=True, truncation=True, is_split_into_words=False, padding="max_length"
			)
			self.examples = [{"input_ids": np.array(e)} for e in batch_encoding["input_ids"]]
			for l, e, a in zip(self.labels, self.examples, self.arznei):
				e.update({"labels": l})
				e.update({"arznei": a})
		elif task == "CSC":
			self.texts, self.labels = list(), list()
			files = glob.glob(str(Path(self.file_path) / "*"))
			for file in files:
				with open(file, encoding="utf-8") as f:
					pars = [line.strip() for line in f.read().split("\n\n") if len(line.split("\t")) > 1]
					for par in pars:
						text = list()
						clean_text = list()
						lines = par.split("\n")[1:]
						for line in lines:
							token = line.split("\t")[2]
							label = line.split("\t")[6]
							if label == "_":
								break
							if not clean or token.lower() not in stop_words:
								text.append(token)
							if token.lower() not in stop_words:
								clean_text.append(token)
						if clean_text:
							self.texts.append(text)
							self.labels.append(label)

			self.labels = np.array(self.LE.fit_transform(self.labels))
			self.ohe_labels = self.OHE.fit_transform(np.array(self.labels).reshape(-1, 1))

			batch_encoding = tokenizer(
				self.texts,
				add_special_tokens=True,
				truncation=True,
				is_split_into_words=True,
				padding="max_length"
			)
			self.examples = [{"input_ids": np.array(e)} for e in batch_encoding["input_ids"]]
			for l, e, a in zip(self.labels, self.examples, self.arznei):
				e.update({"labels": l})
				e.update({"arznei": a})
			
			self.texts = [" ".join(x) for x in self.texts]


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, i):
		return self.examples[i]

	def __str__(self):
		return Path(self.file_path).stem
