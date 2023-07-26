import datetime
import logging
import pickle
import random
from pathlib import Path

import dgl
import optuna
import torch
import torch.utils.data as Data
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint
# from ignite.handlers.param_scheduler import LRScheduler, create_lr_scheduler_with_warmup
from ignite.metrics import Accuracy, ClassificationReport, Loss
from ignite.utils import setup_logger
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from metrics import SklearnClassificationReport
from model import BertGCN
from params import parse_args
from utils import *

args = parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

MODELTYPE = "deepset/gbert-base"
BERTLR = 5e-5
GCNLR = 3e-5
LR = 4e-5
BATCHSIZE = 8
NEPOCHS = 50
ACCUSTEPS = 8
LOGINTERVALL = 100

if args.data == "MIC":
	DATASET = "med_indication_all_RF_diag"
	DATASETPATH =  Path("data") / f"ind.{DATASET}"
	BERTPATH = Path("models/finetuned/gbert-base_med_indication_all_RF_diag_best.pt")
elif args.data == "CSC":
	DATASET = "CARDIODE400_main"
	DATASETPATH =  Path("data") / f"ind.{DATASET}"
	BERTPATH = Path("models/finetuned/gbert-base_CARDIODE400_main_best.pt")
elif args.data == "Patho":
	pass


tokenizer = AutoTokenizer.from_pretrained(MODELTYPE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
	format=f"%(asctime)s - %(message)s",
	# format=f"%(asctime)s ({args.mixfactor}) - %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
	handlers=[
		logging.StreamHandler(),
	],
)

logging.info(f"{'=== Params ===':>32}")
for k, v in vars(args).items():
	logging.info(f"{k:>25} : {str(v):<25}")

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

if args.data == "MIC":
    dataset_file = Path("data") / "medindcls_bert.json"
    if not dataset_file.exists():
        print("Creating dataset")
        dataset = CleanClinicDataset(tokenizer=tokenizer, task="MIC", clean=False)
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
        train_dataset = CleanClinicDataset(tokenizer=tokenizer, task="CSC", clean=False, mode="train")
        with open(train_dataset_file, "wb") as f:
            print(f"Saving dataset under {train_dataset_file}")
            pickle.dump(train_dataset, f)
    else:
        print(f"Loading train dataset from: {train_dataset_file}")
        with open(train_dataset_file, "rb") as f:
            train_dataset = pickle.load(f)
    if not test_dataset_file.exists():
        print("Creating test dataset")
        test_dataset = CleanClinicDataset(tokenizer=tokenizer, task="CSC", clean=False, mode="test")
        with open(test_dataset_file, "wb") as f:
            print(f"Saving dataset under {test_dataset_file}")
            pickle.dump(test_dataset, f)
    else:
        print(f"Loading test dataset from: {test_dataset_file}")
        with open(test_dataset_file, "rb") as f:
            test_dataset = pickle.load(f)
    dataset = train_dataset
elif args.data == "Patho":
    pass

GCNNAME = f"{Path(MODELTYPE).stem}_{dataset}.pt"
SAVEPATH = Path(f"models/gcn/{args.mixfactor}")
GCNPATH = SAVEPATH / GCNNAME

if args.data == "MIC":
	idx = np.arange(len(dataset))
	random.shuffle(idx)
	train_idx, val_idx, test_idx = (
		idx[: int(len(idx) * 0.7)],
		idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)],
		idx[int(len(idx) * 0.8) :],
	)
	# train_dataset = Subset(dataset, train_idx)
	# val_dataset = Subset(dataset, val_idx)
	# test_dataset = Subset(dataset, test_idx)
elif args.data == "CSC":
	idx = np.arange(len(train_dataset))
	random.shuffle(idx)
	train_idx, val_idx = idx[: int(len(idx) * 0.9)], idx[int(len(idx) * 0.9) :]
	# train_dataset = Subset(dataset, train_idx)
	# val_dataset = Subset(dataset, val_idx)
elif args.data == "Patho":
	pass


def run(trial):

	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, _, _ = load_corpus(DATASETPATH)

	nb_node = features.shape[0]
	nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
	nb_word = nb_node - nb_train - nb_val - nb_test
	nb_class = y_train.shape[1]

	model = BertGCN(
		nb_class=nb_class,
		pretrained_model="deepset/gbert-base", # parameters will be overwritten with fine-tuned weights
		mix_factor=args.mixfactor if trial in ["train", "test"] else trial.suggest_uniform("mix_factor", 0, 1),
		gcn_layers=2,
		n_hidden=200,
		dropout=0.5,
	)

	if trial == "test":
		logging.info(
			f"Loading pretrained gcn model from {GCNPATH} saved on {datetime.datetime.fromtimestamp(GCNPATH.stat().st_ctime)}"
		)
		model.load_state_dict(torch.load(GCNPATH))
		# model.load_state_dict(torch.load(GCNPATH, map_location=torch.device("cpu")))
	else:
		logging.info(
			f"Loading pretrained bert model from {BERTPATH} saved on {datetime.datetime.fromtimestamp(BERTPATH.stat().st_ctime)}"
		)
		ckpt = torch.load(BERTPATH, map_location=device)
		model.bert_model.load_state_dict(ckpt["bert_model"])
		model.classifier.load_state_dict(ckpt["classifier"])

	# transform one-hot label to class ID for pytorch computation
	y = y_train + y_test + y_val
	y_train = y_train.argmax(axis=1)
	y = y.argmax(axis=1)

	# document mask used for update feature
	doc_mask = train_mask + val_mask + test_mask

	# tokenizer = AutoTokenizer.from_pretrained(MODELTYPE)

	# logging.info(f"Len idx: {len(train_idx)} {len(val_idx)} {len(test_idx)}")

	if args.data == "MIC":
		input_ids = torch.cat(
			[
				torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[train_idx]])),
				torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
				torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[val_idx]])),
				torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[test_idx]])),
			]
		)
	elif args.data == "CSC":
		input_ids = torch.cat(
			[
				torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[train_idx]])),
				torch.zeros((nb_word, tokenizer.model_max_length), dtype=torch.long),
				torch.tensor(np.array([x["input_ids"] for x in np.array(dataset.examples)[val_idx]])),
				torch.tensor(np.array([x["input_ids"] for x in test_dataset.examples])),
			]
		)
	elif args.data == "Patho":
		pass

	# logging.info(f"First train text examples (first 300 chars): {tokenizer.decode(input_ids[0])[:300]}")
	# logging.info(f"Label: {y[0]}, {dataset.LE.classes_[y[0]]}")
	# train_labels = np.unique(dataset.labels[train_idx])
	# logging.info(f"Train labels: {train_labels}, {len(train_labels)}")
	assert np.array_equal(y[:nb_train], dataset.labels[train_idx])
	# logging.info(f"First val text examples (first 300 chars): {tokenizer.decode(input_ids[nb_train + nb_word])[:300]}")
	# logging.info(f"Label: {y[nb_train + nb_word]}, {dataset.LE.classes_[y[nb_train + nb_word]]}")
	# val_labels = np.unique(dataset.labels[val_idx])
	# logging.info(f"Val labels: {val_labels}, {len(val_labels)}")
	assert np.array_equal(y[nb_train + nb_word : nb_train + nb_word + nb_val], dataset.labels[val_idx])
	# logging.info(f"First test text examples (first 300 chars): {tokenizer.decode(input_ids[-nb_test])[:300]}")
	# logging.info(f"Label: {y[-nb_test]}, {dataset.LE.classes_[y[-nb_test]]}")
	# test_labels = np.unique(dataset.labels[test_idx])
	# logging.info(f"Test labels: {test_labels}, {len(test_labels)}")
	assert np.array_equal(y[-nb_test:], test_dataset.labels)

	# build DGL Graph
	adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

	train_idx_dataset = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
	val_idx_dataset = Data.TensorDataset(
		torch.arange(nb_train + nb_word, nb_train + nb_word + nb_val, dtype=torch.long)
	)
	test_idx_dataset = Data.TensorDataset(torch.arange(nb_node - nb_test, nb_node, dtype=torch.long))

	idx_loader_train = Data.DataLoader(train_idx_dataset, batch_size=BATCHSIZE)
	idx_loader_val = Data.DataLoader(val_idx_dataset, batch_size=BATCHSIZE)
	idx_loader_test = Data.DataLoader(test_idx_dataset, batch_size=BATCHSIZE)

	optimizer = torch.optim.Adam(
		[
			{"params": model.bert_model.parameters(), "lr": BERTLR},
			{"params": model.classifier.parameters(), "lr": BERTLR},
			{"params": model.gcn.parameters(), "lr": GCNLR},
		],
		lr=GCNLR,
	)

	criterion = torch.nn.CrossEntropyLoss()

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

	def train_step(engine, batch):
		nonlocal model, graph, optimizer, criterion
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
		return train_loss

	def update_feature():
		nonlocal graph, model
		dataloader = Data.DataLoader(Data.TensorDataset(graph.ndata["input_ids"][doc_mask]), batch_size=64)
		with torch.no_grad():
			model = model.to(device)
			model.eval()
			cls_list = []
			logging.info("Updating features...")
			for batch in dataloader:
				input_ids = [x.to(device) for x in batch][0]
				output = model.bert_model(input_ids=input_ids)[0][:, 0]
				cls_list.append(output.cpu())
			cls_feat = torch.cat(cls_list, axis=0)
		graph = graph.to("cpu")
		graph.ndata["cls_feats"][doc_mask] = cls_feat

	trainer = Engine(train_step)
	trainer.logger = setup_logger("trainer", level=30)

	scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)

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
		nonlocal model, graph
		with torch.no_grad():
			model.eval()
			model = model.to(device)
			graph = graph.to(device)
			(idx,) = [x.to(device) for x in batch]
			y_pred = model(graph, idx)
			y_true = graph.ndata["label"][idx]
			return y_pred, y_true

	# train_evaluator = Engine(eval_step)
	# train_evaluator.logger = setup_logger("train evaluator", level=30)

	val_evaluator = Engine(eval_step)
	val_evaluator.logger = setup_logger("val evaluator", level=30)

	if trial not in ["train", "test"]:
		pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", trainer)
		val_evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

	test_evaluator = Engine(eval_step)
	test_evaluator.logger = setup_logger("test evaluator", level=30)

	metrics = {
		"accuracy": Accuracy(),
		"nll": Loss(criterion),
		"cr": SklearnClassificationReport(target_names=dataset.LE.classes_),
	}

	# for n, f in metrics.items():
	#     f.attach(train_evaluator, n)

	for n, f in metrics.items():
		f.attach(val_evaluator, n)

	for n, f in metrics.items():
		f.attach(test_evaluator, n)

	def score_function(engine):
		return -1.0 * engine.state.metrics["nll"]

	# val_evaluator.run(idx_loader_val)

	to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}

	# checkpoint = ModelCheckpoint(
	#     to_save,
	#     SAVEPATH,
	#     n_saved=1,
	#     filename_pattern=GCNNAME,
	#     score_function=score_function,
	#     score_name="accuracy",
	#     global_step_transform=lambda *_: trainer.state.epoch,
	#     require_empty=False,
	# )

	model_checkpoint = ModelCheckpoint(
		SAVEPATH,
		n_saved=1,
		filename_pattern=GCNNAME,
		score_function=score_function,
		score_name="accuracy",
		global_step_transform=lambda *_: trainer.state.epoch,
		require_empty=False,
	)

	val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

	stopping_handler = EarlyStopping(patience=args.patience, score_function=score_function, trainer=trainer)
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
		scheduler.step(metrics["nll"])
		logging.info(
			f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
		)

	# @trainer.on(Events.COMPLETED)
	# def log_test_results(trainer):
	#     test_evaluator.run(idx_loader_test)
	#     metrics = test_evaluator.state.metrics
	#     logging.info(
	#         f"Test Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
	#     )
	#     logging.info(metrics["cr"])

	if not args.suppressupdates:
		update_feature()

	if trial != "test":
		trainer.run(idx_loader_train, max_epochs=NEPOCHS)
		if trial != "train":
			return val_evaluator.state.metrics["accuracy"]

	logging.info(
		f"Loading best gcn model from {GCNPATH} saved on {datetime.datetime.fromtimestamp(GCNPATH.stat().st_ctime)}"
	)
	model.load_state_dict(torch.load(GCNPATH))

	# checkpoint = torch.load(GCNPATH, map_location=device) 
	# Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint) 

	
	update_feature()
	test_evaluator.run(idx_loader_test)
	metrics = test_evaluator.state.metrics
	logging.info(
		f"Test results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}"
	)
	logging.info(metrics["cr"])


def optimize():
	study = optuna.create_study(direction="maximize")
	study.optimize(run, n_trials=10)

	print("Number of finished trials: ", len(study.trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))


if args.optimize:
	optimize()
else:
	run("train")
	# run("test")
