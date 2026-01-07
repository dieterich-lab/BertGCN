# Copilot Instructions

## Architecture
- The production pipeline is sequential: [bertgcn/preprocess.py](bertgcn/preprocess.py) cleans `data/med_indication_all_RF_diag.csv`, tokenizes it with the configured Hugging Face tokenizer, and saves `data/processed/tokenized_dataset` plus the label encoders that every downstream training path consumes.
- [bertgcn/train_bert.py](bertgcn/train_bert.py) is the Hydra/MLflow-powered BERT finetune entry point: it loads the processed dataset, optionally stratifies splits, applies class weighting, and defers to `transformers.Trainer` with `mlflow.transformers.autolog()` (fallbacks to `mlflow.pytorch`). Every run uses the composed config rooted in [conf/config.yaml](conf/config.yaml).
- [bertgcn/build_graph.py](bertgcn/build_graph.py) plus [bertgcn/clinic_datasets.py](bertgcn/clinic_datasets.py) generate the document-word graph for the original BertGCN model via sliding windows, PMI edges, and TF-IDF edges; graph components (`adj`, `x`, `tx`, etc.) land in `data/ind.*` files for downstream use.
- The minimal educational GCN runner in [bertgcn/train_gcn.py](bertgcn/train_gcn.py) consumes the same processed dataset, builds a random adjacency, and follows the hyperparameters defined in [conf/minimal_gcn_config.yaml](conf/minimal_gcn_config.yaml).

## Workflows
- Run `poetry install` to provision `torch==2.8.0`, the PyTorch Geometric dependencies, Hydra, and MLflow as listed in [pyproject.toml](pyproject.toml).
- Preprocess the CSV once (and after any recipe changes) via `poetry run python -m bertgcn.preprocess`; look for `data/processed/tokenized_dataset` plus `label_encoder.joblib`/`meds_label_encoder.joblib` afterward.
- Build the document-word graph with `poetry run build-graph`; override CLI flags backed by [bertgcn/params.py](bertgcn/params.py) (e.g., `--window-size 30 --bertmodel gbert`).
- Fine-tune the transformer with `poetry run finetune-bert`. Hydra controls the run directory (`hydra.run.dir=hydra/finetune/<timestamp>` by default) and accepts overrides such as `hparams.learning_rate=3e-5` or `hydra.run.dir=outputs/custom_run` directly on the command line.
- Train the toy GCN with `poetry run train-bertgcn` when you need a quick sanity check; this hydrates the `minimal_gcn_config` defaults and still logs to MLflow (`minimal_gcn_training`).
- Inspect metrics/artifacts with `mlflow ui --backend-store-uri mlruns` (the project defaults to a file store at `mlruns/` unless `mlflow_tracking_uri` under [conf/main.yaml](conf/main.yaml) overrides it).
- Run tests with `poetry run pytest`; the suite exercises the Hydra run-dir guardrails and graph-building edge cases.

## Config & Patterns
- [conf/config.yaml](conf/config.yaml) composes `main`, `hparams`, and `dataset`, and it swaps in the Optuna sweeper via `override hydra/sweeper: optuna`. Sweeps land under `multirun/<timestamp>` and expect `hydra.job.num` to pattern the subdirectories.
- Global defaults live in [conf/main.yaml](conf/main.yaml) (project name, MLflow experiment, optional `mlflow_tracking_uri`), while [conf/hparams.yaml](conf/hparams.yaml) bundles learning-rate/batch-size, stratification/class-weight toggles, XAI knobs, and the `keep_local_copy` flag (default `false` to avoid duplicate models when MLflow autolog succeeds).
- [conf/dataset.yaml](conf/dataset.yaml) captures the standard split (`train_ratio=0.7`, `val_ratio=0.1`) that `split_dataset` honors but allows overrides for experimental splits.
- All CLI scripts are exposed via [pyproject.toml](pyproject.toml) entry points: `finetune-bert`, `build-graph`, and `train-bertgcn` to ensure Hydra, Typer, and Python paths are resolved consistently.
- `build_graph` selects a BERT checkpoint from [bertgcn/config.py](bertgcn/config.py) (the default is `medbert`) and enforces adjacency symmetry/connections before dumping the sparse matrices; if you change the data source, keep the `CleanClinicDataset` contract (`LE`, `ohe_labels`, `texts`).
- `preprocess` pulls German stopwords via `nltk.download` and tokenizes again with the same `model_name_or_path` as the finetuner to keep vocab alignment. Regenerate both tokenizer+dataset if you update text cleaning.

## Testing & Stability
- `tests/test_hydra_run_dir_behavior.py` ensures Hydra runs create their directories, checkpoint files stay inside the run dir, and no stray model files appear when `hparams.keep_local_copy=false`.
- `tests/test_build_graph*.py` cover the PMI/TF-IDF builders; any change to `GraphBuilder` should keep the assertions about symmetry, connectivity, and saved `ind.*` files passing.
- Keep Hydra output under version control (only configs) by acknowledging that everything under `hydra/`, `mlruns/`, and `outputs/` is ephemeral.

Please flag any section above that is unclear or missing so I can iterate further.