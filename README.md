Adapted from: https://github.com/ZeroRin/BertGCN

# BertGCN

## Project Structure

- `src/bertgcn/` - Main package code
- `conf/` - Hydra configuration files
- `tests/` - Unit tests

## Running Fine-tuning with Hydra

To run the fine-tuning pipeline with Hydra:

```bash
poetry run python -m bertgcn.finetune_bert
```

Or using the CLI entry point (if installed as a package):

```bash
poetry run finetune-bert
```

### Script Overview

`src/bertgcn/finetune_bert.py` fine‑tunes a Hugging Face Transformer with:

- Hydra configuration (structured + overridable from CLI)
- Stratified (optional) train/val/test splitting
- Class weighting (optional)
- Early stopping (optional; requires patience > 0)
- MLflow tracking & autologging (Transformers → PyTorch fallback)
- Confusion matrix artifact logging

Hydra manages the *run working directory*; all ephemeral training outputs
(checkpoints, confusion matrix JSON) live under `hydra.run.dir`. MLflow is used
as the canonical artifact & metric store. Local model copies are only saved if
`hparams.keep_local_copy=true` OR all MLflow model logging fallbacks fail.

### Hydra Basics

Hydra composes `conf/config.yaml` + defaults (see the `defaults:` block). Key paths:

- Single run outputs: `hydra.run.dir` (default: `hydra/finetune/<timestamp>`)
- Multirun / sweeps: `hydra.sweep.dir` (default: `multirun/<timestamp>`)

Override any config value from the CLI. Examples:

```bash
# Change learning rate
poetry run finetune-bert hparams.learning_rate=3e-5

# Change batch size and disable stratified split
poetry run finetune-bert hparams.batch_size=16 hparams.use_stratified_split=false

# Set a custom run directory (relative path)
poetry run finetune-bert hydra.run.dir=outputs/custom_run

# Launch a simple sweep (Optuna via hydra-optuna-sweeper)
poetry run finetune-bert -m hparams.learning_rate='choice(1e-5,3e-5)' hparams.batch_size='choice(8,16)'
```

### MLflow Integration

Behavior summary:

1. Tracking URI:
	 - If already set externally (env / prior call), it's respected.
	 - Otherwise defaults to `<project_root>/mlruns`.
2. Autologging tries `mlflow.transformers.autolog()`, falling back to `mlflow.pytorch.autolog()`.
3. If autolog is active, manual param & metric logging is skipped (avoids duplicates).
4. Model logging preference order when autolog disabled:
	 - `mlflow.transformers.log_model`
	 - Fallback `mlflow.pytorch.log_model`
	 - Fallback local save (only if `hparams.keep_local_copy=true` or both MLflow attempts fail).
5. Confusion matrix stored at `<hydra.run.dir>/confusion_matrix.json` and logged once to MLflow under `evaluation/`.

Launch MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

### Key Hyperparameters (from `hparams.yaml`)

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Hugging Face model checkpoint name or path |
| `learning_rate` | Optimizer LR |
| `batch_size` | Per-device batch size (train & eval) |
| `num_train_epochs` | Number of epochs |
| `weight_decay` | Weight decay factor |
| `warmup_ratio` | Warmup steps ratio for scheduler |
| `use_stratified_split` | Enable stratified train/val/test split |
| `use_class_weights` | Enable balanced class weighting in loss |
| `early_stopping_patience` | >0 enables early stopping callback |
| `keep_local_copy` | If true, keep local model/tokenizer even when MLflow logs model |

### Artifact Layout

Single run (example):

```
hydra/finetune/2025-08-28_12-34-56/
	.hydra/                # Hydra metadata (config + overrides + hydra.yaml)
	checkpoint-1/          # Trainer checkpoint(s)
	confusion_matrix.json  # Matrix + label list
```

MLflow experiment directory (`mlruns/<exp_id>/<run_id>/`):

```
artifacts/
	model/                 # Logged model (transformers or pytorch flavor)
	evaluation/confusion_matrix.json
meta.yaml
metrics/params/tags
```

### Preventing Redundant Outputs

- Hydra run dir: contains only transient training artifacts & confusion matrix.
- MLflow: authoritative for model versioning & metrics.
- Set `hparams.keep_local_copy=false` (default) to avoid duplicate local saves.
- For space-constrained environments, you can also trim checkpoints via
	`save_total_limit` (already set to 2) or disable checkpointing entirely by
	overriding (depends on Transformers version):

```bash
poetry run finetune-bert training_args.save_strategy='no'
```

### Reproducing a Run

Each Hydra run stores the exact composed config under `.hydra/`:

```
hydra/finetune/<ts>/.hydra/config.yaml
```

You can replay with:

```bash
poetry run finetune-bert --config-path hydra/finetune/<ts>/.hydra --config-name config
```

Or with MLflow params (if autolog captured them):

```bash
mlflow runs describe <run_id>
```

### Common Overrides & Examples

```bash
# Enable class weights + early stopping
poetry run finetune-bert hparams.use_class_weights=true hparams.early_stopping_patience=3

# Store a local model copy even with successful MLflow logging
poetry run finetune-bert hparams.keep_local_copy=true

# Custom experiment name
poetry run finetune-bert mlflow_experiment_name=my_exp
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| ValueError about save/eval strategy | Older / variant Transformers arg filtering | We auto-align strategies; upgrade `transformers` if persists |
| EarlyStopping assertion (eval_strategy NO) | Missing eval strategy after filtering | Automatically patched; ensure patience > 0 only when wanted |
| Duplicate model artifacts | Manual save + autolog | Leave `keep_local_copy=false`; rely on MLflow model |
| No model in MLflow run | Autolog + logging failure fallback triggered | Enable `hparams.keep_local_copy=true` for local copy |

### Minimal Programmatic Invocation

```python
from omegaconf import OmegaConf
from bertgcn.finetune_bert import main

cfg = OmegaConf.create({
	'hparams': {
		'seed': 42,
		'model_name_or_path': 'bert-base-uncased',
		'learning_rate': 2e-5,
		'batch_size': 8,
		'num_train_epochs': 1,
		'weight_decay': 0.01,
		'warmup_ratio': 0.0,
		'eval_steps': 50,
		'use_stratified_split': False,
		'use_class_weights': False,
	},
	'mlflow_experiment_name': 'manual_invocation',
	'hydra': {'run': {'dir': 'hydra/finetune/manual_example'}}
})

# Call the undecorated function to bypass Hydra CLI wrapper
main.__wrapped__(cfg)
```

### CI Considerations

- Tests include a dry-run and hydra run-dir behavior checks.
- For faster CI, you can set `hparams.num_train_epochs=0.1` (some HF versions allow fractional epochs) or drastically reduce dataset size via preprocessing.
- Pin `pydantic <2.0` (already in `pyproject.toml`) to avoid deprecation noise until upstream libs migrate.

### MLflow Model Registry (Optional)

To register models automatically, set `mlflow_model_name` (either at root or under `hparams`) and ensure your tracking server has registry support:

```bash
poetry run finetune-bert mlflow_model_name=bertgcn_finetuned
```

### Cleaning Up Artifacts

Remove older Hydra run directories & stale checkpoints:

```bash
find hydra/finetune -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
```

Prune MLflow runs by age or status using the MLflow UI or CLI:

```bash
mlflow gc --backend-store-uri mlruns
```

---

For deeper graph + BERT joint training (GCN integration), see future documentation TODO.

## Configuration

Edit configs in the `conf/` directory. The main config is `config.yaml`, which composes from `main.yaml`, `hparams.yaml`, and `dataset.yaml`.

## Project Installation

Install dependencies and set up the project:

```bash
poetry install
```

## Testing

Run tests with:

```bash
poetry run pytest
```