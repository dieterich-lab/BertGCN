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