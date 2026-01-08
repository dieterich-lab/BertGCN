#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=interpret_gcn
#SBATCH --output=/home/pwiesenbach/BertGCN/slurm/interpret_gcn.log
#SBATCH --partition=gpu
#SBATCH --mem=50G

# Ensure we run from the repository root so relative paths resolve.
cd /home/pwiesenbach/BertGCN

# Activate virtual environment
source /beegfs/homes/pwiesenbach/BertGCN/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/pwiesenbach/BertGCN:$PYTHONPATH

# Run all document-level interpretation variants
python -m bertgcn.interpret_docs_neighbors
python -m bertgcn.interpret_docs_ig
python -m bertgcn.interpret_docs_shap
