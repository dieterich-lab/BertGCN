#!/bin/bash

#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=gcn_sweep_p1
#SBATCH --output=/home/pwiesenbach/BertGCN/slurm/train_gcn_sweep_phase1.log
#SBATCH --partition=gpu
#SBATCH --mem=50G

cd /home/pwiesenbach/BertGCN
source /beegfs/homes/pwiesenbach/BertGCN/.venv/bin/activate
export PYTHONPATH=/home/pwiesenbach/BertGCN:$PYTHONPATH

# Phase 1 sweep: hidden_dim × mix_factor × dropout with linear warmup scheduler
python -m bertgcn.train_gcn --multirun \
  mode=train_gcn \
  hydra/sweeper=basic \
  hydra.sweep.dir=outputs/train_gcn/sweeps \
  'hydra.sweep.subdir=${hydra.job.num}' \
  gcn.gcn_layers=3 \
  gcn.n_hidden=300,400,512 \
  gcn.mix_factor=0.5,0.65,0.8 \
  gcn.dropout=0.3,0.4 \
  model.n_hidden=\${gcn.n_hidden} \
  model.dropout=\${gcn.dropout} \
  training.lr=0.0005 \
  training.bert_lr=0.00002 \
  training.weight_decay=0.01 \
  training.epochs=70 \
  training.early_stopping_patience=3 \
  training.batch_size=48 \
  training.zero_word_features=true \
  training.scheduler_type=linear_warmup \
  training.grad_clip_enabled=true
