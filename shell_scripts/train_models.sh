#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}

DATA_DIR="data/helical/testing/gaussian/processed"
SAVE_DIR="models/testing/gaussian/models"

#train all params
PARAMS="d0,sin,cos,ptinv,dz,tanl"

$PY scripts/train_models.py \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --params_to_train "$PARAMS" \
  --layer_config "500,500,500" \
  --expansion 5 \
  --dropout 0.0 \
  --learning_rate 1e-4 \
  --warmup_epochs 12 \
  --l2_reg 0.0 \
  --batch_size 4096 \
  --epochs 1000 \
  --patience 1000 \
  --plot_interval 10
