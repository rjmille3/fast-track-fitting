#!/usr/bin/env bash

#ls-realistic-path and ls-idealized-path correspond to the paths of the LS predictions

python scripts/eval_models.py \
  --model-dir "models/gaussian/models" \
  --train-path "data/helical/gaussian/raw/train.txt" \
  --ls-realistic-path "data/helical/gaussian/ls/realistic/100000.txt" \
  --ls-idealized-path "data/helical/gaussian/ls/idealized/100000.txt" \
  --x-test-path "data/helical/gaussian/ls/realistic/X_vals.txt"
