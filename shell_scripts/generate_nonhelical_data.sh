#!/usr/bin/env bash

python scripts/make_hits_nonhelical.py \
  --min-r0 1.0 \
  --max-r0 10.0 \
  --nlayers 10 \
  --sigma 0.01 \
  --seed 1 \
  --n-tracks 100000 \
  --noise gaussian \
  --out data/nonhelical/100k_gaussian_nonhelical.txt \

