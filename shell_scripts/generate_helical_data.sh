#!/usr/bin/env bash

python scripts/make_hits_helical.py \
  --min-r0 1.0 \
  --max-r0 10.0 \
  --nlayers 10 \
  --sigma 0.01 \
  --seed 1 \
  --num-tracks 1000000 \
  --noise gaussian \
  --out data/helical/1M_gaussian.txt \

