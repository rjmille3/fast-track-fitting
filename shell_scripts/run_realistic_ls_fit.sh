#!/usr/bin/env bash

INPUT="data/helical/gaussian/raw/test.txt"
OUTDIR="data/helical/gaussian/ls/realistic"
N=100000
WORKERS=20

python scripts/ls_fit.py --input "$INPUT" --outdir "$OUTDIR" -n "$N" --workers "$WORKERS" \
  --print_every 100 --checkpoint_every 10000 --seed 0