#noise for dz and tanl parameters was adjusted so that the LS fitter would actually run (with too small noise, the optimizer would "converge" before running). To increase the noise, we used the std from the NN model predictions.

python scripts/idealized_ls_fit.py \
  --input "data/helical/gaussian/raw/test.txt" \
  --outdir "data/helical/gaussian/ls/idealized/final" \
  -n 100000 \
  --noise_stds 1e-6 1e-6 1e-6 0.00715 0.00122
