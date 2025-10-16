# Fast-Track Fitting

End-to-end pipeline for generating datasets, preprocessing, training neural models, and running least-squares (LS) fits for helical and non-helical tracks.

---

## 1) Environment Setup

    # Create and activate environment
    conda create -n fast_track_fitting python=3.7.12
    conda activate fast_track_fitting

    # CUDA and cuDNN (conda-forge)
    conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1

    # Core deps (pin versions as needed)
    conda install -y numpy=1.21.6
    pip install tensorflow-gpu==2.10.1
    conda install -y scikit-learn=1.0.2 --no-update-deps
    conda install -y matplotlib=3.5.3 --no-update-deps
    conda install -y -c conda-forge tqdm=4.67.1 --no-update-deps

    # Ensure runtime can find CUDA/cuDNN from this env
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

---

## 2) Dataset Generation

Generate helical **or** non-helical datasets:

    # Helical
    shell_scripts/generate_helical_data.sh

    # Non-helical
    shell_scripts/generate_nonhelical_data.sh

---

## 3) Split Datasets

Create train/val/test splits:

    shell_scripts/split_data.sh

---

## 4) Preprocess to `.npz`

Convert the splits to compressed `.npz` files:

    shell_scripts/preprocess_data.sh

> **Important:** Training expects **train/val/test** files in `.npz` format.

---

## 5) Train Models

    shell_scripts/train_models.sh

---

## 6) Run Least-Squares Fits

- **Realistic LS fit:**

      shell_scripts/run_realistic_ls_fit.sh

- **Idealized LS fit (Gaussian or Skewed):**

      shell_scripts/run_idealized_ls_fit_gaussian.sh
      shell_scripts/run_idealized_ls_fit_skewed.sh

---

## 7) Evaluate (NN & LS)

    shell_scripts/eval.sh

---
