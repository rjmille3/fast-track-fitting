#!/usr/bin/env python3
import os

import argparse
import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation
from tensorflow.keras.initializers import HeNormal
from sklearn.preprocessing import StandardScaler

from utils import (
    parse_data,
    reduce_dim,
    parse_data_ls_realistic,
    parse_data_ls_idealized,
    invert_idx,
)

# ------------------------- Layers / Loss -------------------------
class ExpansionBlock(Layer):
    def __init__(self, hidden_dim, expansion=8, dropout=0.0,
                 kernel_initializer='he_normal', kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.expansion = expansion
        self.dropout_rate = dropout
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.expanded_dim = hidden_dim * expansion
        self.expand = Dense(self.expanded_dim, kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer)
        self.act = Activation('gelu')
        self.dropout1 = Dropout(self.dropout_rate)
        self.project = Dense(self.hidden_dim, kernel_initializer=self.kernel_initializer,
                             kernel_regularizer=self.kernel_regularizer)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.act(x)
        x = self.dropout1(x, training=training)
        x = self.project(x)
        x = self.dropout2(x, training=training)
        return x

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# ------------------------- Inference helpers -------------------------
def model_path_list(model_root: str):
    """
    Builds an ordered list of model paths:
      d0, sin(phi0), cos(phi0), 1/pt, dz, tanl
    """
    parts = [
        ("param_d0",     "d0_model.h5"),
        ("param_sin",    "sin_model.h5"),
        ("param_cos",    "cos_model.h5"),
        ("param_ptinv",  "ptinv_model.h5"),
        ("param_dz",     "dz_model.h5"),
        ("param_tanl",   "tanl_model.h5"),
    ]
    return [os.path.join(model_root, sub, fname) for sub, fname in parts]


def predict_with_ensemble(model_paths, x_test, scaler):
    predictions = []
    for path in model_paths:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'custom_loss': custom_loss, 'ExpansionBlock': ExpansionBlock, 'HeNormal': HeNormal}
        )
        pred = model.predict(x_test, batch_size=100000).squeeze()  # (N,)
        predictions.append(pred)

    combined = np.stack(predictions, axis=-1)  # (N, 6)
    dummy_column = np.ones((combined.shape[0], 1))
    combined_with_dummy = np.hstack((combined, dummy_column))      # (N, 7)
    descaled = scaler.inverse_transform(combined_with_dummy)       # inverse on 7 columns
    return descaled[:, :6]                                         # back to (N, 6)


def wrap_column1(residuals):
    """
    Given a (N, D) array of residuals, return a copy where
    only column 1 has been wrapped into [–π, +π].
    """
    res = residuals.copy()
    diff = res[:, 1]
    res[:, 1] = np.arctan2(np.sin(diff), np.cos(diff))
    return res


# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate NN ensemble vs LS (realistic/idealized).")
    parser.add_argument(
        "--model-dir",
        default="/baldig/chemistry2/ryan/physics_temp/fast-track-fitting/models/gaussian/models",
        help="Root directory containing param_* subdirectories with *.h5 models."
    )
    parser.add_argument(
        "--train-path",
        default="/baldig/chemistry2/ryan/physics_temp/fast-track-fitting/data/helical/gaussian/raw/train.txt",
        help="Path to raw training data text file."
    )
    parser.add_argument(
        "--ls-realistic-path",
        default="/baldig/chemistry2/ryan/physics_temp/fast-track-fitting/data/helical/gaussian/ls/realistic/origv2/100000.txt",
        help="Path to LS realistic results text file."
    )
    parser.add_argument(
        "--ls-idealized-path",
        default="/baldig/chemistry2/ryan/physics_temp/fast-track-fitting/data/helical/gaussian/ls/idealized/final/100000.txt",
        help="Path to LS idealized results text file."
    )
    parser.add_argument(
        "--x-test-path",
        default="/baldig/chemistry2/ryan/physics_temp/fast-track-fitting/data/helical/gaussian/ls/realistic/origv2/X_vals.txt",
        help="Path to X_test values used for LS comparisons."
    )
    args = parser.parse_args()

    # Load training data (for scaler fit)
    X_train, y_train = parse_data(args.train_path)
    y_train[:, 3] = 1.0 / y_train[:, 3]  # invert pt to 1/pt

    # Fit scaler on y with a dummy column to match historical behavior
    scaler = StandardScaler()
    dummy_col = np.ones((y_train.shape[0], 1))
    y_train_with_dummy = np.hstack((y_train, dummy_col))
    _ = scaler.fit_transform(y_train_with_dummy)

    # Load LS and test data
    ls_helical_chisq, ls_helical_preds = parse_data_ls_realistic(args.ls_realistic_path)
    X_test, y_test = parse_data(args.x_test_path)
    ls_idealized_helical_chisq, ls_idealized_helical_preds = parse_data_ls_idealized(args.ls_idealized_path)

    # Ensemble NN predictions
    mpaths = model_path_list(args.model_dir)
    nn_helical_preds = predict_with_ensemble(mpaths, X_test, scaler)
    nn_helical_preds = reduce_dim(nn_helical_preds)

    # Prepare ground truth and LS predictions (angle handling, index inversion)
    y_test_reduced = reduce_dim(y_test)
    y_test_reduced_inverted = invert_idx(y_test_reduced, 2)

    nn_helical_preds_inverted = nn_helical_preds
    ls_helical_preds_inverted = invert_idx(ls_helical_preds, 2)
    ls_idealized_helical_preds_inverted = invert_idx(ls_idealized_helical_preds, 2)

    # Residuals
    nn_res = y_test_reduced_inverted - nn_helical_preds_inverted
    ls_res = y_test_reduced_inverted - ls_helical_preds_inverted
    ls_idealized_res = y_test_reduced_inverted - ls_idealized_helical_preds_inverted

    # Wrap only column 1 (phi)
    nn_res_wrapped = wrap_column1(nn_res)
    ls_res_wrapped = wrap_column1(ls_res)
    ls_idealized_res_wrapped = wrap_column1(ls_idealized_res)

    # Metrics
    nn_mae_per_column = np.mean(np.abs(nn_res_wrapped), axis=0)
    nn_std_per_column = np.std(nn_res_wrapped, axis=0)

    ls_mae_per_column = np.mean(np.abs(ls_res_wrapped), axis=0)
    ls_std_per_column = np.std(ls_res_wrapped, axis=0)

    ls_idealized_mae_per_column = np.mean(np.abs(ls_idealized_res_wrapped), axis=0)
    ls_idealized_std_per_column = np.std(ls_idealized_res_wrapped, axis=0)

    # Output
    print("NN MAE per column:", nn_mae_per_column)
    print("NN Std per column:", nn_std_per_column)
    print("LS MAE per column:", ls_mae_per_column)
    print("LS Std per column:", ls_std_per_column)
    print("LS idealized MAE per column:", ls_idealized_mae_per_column)
    print("LS idealized Std per column:", ls_idealized_std_per_column)


if __name__ == "__main__":
    main()
