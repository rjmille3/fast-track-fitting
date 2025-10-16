#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# --------------------------
# Constants
# --------------------------
PARAM_LABELS = ['d0', 'sin', 'cos', 'ptinv', 'dz', 'tanl']

# --------------------------
# Inverted-Bottleneck Block
# --------------------------
class ExpansionBlock(Layer):
    def __init__(self, hidden_dim, expansion=5, dropout=0.0,
                 kernel_initializer='he_normal', kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.expansion = expansion
        self.dropout_rate = dropout
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_initializer_config = kernel_initializer  # for serialization
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_regularizer_config = kernel_regularizer  # for serialization

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "expansion": self.expansion,
            "dropout": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer_config,
            "kernel_regularizer": self.kernel_regularizer_config
        })
        return config

# --------------------------
# Utilities
# --------------------------
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def linear_warmup_scheduler(base_lr, warmup_epochs):
    def scheduler(epoch, lr):
        return base_lr * (epoch + 1) / warmup_epochs if epoch < warmup_epochs else base_lr
    return scheduler

def save_hyperparams(path, cfg_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path.replace('.h5', '_config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)

class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, interval=10):
        super().__init__()
        self.save_path = save_path
        self.interval = interval
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if (epoch + 1) % self.interval == 0:
            self.plot_loss(epoch + 1)

    def plot_loss(self, epoch):
        plt.figure()
        plt.yscale('log')
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Train Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training vs. Validation Loss (Epoch {epoch})')
        plt.legend()
        plt.savefig(self.save_path.replace('.h5', f'_loss_curve_partial.png'))
        plt.close()

# --------------------------
# Model builder
# --------------------------
def build_encoder(input_dim, layer_config, expansion, dropout, l2_reg):
    init = HeNormal()
    reg = l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in layer_config:
        x = ExpansionBlock(hidden_dim=units,
                           expansion=expansion,
                           dropout=dropout,
                           kernel_initializer=init,
                           kernel_regularizer=reg)(x)
    outputs = Dense(1, kernel_initializer=init, kernel_regularizer=reg)(x)
    return tf.keras.Model(inputs, outputs)

# --------------------------
# Training Function
# --------------------------
def train_single_param(args):
    assert args.param_to_train in PARAM_LABELS, f"--param_to_train must be one of {PARAM_LABELS}"
    param_idx = PARAM_LABELS.index(args.param_to_train)

    # Optional: multi-GPU
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    strategy = tf.distribute.MirroredStrategy()

    # Paths
    train_path = os.path.join(args.data_dir, 'train.npz')
    val_path   = os.path.join(args.data_dir, 'val.npz')
    assert os.path.isfile(train_path), f"Missing {train_path}"
    assert os.path.isfile(val_path), f"Missing {val_path}"

    # Load data
    tr = np.load(train_path)
    va = np.load(val_path)
    X_train, y_train = tr['X_train'], tr['y_train']
    X_val,   y_val   = va['X_val'], va['y_val']

    # Convert pt -> 1/pt
    y_train = y_train.copy()
    y_val = y_val.copy()
    y_train[:, 3] = 1.0 / y_train[:, 3]
    y_val[:, 3] = 1.0 / y_val[:, 3]

    # Scale all targets, then slice the one to learn
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)

    y_train_param = y_train_scaled[:, param_idx:param_idx+1]
    y_val_param = y_val_scaled[:, param_idx:param_idx+1]

    # Datasets
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train_param))
                .shuffle(10000)
                .batch(args.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val_param))
              .batch(args.batch_size)
              .prefetch(tf.data.AUTOTUNE))

    # Save path(s)
    param_subdir = os.path.join(args.save_dir, f'param_{args.param_to_train}')
    os.makedirs(param_subdir, exist_ok=True)
    save_path = os.path.join(param_subdir, f'{args.param_to_train}_model.h5')

    # Build + compile
    with strategy.scope():
        model = build_encoder(
            input_dim=X_train.shape[1],
            layer_config=args.layer_config,
            expansion=args.expansion,
            dropout=args.dropout,
            l2_reg=args.l2_reg
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss=custom_loss)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True),
        LossPlotCallback(save_path, interval=args.plot_interval),
        tf.keras.callbacks.LearningRateScheduler(
            schedule=linear_warmup_scheduler(args.learning_rate, args.warmup_epochs),
            verbose=0
        ),
    ]

    # Train
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks)

    # Final loss plot
    plt.figure()
    plt.yscale('log')
    plt.plot(history.history['loss'][1:], label='Train Loss')
    plt.plot(history.history['val_loss'][1:], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Final Loss Curve ({args.param_to_train})')
    plt.legend()
    plt.savefig(save_path.replace('.h5', '_loss_curve.png'))
    plt.close()

    # Save hyperparams/config
    cfg_out = {
        "param_index": PARAM_LABELS.index(args.param_to_train),
        "param_name": args.param_to_train,
        "LAYER_CONFIG": args.layer_config,
        "EXPANSION": args.expansion,
        "DROPOUT": args.dropout,
        "LEARNING_RATE": args.learning_rate,
        "WARMUP_EPOCHS": args.warmup_epochs,
        "L2_REG": args.l2_reg,
        "BATCH_SIZE": args.batch_size,
        "EPOCHS": args.epochs,
        "PATIENCE": args.patience,
        "DATA_DIR": args.data_dir,
        "SAVE_DIR": args.save_dir,
    }
    save_hyperparams(save_path, cfg_out)

# --------------------------
# Argparse
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train helical parameter models")

    # Required dirs
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    # Parameters to train (comma-separated list)
    p.add_argument("--params_to_train", type=str, default="dz",
                   help="Comma-separated list of target parameters to predict, e.g. 'd0,sin,cos,ptinv,dz,tanl'")

    # Model / training config
    p.add_argument("--layer_config", type=str, default="500,500,500")
    p.add_argument("--expansion", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=12)
    p.add_argument("--l2_reg", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=1000)
    p.add_argument("--plot_interval", type=int, default=10)

    args = p.parse_args()

    args.layer_config = [int(x) for x in args.layer_config.split(",") if x.strip()]
    args.params_to_train = [p.strip() for p in args.params_to_train.split(",") if p.strip()]
    return args


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    args = parse_args()
    for param in args.params_to_train:
        args.param_to_train = param
        print(f"\n=== Training parameter: {param} ===")
        train_single_param(args)