#!/usr/bin/env python3
import argparse
import os
import numpy as np


def load_raw(filepath: str, n_hits: int = 10):
    """
    Reads raw .txt file with blocks separated by blank/other lines:
      - lines with 5 values → targets [d0, phi, 1/pt, dz, tanl]
      - lines with 3 values → hits [x, y, z] (expect n_hits lines per block)
      - any other / blank line → signals end of one track

    Returns:
      X: (n_tracks, 3*n_hits)  flattened hit coordinates
      y: (n_tracks, 7)         [d0, sin(phi), cos(phi), 1/pt, dz, tanl, phi]
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    datapoints, tgts, features = [], [], []
    newdata = False

    for line in lines:
        # robust split: allow ", " or ","
        parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]

        # if we hit a separator after completing hits, close out the block
        if newdata and len(parts) >= 3 and len(features) == n_hits:
            datapoints.append(features)
            features = []
            newdata = False

        if len(parts) == 5:          # target line
            tgts.append([float(v) for v in parts])
        elif len(parts) == 3:        # hit line
            features.append([float(v) for v in parts])
        else:                        # separator / anything else
            newdata = True

    # flush last block if present
    if features:
        datapoints.append(features)

    # to arrays
    datapoints = np.array(datapoints, dtype=float)               # (N, n_hits, 3)
    if datapoints.ndim != 3 or datapoints.shape[1] != n_hits or datapoints.shape[2] != 3:
        raise ValueError(
            f"Parsed hits have shape {datapoints.shape}, expected (N, {n_hits}, 3). "
            f"Check the input format."
        )
    X = datapoints.reshape(len(datapoints), 3 * n_hits)          # (N, 3*n_hits)

    tgts = np.array(tgts, dtype=float)                           # (N, 5)
    if tgts.ndim != 2 or tgts.shape[1] != 5:
        raise ValueError(
            f"Parsed targets have shape {tgts.shape}, expected (N, 5). "
            f"Check the input format."
        )

    # targets layout in file: [d0, phi, 1/pt, dz, tanl]
    phi = tgts[:, 1]
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # new target layout: [d0, sin(phi), cos(phi), 1/pt, dz, tanl, phi]
    y = np.hstack([
        tgts[:, :1],            # d0
        sin_phi[:, None],       # sin(phi)
        cos_phi[:, None],       # cos(phi)
        tgts[:, 2:],            # [1/pt, dz, tanl]
        phi[:, None],           # original phi
    ])

    # sanity: N matches
    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f"Mismatch in counts: X {X.shape[0]} vs y {y.shape[0]}")

    return X, y


def save_npz(out_path: str, **arrays):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **arrays)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw helix split .txt files to processed .npz with flattened hits and augmented targets."
    )
    p.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing train.txt, val.txt, test.txt."
    )
    p.add_argument(
        "--processed-dir",
        required=True,
        help="Output directory for train.npz, val.npz, test.npz."
    )
    p.add_argument(
        "--train-file", default="train.txt",
        help="Train split filename in raw-dir (default: train.txt)."
    )
    p.add_argument(
        "--val-file", default="val.txt",
        help="Validation split filename in raw-dir (default: val.txt)."
    )
    p.add_argument(
        "--test-file", default="test.txt",
        help="Test split filename in raw-dir (default: test.txt)."
    )
    p.add_argument(
        "--n-hits", type=int, default=10,
        help="Number of hit lines per example (default: 10)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    raw_train = os.path.join(args.raw_dir, args.train_file)
    raw_val   = os.path.join(args.raw_dir, args.val_file)
    raw_test  = os.path.join(args.raw_dir, args.test_file)

    # Load
    X_train, y_train = load_raw(raw_train, n_hits=args.n_hits)
    X_val,   y_val   = load_raw(raw_val,   n_hits=args.n_hits)
    X_test,  y_test  = load_raw(raw_test,  n_hits=args.n_hits)

    # Save
    os.makedirs(args.processed_dir, exist_ok=True)
    save_npz(os.path.join(args.processed_dir, "train.npz"), X_train=X_train, y_train=y_train)
    save_npz(os.path.join(args.processed_dir, "val.npz"),   X_val=X_val,     y_val=y_val)
    save_npz(os.path.join(args.processed_dir, "test.npz"),  X_test=X_test,   y_test=y_test)

    # Logs
    print("Saved npz files to:", args.processed_dir)
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape,   y_val.shape)
    print("Test: ", X_test.shape,  y_test.shape)


if __name__ == "__main__":
    main()
