#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path: str, eot_token: str = "EOT", delim: str = ", "):
    """
    Load data from a text file where each example is separated by an EOT token.
    The first line of each example contains target parameters (floats),
    followed by multiple lines of 3D points (x, y, z).
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Split into data points and clean
    chunks = [dp.strip() for dp in content.split(eot_token) if dp.strip()]
    # lines -> rows of floats
    data_points = []
    for dp in chunks:
        rows = dp.split("\n")
        # drop empty lines inside a chunk
        rows = [r.strip() for r in rows if r.strip()]
        # parse rows: targets first row, then hits
        parsed = [[float(c) for c in row.split(delim)] for row in rows]
        data_points.append(parsed)

    return data_points


def save_split(X: np.ndarray, y: np.ndarray, filename: str):
    """
    Save a split to disk in the same text format:
    - First line: 5 target params (float, 1.8f) separated by ', '
    - Then one line per hit: x, y, z (float, 1.8f)
    - End with 'EOT' and a blank line
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for i in range(len(y)):
            params = tuple(y[i])
            xs = X[i][:, 0]
            ys = X[i][:, 1]
            zs = X[i][:, 2]
            f.write("%1.8f, %1.8f, %1.8f, %1.8f, %1.8f\n" % params)
            # hits
            for j in range(len(xs)):
                f.write("%1.8f, %1.8f, %1.8f\n" % (xs[j], ys[j], zs[j]))
            f.write("EOT\n\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Split helix dataset into train/val/test and write text files."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to the input TXT file (e.g., 1M.txt).",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Directory to write train.txt, val.txt, test.txt (e.g., /path/to/raw).",
    )
    p.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction for train split (default: 0.8).",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction for validation split (default: 0.1).",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction for test split (default: 0.1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and splitting (default: 42).",
    )
    p.add_argument(
        "--eot_token",
        type=str,
        default="EOT",
        help="End-of-example token in the input file (default: 'EOT').",
    )
    p.add_argument(
        "--delim",
        type=str,
        default=", ",
        help="Delimiter between float values in each row (default: ', ').",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate split fractions
    total = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_frac + val_frac + test_frac must equal 1.0, got {total:.6f}"
        )
    if any(frac < 0 for frac in (args.train_frac, args.val_frac, args.test_frac)):
        raise ValueError("Split fractions must be non-negative.")

    # Seed everything reproducibly
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data_points = load_data(args.input, eot_token=args.eot_token, delim=args.delim)

    # Extract targets and inputs
    targets = [dp[0] for dp in data_points]  # first row = target params
    inputs = [dp[1:] for dp in data_points]  # remaining rows = hits

    np_targets = np.array(targets, dtype=float)
    np_inputs = np.array(inputs, dtype=float)

    # Combine and shuffle deterministically
    combined = list(zip(np_targets, np_inputs))
    random.shuffle(combined)
    shuffled_targets, shuffled_inputs = zip(*combined)

    # First split: train vs temp
    temp_frac = args.val_frac + args.test_frac
    if temp_frac <= 0 or temp_frac >= 1:
        raise ValueError("val_frac + test_frac must be in (0, 1).")

    X_train, X_temp, y_train, y_temp = train_test_split(
        shuffled_inputs,
        shuffled_targets,
        test_size=temp_frac,
        random_state=args.seed,
        shuffle=True,
    )

    # Second split: val vs test within temp
    test_within_temp = args.test_frac / temp_frac
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_within_temp, random_state=args.seed, shuffle=True
    )

    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=float)
    X_val = np.array(X_val, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_val = np.array(y_val, dtype=float)
    y_test = np.array(y_test, dtype=float)

    # Sanity prints
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val   shape: {X_val.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val   shape: {y_val.shape}")
    print(f"y_test  shape: {y_test.shape}")

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    save_split(X_train, y_train, os.path.join(args.outdir, "train.txt"))
    save_split(X_val, y_val, os.path.join(args.outdir, "val.txt"))
    save_split(X_test, y_test, os.path.join(args.outdir, "test.txt"))

    print(f"Saved splits to: {args.outdir}")


if __name__ == "__main__":
    main()
