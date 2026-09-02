import sys

import numpy as np
import pandas as pd

FEATURE_EXCLUDE = {"release_id", "label"}


def get_feature_columns(df):
    """Return the 78 feature column names, in file order, excluding release_id/label."""
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def load_xy(path, feature_cols=None, has_label=True):
    """
    Read a part_ab_*.csv file.

    Returns:
        X: (n, 78) float64 ndarray of feature values, in `feature_cols` order
           (or file order if feature_cols is None).
        y: (n,) int ndarray of labels, or None if has_label is False.
        feature_cols: the feature column names used (so the caller can reuse
           the same order for other splits).
    """
    df = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64) if has_label else None

    return X, y, feature_cols


def standardize_fit(X_train):
    """
    Compute per-feature mean and std from the TRAINING set only.

    Returns:
        mean: (78,) ndarray
        std:  (78,) ndarray
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean, std


def standardize_apply(X, mean, std):
    """Apply a previously-fit (mean, std) transform to X. Never refit here."""
    return (X - mean) / std


def compute_logits(X, W, b):
    """X: (n,78), W: (78,3), b: (3,) -> Z: (n,3) raw per-class scores."""
    return X @ W + b


def softmax(Z):
    """
    Z: (n,3) logits -> (n,3) class probabilities (rows sum to 1).

    Numerically stable per spec: subtract the row-wise max logit, then clip
    the shifted logits to [-60, 0] before exponentiating.
    """
    row_max = Z.max(axis=1, keepdims=True)      # (n,1) - largest logit per row
    Z_shifted = Z - row_max                     # broadcast: (n,3) - (n,1) -> (n,3)
    Z_clipped = np.clip(Z_shifted, -60.0, 0.0)
    exp_Z = np.exp(Z_clipped)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss(X, y, W, b):
    """Mean cross-entropy loss over all rows of X (no regularization)."""
    n = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    true_class_probs = P[np.arange(n), y]        # (n,) - P[i, y[i]] for every i
    return -np.mean(np.log(true_class_probs))


def one_hot(y, num_classes):
    """y: (m,) int labels -> (m, num_classes) one-hot matrix."""
    return np.eye(num_classes)[y]


def compute_gradients(X, y, W, b):
    """
    Gradient of mean cross-entropy loss w.r.t. W and b, for the batch (X, y).

    Divides by the CURRENT batch size m = X.shape[0] (per spec), not the
    full training-set size n.
    """
    m = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    Y = one_hot(y, num_classes=W.shape[1])
    E = P - Y                      # (m,3) - error per example per class

    grad_W = (X.T @ E) / m         # (78,m) @ (m,3) -> (78,3), matches W's shape
    grad_b = E.sum(axis=0) / m     # (3,) - average error per class

    return grad_W, grad_b


def main():
    if len(sys.argv) != 6:
        print(
            "usage: python3 part_a.py train.csv test.csv "
            "{full_batch,mini_batch,sgd,adagrad} predictions.txt weights.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path, test_path, method, predictions_path, weights_path = sys.argv[1:]

    X_train_raw, y_train, feature_cols = load_xy(train_path, has_label=True)
    X_test_raw, _, _ = load_xy(test_path, feature_cols=feature_cols, has_label=False)

    mean, std = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, mean, std)
    X_test = standardize_apply(X_test_raw, mean, std)

    # TODO (later steps): train with the requested method, write predictions.txt
    # and weights.txt.
    raise NotImplementedError(f"method dispatch for '{method}' not implemented yet")


if __name__ == "__main__":
    main()
