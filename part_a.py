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
    std[std == 0] = 1.0    # constant columns (e.g. signal_mad) would divide 0/0 -> nan;
                            # every value in such a column equals the mean, so (x-mean)=0
                            # regardless of the divisor -- replacing 0 with 1 keeps that 0
                            # without introducing nan.
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


def iterate_batches(n, batch_size, rng=None):
    """
    Yield index arrays that partition range(n) into consecutive batches of
    `batch_size` (the last batch may be smaller).

    If `rng` is given, indices are freshly shuffled first (rng.permutation(n))
    before slicing. If `rng` is None, the original order 0..n-1 is used
    unshuffled -- this is what full-batch GD requires ("no shuffling occurs").
    """
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def train_full_batch(X, y, epochs=500, eta=0.3):
    """Method 1: one gradient step per epoch, over the entire training set."""
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    losses = []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=n, rng=None):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            W -= eta * grad_W
            b -= eta * grad_b
        losses.append(cross_entropy_loss(X, y, W, b))

    return W, b, losses


def train_mini_batch(X, y, epochs=200, eta=0.03, batch_size=32, seed=774):
    """Method 2 (and Method 3, SGD, via batch_size=1): reshuffle every epoch,
    one gradient step per consecutive chunk of `batch_size` indices."""
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            W -= eta * grad_W
            b -= eta * grad_b
        losses.append(cross_entropy_loss(X, y, W, b))

    return W, b, losses


def train_sgd(X, y, epochs=30, eta=0.001, seed=774):
    """Method 3: mini-batch GD with batch size 1."""
    return train_mini_batch(X, y, epochs=epochs, eta=eta, batch_size=1, seed=seed)


def train_adagrad(X, y, epochs=200, eta=0.3, batch_size=32, eps=1e-8, seed=774):
    """Method 4: mini-batch GD with a per-parameter adaptive learning rate."""
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    G_W = np.zeros_like(W)         # per-parameter accumulated squared gradients
    G_b = np.zeros_like(b)
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            G_W += grad_W * grad_W
            G_b += grad_b * grad_b
            W -= eta * grad_W / (np.sqrt(G_W) + eps)
            b -= eta * grad_b / (np.sqrt(G_b) + eps)
        losses.append(cross_entropy_loss(X, y, W, b))

    return W, b, losses


METHODS = {
    "full_batch": lambda X, y: train_full_batch(X, y, epochs=500, eta=0.3),
    "mini_batch": lambda X, y: train_mini_batch(X, y, epochs=200, eta=0.03, batch_size=32),
    "sgd": lambda X, y: train_sgd(X, y, epochs=30, eta=0.001),
    "adagrad": lambda X, y: train_adagrad(X, y, epochs=200, eta=0.3, batch_size=32, eps=1e-8),
}


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
