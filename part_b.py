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


def one_hot(y, num_classes):
    """y: (m,) int labels -> (m, num_classes) one-hot matrix."""
    return np.eye(num_classes)[y]


def iterate_batches(n, batch_size, rng=None):
    """
    Yield index arrays that partition range(n) into consecutive batches of
    `batch_size` (the last batch may be smaller).

    If `rng` is given, indices are freshly shuffled first (rng.permutation(n))
    before slicing. If `rng` is None, the original order 0..n-1 is used.
    """
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def format_row(values):
    """Comma-separated values, each with enough precision to round-trip a float64."""
    return ",".join(f"{v:.17g}" for v in values)


def write_weights(path, W, b):
    """weights.txt: line 1 = bias (3 values), lines 2-79 = rows of W (3 values each)."""
    with open(path, "w") as f:
        f.write(format_row(b) + "\n")
        for row in W:
            f.write(format_row(row) + "\n")


def write_predictions(path, P):
    """predictions.txt: one line per row, p(N),p(A),p(O), in P's row order."""
    with open(path, "w") as f:
        for row in P:
            f.write(format_row(row) + "\n")


def compute_class_alpha(y, num_classes=3):
    """
    Inverse-frequency class weight per class: alpha_k = n / (3 * n_k).

    Returns:
        alpha_per_class: (num_classes,) ndarray, alpha_per_class[k] = alpha_k.
    """
    n = len(y)
    counts = np.bincount(y, minlength=num_classes)   # n_k for each class
    return n / (3.0 * counts)


def weighted_cross_entropy_loss(X, y, W, b, alpha_per_example):
    """
    Global weighted mean cross-entropy: sum_i(alpha_i * -log p_i) / sum_i(alpha_i).
    alpha_per_example: (n,) weight per row, ALREADY selected for each row's class
    (e.g. alpha_per_class[y] or alpha_per_class[y] ** p). alpha_i == 1 for every
    row reduces this exactly to Part (a)'s plain mean cross-entropy.
    """
    n = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    true_class_probs = P[np.arange(n), y]
    per_example_losses = -np.log(true_class_probs)
    return np.sum(alpha_per_example * per_example_losses) / np.sum(alpha_per_example)


def compute_gradients_weighted(X, y, W, b, alpha_per_example):
    """
    Weighted-loss gradient for a batch (X, y).

    alpha_per_example: (m,) per-row weight, NOT yet normalized -- this function
    normalizes it WITHIN the batch (weights summing to 1 over this batch), per
    spec, which replaces Part (a)'s plain 1/|B| division.
    """
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    Y = one_hot(y, num_classes=W.shape[1])
    E = P - Y                                          # (m,3)

    batch_weights = alpha_per_example / alpha_per_example.sum()   # (m,) sums to 1
    E_weighted = E * batch_weights[:, None]            # broadcast (m,) -> (m,1) * (m,3)

    grad_W = X.T @ E_weighted                          # (78,3)
    grad_b = E_weighted.sum(axis=0)                    # (3,)

    return grad_W, grad_b


def train_weighted_adagrad(X, y, alpha_full, alpha_power, epochs=200, eta=0.3,
                            batch_size=32, eps=1e-8, seed=774):
    """
    Mini-batch AdaGrad (Part (a)'s Method 4) with the weighted gradient above.

    alpha_full: (3,) alpha_per_class from compute_class_alpha, computed ONCE
        from the full training set's class counts.
    alpha_power: exponent applied to alpha_full before use (1.0 for Baseline/
        Classweight, 0.3 for Classweight2; Baseline additionally passes an
        all-ones alpha_full so alpha_i == 1 regardless of alpha_power).
    """
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    G_W = np.zeros_like(W)
    G_b = np.zeros_like(b)
    rng = np.random.default_rng(seed)

    alpha_per_class = alpha_full ** alpha_power         # (3,)
    alpha_per_example_full = alpha_per_class[y]         # (n,) - for epoch-level loss logging
    losses = []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            alpha_B = alpha_per_class[y_B]               # (m,) this batch's per-row weights
            grad_W, grad_b = compute_gradients_weighted(X_B, y_B, W, b, alpha_B)
            G_W += grad_W * grad_W
            G_b += grad_b * grad_b
            W -= eta * grad_W / (np.sqrt(G_W) + eps)
            b -= eta * grad_b / (np.sqrt(G_b) + eps)
        losses.append(weighted_cross_entropy_loss(X, y, W, b, alpha_per_example_full))

    return W, b, losses


def focal_loss(X, y, W, b, alpha_prime_per_class, gamma=2.0):
    """Mean focal loss over all rows of X, per the Part (b) focal-loss formula."""
    n = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    p_t = P[np.arange(n), y]
    p_t_clipped = np.clip(p_t, 1e-12, 1 - 1e-12)   # only for the log() call below
    alpha_prime_t = alpha_prime_per_class[y]
    per_example = alpha_prime_t * (1 - p_t) ** gamma * (-np.log(p_t_clipped))
    return np.mean(per_example)


def compute_gradients_focal(X, y, W, b, alpha_prime_per_class, gamma=2.0):
    """Focal-loss gradient for a batch (X, y), averaged (mean) over the batch."""
    m = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    Y = one_hot(y, num_classes=W.shape[1])
    E = P - Y                                       # (m,3) - same error term as Part (a)

    p_t = P[np.arange(m), y]
    p_t_clipped = np.clip(p_t, 1e-12, 1 - 1e-12)
    log_pt = np.log(p_t_clipped)
    alpha_prime_t = alpha_prime_per_class[y]

    factor = alpha_prime_t * (1 - p_t) ** (gamma - 1) * ((1 - p_t) - gamma * p_t * log_pt)  # (m,)

    dLdZ = factor[:, None] * E                      # (m,3) - E scaled per-example

    grad_W = X.T @ dLdZ / m
    grad_b = dLdZ.sum(axis=0) / m

    return grad_W, grad_b


def train_focal_adagrad(X, y, alpha_prime_per_class, gamma=2.0, epochs=200, eta=0.3,
                         batch_size=32, eps=1e-8, seed=774):
    """Mini-batch AdaGrad (Part (a)'s Method 4), with the focal-loss gradient above."""
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    G_W = np.zeros_like(W)
    G_b = np.zeros_like(b)
    rng = np.random.default_rng(seed)
    losses = []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients_focal(X_B, y_B, W, b, alpha_prime_per_class, gamma)
            G_W += grad_W * grad_W
            G_b += grad_b * grad_b
            W -= eta * grad_W / (np.sqrt(G_W) + eps)
            b -= eta * grad_b / (np.sqrt(G_b) + eps)
        losses.append(focal_loss(X, y, W, b, alpha_prime_per_class, gamma))

    return W, b, losses


def main():
    if len(sys.argv) != 6:
        print(
            "usage: python3 part_b.py train.csv test.csv "
            "{baseline,classweight,classweight2,focal} predictions.txt weights.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path, test_path, method, predictions_path, weights_path = sys.argv[1:]

    X_train_raw, y_train, feature_cols = load_xy(train_path, has_label=True)
    X_test_raw, _, _ = load_xy(test_path, feature_cols=feature_cols, has_label=False)

    mean, std = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, mean, std)
    X_test = standardize_apply(X_test_raw, mean, std)

    if method == "baseline":
        W, b, _losses = train_weighted_adagrad(X_train, y_train, np.ones(3), alpha_power=1.0)
    elif method == "classweight":
        alpha_full = compute_class_alpha(y_train)
        W, b, _losses = train_weighted_adagrad(X_train, y_train, alpha_full, alpha_power=1.0)
    elif method == "classweight2":
        alpha_full = compute_class_alpha(y_train)
        W, b, _losses = train_weighted_adagrad(X_train, y_train, alpha_full, alpha_power=0.3)
    elif method == "focal":
        alpha_prime = compute_class_alpha(y_train) ** 0.5
        W, b, _losses = train_focal_adagrad(X_train, y_train, alpha_prime, gamma=2.0)
    else:
        print(
            f"unknown method '{method}', expected one of "
            "{baseline, classweight, classweight2, focal}",
            file=sys.stderr,
        )
        sys.exit(1)

    P_test = softmax(compute_logits(X_test, W, b))

    write_weights(weights_path, W, b)
    write_predictions(predictions_path, P_test)


if __name__ == "__main__":
    main()
