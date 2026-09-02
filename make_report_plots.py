"""
Generates the Part (a) report plots: training loss and validation loss vs.
wall-clock time, all four methods on the same axes.

NOT part of the graded submission (only part_a.py/part_b.py/part_c.py are
submitted) -- this script exists purely to produce report_assets/*.png for
report.pdf. Paths are hard-coded to this repo's data/ layout since it never
runs on the autograder.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part_a import (
    load_xy, standardize_fit, standardize_apply,
    compute_gradients, cross_entropy_loss, iterate_batches,
)

DATA_DIR = "data"
OUT_DIR = "report_assets"


def train_full_batch_tracked(X, y, X_val, y_val, epochs, eta):
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    t0 = time.time()
    times, train_losses, val_losses = [], [], []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=n, rng=None):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            W -= eta * grad_W
            b -= eta * grad_b
        times.append(time.time() - t0)
        train_losses.append(cross_entropy_loss(X, y, W, b))
        val_losses.append(cross_entropy_loss(X_val, y_val, W, b))

    return times, train_losses, val_losses


def train_mini_batch_tracked(X, y, X_val, y_val, epochs, eta, batch_size, seed=774):
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    times, train_losses, val_losses = [], [], []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            W -= eta * grad_W
            b -= eta * grad_b
        times.append(time.time() - t0)
        train_losses.append(cross_entropy_loss(X, y, W, b))
        val_losses.append(cross_entropy_loss(X_val, y_val, W, b))

    return times, train_losses, val_losses


def train_adagrad_tracked(X, y, X_val, y_val, epochs, eta, batch_size, eps, seed=774):
    n, d = X.shape
    W = np.zeros((d, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    G_W = np.zeros_like(W)
    G_b = np.zeros_like(b)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    times, train_losses, val_losses = [], [], []

    for _ in range(epochs):
        for batch_idx in iterate_batches(n, batch_size=batch_size, rng=rng):
            X_B, y_B = X[batch_idx], y[batch_idx]
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            G_W += grad_W * grad_W
            G_b += grad_b * grad_b
            W -= eta * grad_W / (np.sqrt(G_W) + eps)
            b -= eta * grad_b / (np.sqrt(G_b) + eps)
        times.append(time.time() - t0)
        train_losses.append(cross_entropy_loss(X, y, W, b))
        val_losses.append(cross_entropy_loss(X_val, y_val, W, b))

    return times, train_losses, val_losses


def main():
    X_train_raw, y_train, feature_cols = load_xy(f"{DATA_DIR}/part_ab_train.csv", has_label=True)
    X_val_raw, y_val, _ = load_xy(f"{DATA_DIR}/part_ab_val.csv", feature_cols=feature_cols, has_label=True)

    mean, std = standardize_fit(X_train_raw)
    X_train = standardize_apply(X_train_raw, mean, std)
    X_val = standardize_apply(X_val_raw, mean, std)

    print("training full_batch...")
    results = {
        "full_batch": train_full_batch_tracked(X_train, y_train, X_val, y_val, epochs=500, eta=0.3),
    }
    print("training mini_batch...")
    results["mini_batch"] = train_mini_batch_tracked(
        X_train, y_train, X_val, y_val, epochs=200, eta=0.03, batch_size=32
    )
    print("training sgd...")
    results["sgd"] = train_mini_batch_tracked(
        X_train, y_train, X_val, y_val, epochs=30, eta=0.001, batch_size=1
    )
    print("training adagrad...")
    results["adagrad"] = train_adagrad_tracked(
        X_train, y_train, X_val, y_val, epochs=200, eta=0.3, batch_size=32, eps=1e-8
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (times, train_losses, _val_losses) in results.items():
        ax.plot(times, train_losses, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("training loss (mean cross-entropy)")
    ax.set_title("Part (a): training loss vs. time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/train_loss_vs_time.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (times, _train_losses, val_losses) in results.items():
        ax.plot(times, val_losses, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("validation loss (mean cross-entropy)")
    ax.set_title("Part (a): validation loss vs. time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/val_loss_vs_time.png", dpi=150)

    print(f"saved plots to {OUT_DIR}/train_loss_vs_time.png and {OUT_DIR}/val_loss_vs_time.png")


if __name__ == "__main__":
    main()
