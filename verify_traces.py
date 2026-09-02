"""
Verifies part_a.py's and part_b.py's training loops against the reference
weight/loss snapshots in data/weight_traces/.

NOT part of the graded submission -- only part_a.py/part_b.py/part_c.py/
report.pdf go in the submission zip. This script exists so you can re-run
the correctness check yourself anytime, rather than trusting it blindly.

Usage:
    python3 verify_traces.py            # checks both part (a) and part (b)
    python3 verify_traces.py a          # only part (a)
    python3 verify_traces.py b          # only part (b)
"""

import sys

import numpy as np
import pandas as pd

import part_a
import part_b

DATA_DIR = "data"
TRACE_DIR = f"{DATA_DIR}/weight_traces"
NUM_EPOCHS_TO_CHECK = 5
WEIGHT_TOLERANCE = 1e-6      # max allowed |our_value - reference_value|
LOSS_TOLERANCE = 1e-6

LOSS_REF = pd.read_csv(f"{TRACE_DIR}/loss_by_epoch.csv")


def load_reference_weights(part, method, epoch):
    path = f"{TRACE_DIR}/{part}/{method}_epoch{epoch}.txt"
    lines = open(path).read().strip().split("\n")
    values = np.array([[float(x) for x in line.split(",")] for line in lines])
    return values[0], values[1:]   # b_ref (3,), W_ref (78,3)


def reference_loss(part, method, epoch):
    row = LOSS_REF[
        (LOSS_REF["part"] == part) & (LOSS_REF["method"] == method) & (LOSS_REF["epoch"] == epoch)
    ]
    return row["train_loss"].iloc[0]


def check_one(part, method, W, b, losses, epoch):
    b_ref, W_ref = load_reference_weights(part, method, epoch)
    w_diff = np.max(np.abs(W - W_ref))
    b_diff = np.max(np.abs(b - b_ref))
    loss_diff = abs(losses[-1] - reference_loss(part, method, epoch))

    passed = w_diff < WEIGHT_TOLERANCE and b_diff < WEIGHT_TOLERANCE and loss_diff < LOSS_TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] epoch {epoch}: max|dW|={w_diff:.2e} max|db|={b_diff:.2e} loss_diff={loss_diff:.2e}")
    return passed


def verify_part_a():
    print("=== Part (a) ===")
    X_raw, y, feature_cols = part_a.load_xy(f"{DATA_DIR}/part_ab_train.csv", has_label=True)
    mean, std = part_a.standardize_fit(X_raw)
    X = part_a.standardize_apply(X_raw, mean, std)

    configs = [
        ("full_batch", lambda e: part_a.train_full_batch(X, y, epochs=e, eta=0.3)),
        ("mini_batch", lambda e: part_a.train_mini_batch(X, y, epochs=e, eta=0.03, batch_size=32)),
        ("sgd", lambda e: part_a.train_sgd(X, y, epochs=e, eta=0.001)),
        ("adagrad", lambda e: part_a.train_adagrad(X, y, epochs=e, eta=0.3, batch_size=32, eps=1e-8)),
    ]

    all_passed = True
    for method_name, train_fn in configs:
        print(f"--- {method_name} ---")
        for epoch in range(1, NUM_EPOCHS_TO_CHECK + 1):
            W, b, losses = train_fn(epoch)
            all_passed &= check_one("part_a", method_name, W, b, losses, epoch)
    return all_passed


def verify_part_b():
    print("=== Part (b) ===")
    X_raw, y, feature_cols = part_b.load_xy(f"{DATA_DIR}/part_ab_train.csv", has_label=True)
    mean, std = part_b.standardize_fit(X_raw)
    X = part_b.standardize_apply(X_raw, mean, std)

    alpha_full = part_b.compute_class_alpha(y, num_classes=3)
    alpha_prime = alpha_full ** 0.5

    configs = [
        ("baseline", lambda e: part_b.train_weighted_adagrad(X, y, np.ones(3), 1.0, epochs=e)),
        ("classweight", lambda e: part_b.train_weighted_adagrad(X, y, alpha_full, 1.0, epochs=e)),
        ("classweight2", lambda e: part_b.train_weighted_adagrad(X, y, alpha_full, 0.3, epochs=e)),
        ("focal", lambda e: part_b.train_focal_adagrad(X, y, alpha_prime, gamma=2.0, epochs=e)),
    ]

    all_passed = True
    for method_name, train_fn in configs:
        print(f"--- {method_name} ---")
        for epoch in range(1, NUM_EPOCHS_TO_CHECK + 1):
            W, b, losses = train_fn(epoch)
            all_passed &= check_one("part_b", method_name, W, b, losses, epoch)
    return all_passed


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"

    all_passed = True
    if "a" in which:
        all_passed &= verify_part_a()
    if "b" in which:
        all_passed &= verify_part_b()

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
