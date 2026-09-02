# Learning Guide — COL774 A1 Part 2 (Logistic Regression)

Personal revision notes: what we did, why, the math, and the key code idea for each concept.
Math is written in plain text (no LaTeX) since it's read in a terminal.

---

## Part (a) — Overview

Goal: build a 3-class softmax classifier (0=Normal, 1=AF, 2=Other) on 78 hand-engineered ECG
features, trained FOUR different ways:

1. **Full-batch GD** — one gradient step per epoch, using all n examples.
2. **Mini-batch GD** — batches of 32, reshuffled every epoch.
3. **SGD** — mini-batch GD with batch size 1.
4. **Mini-batch AdaGrad** — mini-batch GD (batch 32) with a per-parameter adaptive learning rate.

All hyperparameters (epochs, learning rates, batch sizes, seed) are fixed by the spec — no
tuning here. This is autograded: `data/weight_traces/part_a/*_epoch{1..5}.txt` and
`data/weight_traces/loss_by_epoch.csv` give reference weights/losses for the first 5 epochs of
each method, so we can verify our implementation epoch-by-epoch before trusting it.

Build order:
1. Data loading + standardization
2. Softmax + cross-entropy loss (with the stability trick)
3. Gradient of the loss w.r.t. W, b (one formula, reused by all four methods)
4. The four training loops, each checked against the reference traces
5. Output writing (weights.txt, predictions.txt) in the exact required format
6. CLI wiring (sys.argv, method dispatch, no hard-coded paths)

---

## Step 1: Loading data & standardization

**What we're doing:** reading `part_ab_train.csv` / `part_ab_val.csv` / `part_ab_test_public.csv`,
dropping the `release_id` column (a traceability key, not a feature — must never be used as
model input), separating the 78 feature columns from the `label` column, and standardizing
every feature to zero mean / unit variance using **training-set statistics only**.

**Intuition:** the 78 features live on very different scales (e.g. `rr_mean_s` ~ 0.5-1.5, while
`beat_energy` could be in the hundreds/thousands). Gradient descent uses one learning rate for
all weights at once. If feature scales differ by orders of magnitude, the loss surface becomes a
long narrow valley and GD zig-zags instead of heading straight to the minimum. Standardizing
every feature makes the loss surface roughly round, so a single learning rate works well across
all weights.

**The math:**

For each feature column j, compute from the TRAINING set only:

```
mean_j = (1/n) * sum_i x_i_j
std_j  = sqrt( (1/n) * sum_i (x_i_j - mean_j)^2 )        <- population std, ddof=0 (numpy default)
```

Then transform every row (train, val, AND test) with the same mean_j, std_j:

```
x_tilde_i_j = (x_i_j - mean_j) / std_j
```

**Critical rule:** mean_j and std_j are computed ONCE from training data and reused unchanged on
val/test. Never recompute stats on val/test — that leaks test-distribution info into
preprocessing, and would also produce numbers that don't match the autograder's reference
implementation (losing marks).

**Key code idea:** compute `mean = X_train.mean(axis=0)`, `std = X_train.std(axis=0)` (numpy's
default `ddof=0` is already the population std we want), then apply
`(X - mean) / std` to train/val/test alike — never re-fit on val/test.

**The code (part_a.py):**

```python
def load_xy(path, feature_cols=None, has_label=True):
    df = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = get_feature_columns(df)   # 78 names, in file order, minus release_id/label

    X = df[feature_cols].to_numpy(dtype=np.float64)   # (n, 78) design matrix
    y = df["label"].to_numpy(dtype=np.int64) if has_label else None

    return X, y, feature_cols


def standardize_fit(X_train):
    mean = X_train.mean(axis=0)   # axis=0 -> collapse rows, one mean per column -> shape (78,)
    std = X_train.std(axis=0)     # numpy .std() default is ddof=0 (population std) - matches spec
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std       # broadcasting: (n,78) - (78,) works row-wise, no explicit loop
```

**Why each piece matters:**
- `feature_cols` is computed once from the train file and passed explicitly when loading test
  data, so both splits use the *same* column order even though the test CSV has no `label`
  column to filter out. The autograder checks `weights.txt` rows are in this exact order.
- `dtype=np.float64` on `to_numpy` isn't cosmetic — the spec requires float64 arithmetic
  throughout, and pandas can otherwise infer a narrower dtype from the CSV.
- `(X - mean) / std` relies on numpy **broadcasting**: `X` is shape `(n, 78)`, `mean`/`std` are
  shape `(78,)`. Numpy lines up the trailing dimension (78 vs 78) and repeats the 1D array across
  all `n` rows automatically — so one line standardizes every row without an explicit Python
  loop. This is why `standardize_apply` can be reused unchanged for train, val, and test: it just
  needs whatever `mean`/`std` you pass it.

---

## Step 2: Softmax + cross-entropy loss

**What we're doing:** turning raw per-class scores (logits) into class probabilities via softmax,
then measuring how wrong those probabilities are against the true labels via cross-entropy loss.
This loss is what all four training methods in Step 4 will minimize.

**Intuition:**
- Logits `z = x @ W + b` are arbitrary real numbers, one per class — bigger means "the model
  favors this class more," but they don't sum to 1 and can be negative.
- Softmax exponentiates (making everything positive) and normalizes (dividing by the row total),
  turning logits into a valid probability distribution over the 3 classes.
- Cross-entropy loss penalizes the model based on how much probability it assigned to the
  *correct* class: near 0 penalty if it was confident and correct, huge penalty if it was
  confident and wrong.

**The math:**

Logits for one example:
```
z = x @ W + b        # shape (3,)
```
For a whole batch:
```
Z = X @ W + b         # shape (n, 3)
```

Softmax (naive form):
```
p_k = exp(z_k) / sum_j exp(z_j)          for k = 0, 1, 2
```

Numerically stable form (mathematically identical — shifting every logit in a row by the same
constant doesn't change the result, since the shift factors out of numerator and denominator):
```
m         = row-wise max of Z
Z_shifted = Z - m
Z_clipped = clip(Z_shifted, -60, 0)      <- spec's exact stability rule
P         = exp(Z_clipped) / row-wise sum of exp(Z_clipped)
```

Cross-entropy loss (mean over all rows, no regularization):
```
loss_i = -log( p_i, y_i )                # p_i,y_i = predicted prob of the TRUE class for row i
L = (1/n) * sum_i loss_i
```

**Why the stability trick works:** `exp(z_k - c) / sum_j exp(z_j - c) = exp(z_k) / sum_j exp(z_j)`
for ANY constant `c`, because `exp(z_k - c) = exp(z_k) * exp(-c)`, and the `exp(-c)` factor cancels
between numerator and denominator. Choosing `c = max_j z_j` makes the largest shifted logit exactly
0, so every `exp(...)` term is safely in `(0, 1]` — no overflow.

**The code (part_a.py):**

```python
def compute_logits(X, W, b):
    return X @ W + b


def softmax(Z):
    row_max = Z.max(axis=1, keepdims=True)      # (n,1) - largest logit per row
    Z_shifted = Z - row_max                     # broadcast: (n,3) - (n,1) -> (n,3)
    Z_clipped = np.clip(Z_shifted, -60.0, 0.0)
    exp_Z = np.exp(Z_clipped)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss(X, y, W, b):
    n = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    true_class_probs = P[np.arange(n), y]        # (n,) - P[i, y[i]] for every i
    return -np.mean(np.log(true_class_probs))
```

**Why each piece matters:**
- `keepdims=True` on both the max and the sum is essential: it keeps the result shape `(n,1)`
  instead of `(n,)`, so broadcasting against `(n,3)` subtracts/divides *each row's own* value
  from every entry in that row, rather than misaligning rows and columns.
- `P[np.arange(n), y]` is numpy fancy indexing: `np.arange(n)` gives row indices `0..n-1` and `y`
  gives the true class per row, so this single expression extracts `P[i, y[i]]` for every `i` at
  once — no Python loop needed.
- This loss function will be called both for epoch-level full-training-set loss logging (Step 4)
  and, if needed, for validation loss — same function, different `X, y` passed in.

---

*(more sections appended as we progress through the assignment)*
