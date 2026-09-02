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

**Bug found during Step 4 verification:** two feature columns (`signal_mad`, `valid_rr_fraction`)
are exactly constant (`1.0`) across the whole training set, so their std is `0`, and
`(X - mean) / std` computed `0/0 = nan` for those columns — which then poisoned every gradient.
Fixed by adding `std[std == 0] = 1.0` in `standardize_fit`: since a constant column's value always
equals its own mean, `(x - mean)` is already `0` for every row of that column, so the divisor
doesn't matter as long as it isn't `0` — this just avoids manufacturing a `nan`.

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

## Step 3: Gradient of the loss w.r.t. W and b

**What we're doing:** deriving the direction to nudge every entry of `W` and `b` to reduce the
cross-entropy loss. One formula, reused unchanged by all four training methods in Step 4 — only
how much data counts as "the batch" differs between them.

**Intuition:** softmax + cross-entropy has a famously clean combined gradient. The "error signal"
per example is just `(predicted probabilities - true one-hot)`. A confident, wrong prediction gives
a big error signal (big gradient step); a near-perfect prediction gives an error signal near zero
(tiny gradient step).

**The math:**

For a batch of `m` examples, with `P` = softmax probabilities `(m,3)` and `Y` = one-hot true labels
`(m,3)`:

```
E = P - Y                     # (m,3) - error per example per class

dL/dW = (1/m) * X^T @ E       # (78,3) - X^T is (78,m), E is (m,3)
dL/db = (1/m) * sum over rows of E     # (3,) - average error per class
```

**Critical rule:** `m` is the CURRENT batch's size (`|B|`), not the full training-set size `n`.
Full-batch GD -> `m = n`. Mini-batch/AdaGrad -> `m = 32`. SGD -> `m = 1`. Same formula every time.

**Why the shapes work:** `X` is `(m,78)` so `X^T` is `(78,m)`; `X^T @ E` is `(78,m) @ (m,3) =
(78,3)`, matching `W`. Entry `dL/dW[j,k] = (1/m) * sum_i X[i,j] * E[i,k]` — how much feature `j`,
weighted by the error on class `k`, mattered across the batch.

**The code (part_a.py):**

```python
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def compute_gradients(X, y, W, b):
    m = X.shape[0]
    Z = compute_logits(X, W, b)
    P = softmax(Z)
    Y = one_hot(y, num_classes=W.shape[1])
    E = P - Y                      # (m,3) - error per example per class

    grad_W = (X.T @ E) / m         # (78,m) @ (m,3) -> (78,3), matches W's shape
    grad_b = E.sum(axis=0) / m     # (3,) - average error per class

    return grad_W, grad_b
```

**Why each piece matters:**
- `np.eye(num_classes)[y]` builds the one-hot matrix in one line: `np.eye(3)` is the 3x3 identity,
  and indexing it by the label array `y` picks out row `y[i]` for every `i` — no Python loop.
- `compute_gradients` reuses `compute_logits`/`softmax` from Step 2 unchanged — this function will
  be called later with whatever `X, y` slice each training method currently considers "the batch."
- Dividing by `m = X.shape[0]` (the batch's own size) rather than a fixed `n` is exactly the
  spec's batch-size gradient rule, and is what makes this one function correct for all four
  methods in Step 4.

---

## Step 4: The four training loops

**What we're doing:** combining Steps 2-3 into actual training for all four required optimisers,
using the exact prescribed hyperparameters, then verifying our implementation epoch-by-epoch
against the reference weight/loss traces in `data/weight_traces/`.

**Prescribed hyperparameters (exact, autograded):**

```
Method             epochs   learning rate   batch size   extra
Full-batch GD        500        0.3         all n        -
Mini-batch GD         200        0.03        32           -
SGD                    30        0.001        1           -
Mini-batch AdaGrad    200        0.3         32           eps = 1e-8
All methods: shuffling seed 774
```

**The shared structure:**

```
W = zeros(78,3), b = zeros(3)                 # same init every method
rng = numpy.random.default_rng(774)           # created ONCE, before training

for epoch in range(num_epochs):
    batches = <index-arrays partitioning all n training rows>
    for batch_indices in batches:
        X_B, y_B = X_train[batch_indices], y_train[batch_indices]
        grad_W, grad_b = compute_gradients(X_B, y_B, W, b)    # |B| = len(batch_indices)
        <update W, b -- differs per method, see below>
    loss = cross_entropy_loss(X_train, y_train, W, b)         # FULL training set, every epoch
    record loss
```

**How batches differ per method:**
- **Full-batch GD:** no shuffling — the one batch is all `n` rows, every epoch, original order.
- **Mini-batch GD:** at the START of every epoch, `order = rng.permutation(n)`, then sliced into
  consecutive chunks of 32 (`order[0:32]`, `order[32:64]`, ...; last chunk possibly smaller).
- **SGD:** identical to mini-batch, chunk size 1.
- **Mini-batch AdaGrad:** same batching as mini-batch GD (chunks of 32, reshuffled every epoch).

**How the update rule differs per method:**

Plain GD (full-batch / mini-batch / SGD — same rule, different `eta` and batch size):
```
W = W - eta * grad_W
b = b - eta * grad_b
```

AdaGrad accumulates squared gradients per parameter (never reset across the whole run, not per
epoch) and divides the learning rate by their square root — parameters with big past gradients
get smaller effective steps, automatically, per-parameter:
```
G_W = 0, G_b = 0                          # accumulators, persist across the WHOLE run
# every batch:
G_W = G_W + grad_W * grad_W               # element-wise square, accumulated
W   = W - eta * grad_W / (sqrt(G_W) + eps)
# same for b, G_b
```

**Recording loss:** after every FULL epoch (not every batch), compute `cross_entropy_loss` over
the ENTIRE training set with the current `W, b`, regardless of which method/batch-size trained it.

**The code (part_a.py):**

```python
def iterate_batches(n, batch_size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def train_full_batch(X, y, epochs=500, eta=0.3):
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
    return train_mini_batch(X, y, epochs=epochs, eta=eta, batch_size=1, seed=seed)


def train_adagrad(X, y, epochs=200, eta=0.3, batch_size=32, eps=1e-8, seed=774):
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
            grad_W, grad_b = compute_gradients(X_B, y_B, W, b)
            G_W += grad_W * grad_W
            G_b += grad_b * grad_b
            W -= eta * grad_W / (np.sqrt(G_W) + eps)
            b -= eta * grad_b / (np.sqrt(G_b) + eps)
        losses.append(cross_entropy_loss(X, y, W, b))
    return W, b, losses
```

**Verification result:** ran all four methods for 5 epochs and compared against
`data/weight_traces/part_a/*_epoch{1-5}.txt` and `loss_by_epoch.csv` — max weight/bias differences
were ~1e-10, and losses matched to ~1e-15, both consistent with ordinary floating-point noise. All
four training loops are confirmed correct.

**Why each piece matters:**
- `iterate_batches` is one function reused by all four methods — passing `rng=None` gives
  unshuffled full-batch behavior, passing an `rng` gives the fresh-shuffle-every-epoch behavior the
  other three need. This avoids duplicating the batching logic four times.
- `train_sgd` is implemented as `train_mini_batch(..., batch_size=1)` since the spec defines SGD as
  exactly that — no separate logic needed.
- AdaGrad's `G_W`, `G_b` accumulators are created ONCE outside the epoch loop and never reset —
  they track the sum of squared gradients over the *entire* training run, which is what makes the
  effective learning rate shrink over time.

---

## Step 5: Writing weights.txt and predictions.txt

**What we're doing:** writing trained `W`, `b`, and predicted test-set probabilities to disk in the
exact format the autograder expects. No new math here — purely matching the spec precisely, since
format mismatches cost marks separately from correctness.

**The required formats:**

```
weights.txt        - exactly 79 lines
  line 1:      b[0],b[1],b[2]              <- bias, classes N,A,O in that order
  lines 2-79:  W[j,0],W[j,1],W[j,2]        <- one line per feature j, feature-column order

predictions.txt     - one line per test row, same order as the input CSV
  p_N,p_A,p_O                              <- three probabilities, comma-separated, sum to 1
                                               (never hard class labels)
```

**Why this "just works" given what we already built:**
- `W`'s row order already matches `weights.txt`'s required feature order — `feature_cols` has been
  threaded through consistently since Step 1, and `W`'s rows were never reordered relative to it.
- `predictions.txt`'s row order already matches `test.csv` — only *training* data gets shuffled
  (Step 4, via index arrays), `X_test` is never touched by that shuffling.

**The code (part_a.py):**

```python
def format_row(values):
    return ",".join(f"{v:.17g}" for v in values)


def write_weights(path, W, b):
    with open(path, "w") as f:
        f.write(format_row(b) + "\n")
        for row in W:
            f.write(format_row(row) + "\n")


def write_predictions(path, P):
    with open(path, "w") as f:
        for row in P:
            f.write(format_row(row) + "\n")
```

**Why each piece matters:**
- `.17g` formats each float with 17 significant digits (`g` = general format, no trailing zeros,
  scientific notation only if needed) — 17 significant digits is the mathematically-guaranteed
  minimum number of digits to round-trip any float64 value exactly, which matters for the
  autograder's "match reference within tolerance" check.
- `format_row` is one shared helper for both files, since both need the same
  "three comma-separated numbers, one line" shape.

---

## Step 6: CLI wiring (main())

**What we're doing:** replacing `main()`'s `raise NotImplementedError` placeholder with the real
dispatch: pick the right trainer for the requested `method`, train it, turn the resulting `W, b`
into test-set probabilities, and write both output files. This finishes Part (a)'s `part_a.py`.

**The logic:**
1. Validate `method` is one of the four keys in the `METHODS` dict (Step 4) — otherwise print a
   usage error and exit rather than crashing with a confusing `KeyError`.
2. Call `METHODS[method](X_train, y_train)` to get back `W, b` (the per-epoch `losses` list is
   discarded here — it's only needed for the report's loss-vs-time plots, generated separately).
3. Get test-set probabilities the same way Step 2 computes training-set probabilities internally —
   `compute_logits` then `softmax` — just with no label to compare against.
4. Write both files with Step 5's writers.

**The code (part_a.py, inside main()):**

```python
    if method not in METHODS:
        print(f"unknown method '{method}', expected one of {sorted(METHODS)}", file=sys.stderr)
        sys.exit(1)

    W, b, _losses = METHODS[method](X_train, y_train)

    P_test = softmax(compute_logits(X_test, W, b))

    write_weights(weights_path, W, b)
    write_predictions(predictions_path, P_test)
```

**End-to-end verification:** ran all four methods against the real `part_ab_train.csv` /
`part_ab_test_public.csv` (924 test rows). Results:
- `weights.txt`: exactly 79 lines for every method.
- `predictions.txt`: exactly 924 lines (one per test row) for every method.
- Every prediction row sums to 1 within ~1e-16 (floating-point noise).
- Full-batch (500 epochs) ~8s, mini-batch (200 epochs) ~6s, SGD (30 epochs) ~11s, AdaGrad (200
  epochs) ~6s — all comfortably fast, no Kaggle/GPU needed for Part (a).

**`part_a.py` is now functionally complete** for Part (a)'s required CLI:
```
python3 part_a.py part_ab_train.csv part_ab_test_public.csv {full_batch,mini_batch,sgd,adagrad} \
    predictions.txt weights.txt
```

**Still open (for the report, not part_a.py itself):** the report needs training-loss AND
validation-loss vs. time plots for all four methods on the same axes. `part_a.py`'s graded CLI only
takes train+test paths (no validation path, per spec) and doesn't track wall-clock time per epoch —
so generating those plots will need a separate small script that reuses these same functions
(`load_xy`, `standardize_fit/apply`, the four `train_*` functions extended to also evaluate
`part_ab_val.csv`'s loss and record `time.time()` per epoch). Not yet built.

---

*(more sections appended as we progress through the assignment)*
