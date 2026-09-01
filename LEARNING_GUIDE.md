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

*(more sections appended as we progress through the assignment)*
