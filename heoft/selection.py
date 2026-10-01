"""Choosing between the two arrangements without a test set.

Arrangement A, the shared head, serves the merged head over the bare backbone
(clients train with rank 0). Arrangement B, the personal arrangement, serves the
merged head over each client's own adapter. The global-prior estimator scores
both as expected accuracy on an example drawn from the federation's class
prior, from per-class counts of correct answers on each client's held-out set.
The paper's estimator is estimate(..., fill="zero") for B against
estimate(..., pooled=True) for A.
"""
import numpy as np

MIN_SHARD = 20        # a client needs enough data to split into train and holdout
VAL_FRAC = 0.1        # held-out fraction of each client's shard
STRAT_PER_CLASS = 1   # held-out examples reserved from every class a client holds
RARE_Q = 2            # a class is rare for a client at <= RARE_Q training examples


def drop_small_clients(parts):
    """Drop clients whose Dirichlet shard is smaller than MIN_SHARD.

    At small alpha the partition can give a client too few examples to split
    into a training and a held-out set. The run reports the effective N.
    """
    keep = [p for p in parts if len(p) >= MIN_SHARD]
    if len(keep) != len(parts):
        print(f"  [partition] dropped {len(parts) - len(keep)}/{len(parts)} "
              f"clients with <{MIN_SHARD} samples -> effective N={len(keep)}",
              flush=True)
    return keep


def stratified_holdout(parts, seed, y=None):
    """Carve each client's held-out set, stratified by the classes it holds.

    Each client reserves STRAT_PER_CLASS examples from every class it holds and
    tops the held-out set up to VAL_FRAC at random. A client holding a single
    example of a class therefore never trains on it, which gives the estimator
    a measurement on classes the client barely knows.
    Returns (training indices, held-out indices) per client.
    """
    rng = np.random.default_rng(seed + 7)
    tr, va = [], []
    for idx in parts:
        idx = np.asarray(idx)
        keep = np.zeros(len(idx), dtype=bool)
        if y is not None:
            lab = np.asarray(y)[idx]
            for c in np.unique(lab):
                pos = np.where(lab == c)[0]
                keep[rng.choice(pos, min(STRAT_PER_CLASS, len(pos)),
                                replace=False)] = True
        deficit = max(1, int(round(VAL_FRAC * len(idx)))) - int(keep.sum())
        if deficit > 0:
            rest = np.where(~keep)[0]
            if len(rest):
                keep[rng.choice(rest, min(deficit, len(rest)), replace=False)] = True
        if (~keep).sum() == 0:          # keep at least one training example
            keep[rng.choice(np.where(keep)[0], 1, replace=False)] = False
        va.append(idx[keep])
        tr.append(idx[~keep])
    return tr, va


def global_prior(counts):
    """p(c) over the federation, from the per-class counts n_{j,c}."""
    tot = np.asarray(counts).sum(0).astype(float)
    return tot / max(tot.sum(), 1e-12)


def per_class_nk(logits, y, C):
    """Per class: the number of held-out examples and the number answered correctly."""
    p = logits.argmax(1).numpy()
    y = np.asarray(y)
    n = np.array([(y == c).sum() for c in range(C)], dtype=float)
    k = np.array([((y == c) & (p == c)).sum() for c in range(C)], dtype=float)
    return n, k


def rare_fill(train_counts_j, n, k, q=RARE_Q):
    """Held-out accuracy on the classes with at most q training examples.

    The client's stand-in for classes it never saw. Rarity is absolute: with
    four classes a client's relatively rarest class still has hundreds of
    examples.
    """
    rare = (np.asarray(train_counts_j) <= q) & (n > 0)
    return float(k[rare].sum() / max(n[rare].sum(), 1.0)) if rare.any() else 0.0


def estimate(pg, nk_list, w, counts, pooled, fill="zero"):
    """Expected accuracy on an example drawn from the global prior pg.

    pooled=True (A): one shared model, so per-class evidence adds up across
    clients and every class held by some client is measured.
    pooled=False (B): one model per client. Client j has evidence only for its
    own classes, and every other class takes a fill. fill="zero" makes the
    estimate a lower bound, fill="rare" uses rare_fill.
    """
    if pooled:
        n = sum(nk[0] for nk in nk_list)
        k = sum(nk[1] for nk in nk_list)
        acc = np.where(n > 0, k / np.maximum(n, 1.0), 0.0)
        return float(np.dot(pg, acc))
    per = []
    for j, (n, k) in enumerate(nk_list):
        f = 0.0 if fill == "zero" else rare_fill(counts[j], n, k)
        acc = np.where(n > 0, k / np.maximum(n, 1.0), f)
        per.append(float(np.dot(pg, acc)))
    return float(np.dot(w, per))
