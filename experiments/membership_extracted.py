#!/usr/bin/env python
"""Membership inference against the shared head, held in plaintext or extracted.

The adversary is a coalition of clients. It holds the public backbone, its own
clients' data and adapters, and a query allowance, and it receives one label
per query. It never sees a score or a logit. It asks whether a given example
was in a client's training set.

The attack is LiRA (Carlini et al., IEEE S&P 2022). For each candidate the
attacker fits one Gaussian to the rescaled logit over the shadow federations
that trained on the candidate and one over those that did not, and scores the
candidate by the likelihood ratio. A shadow federation is N clients, each
training a head for K steps on cached features of a random half of its own
pool (its training and held-out examples together), merged by the
coverage-weighted rule. Sampling from the whole pool makes every candidate a
member of about half the shadows. The backbone is frozen and public, so the
shadows repeat no backbone forward pass.

Members are the clients' training examples and non-members their held-out
examples, so both come from the same clients' data. By default every client is
pooled, because one client's held-out set is too small for a measurement at a
low false-positive rate.

Three surfaces, from the strongest attacker to the weakest:
  truehead   the attacker holds the shared head in plaintext and reads its
             logits. The head is encrypted throughout the protocol, so this
             setting arises only if the encryption is broken. It estimates the
             advantage of an adversary that holds the head, which by the
             data-processing inequality no attack on a copy of the head exceeds.
  extracted  the attacker holds a copy fitted to Q label-only answers to random
             queries, for Q in 2000, 20000 and 200000.
  gap        member if and only if the head classifies the example correctly.
             No query and no shadow. The held-out set reserves one example of
             every class a client holds, so non-members are enriched in rare
             classes and this baseline measures that enrichment as well as
             membership. LiRA calibrates each candidate against its own shadows
             and is not affected.

Arrangement A serves the head over the bare backbone. Arrangement B serves it
over a client's own adapter, here the adapter at index --target of the
artifact's client list, which is the last client at the default -1.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200. The output is results.csv, one row per (task, seed,
arrangement, surface, budget), with the true-positive rate at false-positive
rates of 0.1 and 1 per cent and the area under the ROC curve.

Usage:
  python experiments/membership_extracted.py
  python experiments/membership_extracted.py --tasks ag_news --seeds 42
"""
import argparse

import numpy as np
import torch

from heoft.attacks.extraction import Oracle, fit_linear, head_of, query_random
from heoft.attacks.membership import (N_CAND, SEEDS, SHADOWS, TARGET, TASKS,
                                      features_of, logit_stat, roc_points,
                                      train_head)
from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, dirichlet_partition, text_data
from heoft.models import TextLoRA, load_trainable
from heoft.selection import drop_small_clients, stratified_holdout
from heoft.train import BACKBONE, R

NAME = "membership_extracted"
BUDGETS = [2000, 20000, 200000]


def merge(theta0_W, theta0_b, heads, counts):
    """The coverage-weighted merge in float64. The holders of a class decide its row."""
    counts = np.asarray(counts, dtype=np.float64)
    den = np.where(counts.sum(0) > 0, counts.sum(0), 1.0)
    numW = np.zeros_like(theta0_W)
    numb = np.zeros_like(theta0_b)
    for j, (Wj, bj) in enumerate(heads):
        numW += counts[j][:, None] * (Wj - theta0_W)
        numb += counts[j] * (bj - theta0_b)
    return theta0_W + numW / den[:, None], theta0_b + numb / den


def shadow_federation(F, y, pools, C, theta0_W, theta0_b, rng):
    """One shadow: every client trains on a random half of its own pool.

    The pool is the client's training and held-out examples together. A pool
    of the training split alone would leave every non-member out of every
    shadow, and LiRA would have no IN distribution for it.
    Returns the merged head and the membership mask of the shadow.
    """
    heads, counts, inmask = [], [], np.zeros(len(y), dtype=bool)
    for p in pools:
        take = rng.random(len(p)) < 0.5
        idx = p[take]
        if len(idx) < 4:
            idx = p[:max(4, len(p) // 2)]
        inmask[idx] = True
        heads.append(train_head(F[idx], y[idx], C, theta0_W, theta0_b))
        counts.append(np.bincount(y[idx], minlength=C).astype(np.float64))
    W, b = merge(theta0_W, theta0_b, heads, counts)
    return W, b, inmask


def run(task, seed, rows, cfg):
    SHADOWS, TARGET, N_CAND = cfg.shadows, cfg.target, cfg.candidates
    path = artifacts_dir("accuracy_text") / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        print(f"  [skip] no artifact for {task} s{seed}", flush=True)
        return
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])
    N, ALPHA = int(art.get("N", 10)), float(art.get("alpha", 0.1))

    ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C2 = text_data(task, BACKBONE, seed)
    assert C2 == C, (C2, C)
    y = np.asarray(ytr)
    parts = drop_small_clients(dirichlet_partition(y, N, ALPHA, C, seed))
    tr_parts, va_parts = stratified_holdout(parts, seed, y)
    # TARGET < 0 pools every client, which measures what a coalition learns
    # about the training examples it does not hold.
    if TARGET < 0:
        members = np.concatenate([np.asarray(p) for p in tr_parts])
        nonmembers = np.concatenate([np.asarray(p) for p in va_parts])
    else:
        if TARGET >= len(parts):
            print(f"  [skip] {task} s{seed}: target {TARGET} beyond {len(parts)}",
                  flush=True)
            return
        members = np.asarray(tr_parts[TARGET])
        nonmembers = np.asarray(va_parts[TARGET])
    n = min(len(members), len(nonmembers), N_CAND)
    if n < 50:
        print(f"  [skip] {task} s{seed}: only {n} usable candidates "
              f"({len(members)} members, {len(nonmembers)} non-members)", flush=True)
        return
    # the shadows randomise membership over each client's whole pool
    pools = [np.concatenate([np.asarray(a), np.asarray(v)])
             for a, v in zip(tr_parts, va_parts)]
    rng = np.random.default_rng(seed)
    members = rng.choice(members, n, replace=False)
    nonmembers = rng.choice(nonmembers, n, replace=False)
    cand = np.concatenate([members, nonmembers])
    label = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    print(f"  {task} s{seed}: {n} members and {n} non-members over "
          f"{len(parts)} clients", flush=True)

    for arrangement in ("A", "B"):
        tag = "r0" if arrangement == "A" else "r8"
        r = 0 if arrangement == "A" else R
        set_seed(seed)
        model = TextLoRA(BACKBONE, C, r=r, freeze_a=True).to(DEVICE)
        if arrangement == "B":
            load_trainable(model, art[f"states_{tag}"][TARGET])
        F_cand = features_of(model, ids_tr[cand], mask_tr[cand])
        F_all = features_of(model, ids_tr, mask_tr)
        del model
        empty_cache()

        W, b = head_of(art, arrangement)
        d = W.shape[1]
        theta0 = art[f"theta0_{tag}"]
        hk = [k for k in theta0 if "head" in k]
        wk = [k for k in hk if theta0[k].ndim == 2][0]
        bk = [k for k in hk if theta0[k].ndim == 1][0]
        t0W = theta0[wk].double().numpy()
        t0b = theta0[bk].double().numpy()

        # shadow federations, recording IN/OUT per candidate
        srng = np.random.default_rng(seed * 1000 + 7)
        s_in = [[] for _ in range(len(cand))]
        s_out = [[] for _ in range(len(cand))]
        for s in range(SHADOWS):
            Ws, bs_, inmask = shadow_federation(F_all, y, pools, C, t0W, t0b, srng)
            st = logit_stat(Ws, bs_, F_cand, y[cand])
            for i, c in enumerate(cand):
                (s_in if inmask[c] else s_out)[i].append(st[i])
            if (s + 1) % 16 == 0:
                print(f"    {task} s{seed} {arrangement}: {s+1}/{SHADOWS} shadows",
                      flush=True)
        keep = np.array([len(a) >= 2 and len(o) >= 2
                         for a, o in zip(s_in, s_out)])
        if keep.sum() < 20:
            print(f"  [skip] {task} s{seed} {arrangement}: "
                  f"only {keep.sum()} candidates with both sides", flush=True)
            continue
        SI = np.array([np.mean(a) for a in s_in])
        SO = np.array([np.mean(o) for o in s_out])
        SIs = np.array([np.std(a) + 1e-6 for a in s_in])
        SOs = np.array([np.std(o) + 1e-6 for o in s_out])

        def report(surface, queries, Wa, ba):
            st = logit_stat(Wa, ba, F_cand, y[cand])
            sc = (-0.5 * ((st - SI) / SIs) ** 2 - np.log(SIs)) - \
                 (-0.5 * ((st - SO) / SOs) ** 2 - np.log(SOs))
            t1, t2, auc = roc_points(sc[keep], label[keep])
            rows.append(dict(task=task, C=C, d=d, seed=seed, target=TARGET,
                             arrangement=arrangement, surface=surface,
                             queries=queries, shadows=SHADOWS,
                             n_candidates=int(keep.sum()),
                             tpr_at_0_1pct=round(t1, 5),
                             tpr_at_1pct=round(t2, 5), auc=round(auc, 4)))
            print(f"  {task} s{seed} {arrangement} {surface:<14} q={queries:<7} "
                  f"TPR@0.1%={t1:.4f} TPR@1%={t2:.4f} AUC={auc:.4f}", flush=True)

        report("truehead", 0, W, b)

        scale = 1.0 / np.sqrt(d)
        for nq in BUDGETS:
            orc = Oracle(W, b)
            X, yq = query_random(orc, nq, d, rng, scale)
            if len(np.unique(yq)) < 2:
                continue
            Wh, bh = fit_linear(X, yq, C)
            report("extracted", int(orc.n), Wh, bh)

        # the baseline, member if and only if the served head is right
        pred = np.argmax(F_cand @ W.T + b, axis=1)
        sc = (pred == y[cand]).astype(np.float64)
        t1, t2, auc = roc_points(sc[keep], label[keep])
        rows.append(dict(task=task, C=C, d=d, seed=seed, target=TARGET,
                         arrangement=arrangement, surface="gap", queries=1,
                         shadows=0, n_candidates=int(keep.sum()),
                         tpr_at_0_1pct=round(t1, 5), tpr_at_1pct=round(t2, 5),
                         auc=round(auc, 4)))
        print(f"  {task} s{seed} {arrangement} {'gap':<14} q=1       "
              f"TPR@0.1%={t1:.4f} TPR@1%={t2:.4f} AUC={auc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--shadows", type=int, default=SHADOWS,
                    help="shadow federations per (task, seed)")
    ap.add_argument("--candidates", type=int, default=N_CAND,
                    help="members drawn, and as many non-members")
    ap.add_argument("--target", type=int, default=TARGET,
                    help="client attacked, -1 pools every client")
    cfg = ap.parse_args()
    tasks, seeds = cfg.tasks, cfg.seeds
    rows = []
    for t in tasks:
        for s in seeds:
            run(t, s, rows, cfg)
    if not rows:
        raise SystemExit("no rows produced")

    cols = ["task", "C", "d", "seed", "target", "arrangement", "surface",
            "queries", "shadows", "n_candidates", "tpr_at_0_1pct",
            "tpr_at_1pct", "auc"]
    out = results_dir(NAME) / "results.csv"
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)\n", flush=True)
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
