#!/usr/bin/env python
"""What a row of the shared head carries about the clients that decided it.

Row c of the shared head is the n_{j,c}-weighted mean of the row-c
displacements of the clients holding class c. A class held by one client
therefore has that client's displacement as its row, with no dilution over the
federation. For a linear layer with a bias trained under cross-entropy, the
gradient of row c is xi_c times the feature and the gradient of bias c is xi_c,
so their ratio is the feature (Phong et al., IEEE TIFS 2018). Accumulated over
local training, the ratio of the row-c displacement to the bias-c displacement
becomes a xi-weighted mean of the holders' features, dominated by their class-c
examples.

The adversary holds the row in plaintext. The head is encrypted throughout the
protocol, so this setting arises only if the encryption is broken, and it is the
strongest case. For every class c the experiment measures the cosine between
that ratio and
  cos_mean        the mean feature of the class-c training examples of the
                  holders of c
  cos_top1        the best-matching single example among them
  cos_top1_other  the best-matching example of the training pool outside that
                  set
  cos_rand        64 random Gaussian directions, as the mean absolute cosine,
                  a baseline to read the others against
If cos_top1 is close to cos_mean, the ratio is a class direction that every
class-c example matches about equally, and it singles out no record. If
cos_top1 does not exceed cos_top1_other, the row carries no class signal
either. A class is skipped when no client holds it, when its bias displacement
is below 1e-9 in absolute value, or when its holders have fewer than two
examples of it.

No client adapter is loaded. Arrangement A computes features with the bare
backbone and arrangement B with a freshly initialized rank-8 adapter, whose
up-projection starts at zero, so both use the features of the bare backbone.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200. The output is results.csv, one row per (task, seed,
arrangement, class), with the number of holders of the class and of their
examples of it, and the four cosines.

Usage:
  python experiments/row_leakage.py
  python experiments/row_leakage.py --tasks banking77 --seeds 42 43 44
"""
import argparse

import numpy as np
import torch

from heoft.attacks.extraction import head_of
from heoft.attacks.membership import features_of
from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, dirichlet_partition, text_data
from heoft.models import TextLoRA
from heoft.selection import drop_small_clients, stratified_holdout
from heoft.train import BACKBONE, R

NAME = "row_leakage"
TASKS = ["ag_news", "dbpedia_14", "banking77"]
SEEDS = [42, 43, 44]


def cosine(a, B):
    """Cosine between the vector a and every row of B."""
    a = a / (np.linalg.norm(a) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return Bn @ a


def run(task, seed, rows):
    path = artifacts_dir("accuracy_text") / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        print(f"  [skip] no artifact for {task} s{seed}", flush=True)
        return
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])
    N, ALPHA = int(art.get("N", 10)), float(art.get("alpha", 0.1))

    ids_tr, mask_tr, ytr, *_ = text_data(task, BACKBONE, seed)
    y = np.asarray(ytr)
    parts = drop_small_clients(dirichlet_partition(y, N, ALPHA, C, seed))
    tr_parts, _ = stratified_holdout(parts, seed, y)
    counts = np.asarray(art["counts"], dtype=np.float64)
    rng = np.random.default_rng(seed)

    for arrangement in ("A", "B"):
        tag = "r0" if arrangement == "A" else "r8"
        r = 0 if arrangement == "A" else R
        set_seed(seed)
        model = TextLoRA(BACKBONE, C, r=r, freeze_a=True).to(DEVICE)
        F = features_of(model, ids_tr, mask_tr)
        del model
        empty_cache()

        W, b = head_of(art, arrangement)
        theta0 = art[f"theta0_{tag}"]
        hk = [k for k in theta0 if "head" in k]
        wk = [k for k in hk if theta0[k].ndim == 2][0]
        bk = [k for k in hk if theta0[k].ndim == 1][0]
        dW = W - theta0[wk].double().numpy()
        db = b - theta0[bk].double().numpy()

        # which training rows belong to a client that holds class c
        owner = np.full(len(y), -1, dtype=int)
        for j, p in enumerate(tr_parts):
            owner[np.asarray(p)] = j

        for c in range(C):
            holders = np.flatnonzero(counts[:, c] > 0)
            if len(holders) == 0 or abs(db[c]) < 1e-9:
                continue
            mine = np.flatnonzero((y == c) & np.isin(owner, holders))
            if len(mine) < 2:
                continue
            ratio = dW[c] / db[c]
            sims = cosine(ratio, F)
            cm = float(cosine(ratio, F[mine].mean(0, keepdims=True))[0])
            cr = float(np.abs(cosine(ratio, rng.normal(size=(64, F.shape[1])))).mean())
            # If cos_top1 is close to cos_mean, the ratio is a class direction
            # that every class-c example matches about equally well, and it
            # singles out no record.
            ct1 = float(sims[mine].max())
            other = np.flatnonzero(~((y == c) & np.isin(owner, holders)))
            cot = float(sims[other].max()) if len(other) else float("nan")
            rows.append(dict(task=task, C=C, seed=seed, arrangement=arrangement,
                             cls=c, holders=int(len(holders)),
                             n_holder=int(len(mine)),
                             cos_mean=round(cm, 4), cos_top1=round(ct1, 4),
                             cos_top1_other=round(cot, 4), cos_rand=round(cr, 4)))
        got = [r_ for r_ in rows if r_["seed"] == seed and r_["task"] == task
               and r_["arrangement"] == arrangement]
        if got:
            few = [r_ for r_ in got if r_["holders"] <= 2]
            print(f"  {task} s{seed} {arrangement}: {len(got)} classes, "
                  f"{len(few)} held by at most two clients. "
                  f"cos to class mean {np.mean([r_['cos_mean'] for r_ in got]):.4f}, "
                  f"to the best single example {np.mean([r_['cos_top1'] for r_ in got]):.4f}, "
                  f"to the best other-class example "
                  f"{np.nanmean([r_['cos_top1_other'] for r_ in got]):.4f}, "
                  f"random baseline {np.mean([r_['cos_rand'] for r_ in got]):.4f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    cfg = ap.parse_args()
    tasks, seeds = cfg.tasks, cfg.seeds
    rows = []
    for t in tasks:
        for s in seeds:
            run(t, s, rows)
    if not rows:
        raise SystemExit("no rows produced")

    cols = ["task", "C", "seed", "arrangement", "cls", "holders", "n_holder",
            "cos_mean", "cos_top1", "cos_top1_other", "cos_rand"]
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
