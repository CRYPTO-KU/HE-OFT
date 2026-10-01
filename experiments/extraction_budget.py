#!/usr/bin/env python
"""How many label-only queries a functional copy of the shared head costs.

The adversary is a participating client. It computes query features itself, so
it may submit any vector of the feature space, and it receives one label per
query and nothing else. The served head is linear in the features,
y = argmax_c (W phi + b)_c, so extraction is the problem of learning a linear
classifier from label queries. The attack fits a multinomial logistic
regression to the (query, label) pairs.

Two query strategies bracket the adversary:
  random    queries drawn from N(0, I/d)
  boundary  half the budget random, the other half spent bisecting between
            random queries of different labels, which places queries near a
            decision boundary where a label carries the most information

Fidelity is the agreement of the copy with the head on 20000 held-out points of
the same distribution. Majority is the share of the largest class among those
points, the trivial baseline. Two reference rows carry fidelity 1 by
construction: logits, an interface that returns the logits and lets the head be
solved for from d+1 queries, and disclosed, a head handed over in plaintext at
zero queries.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200. The output is results.csv, one row per (task, seed,
arrangement, strategy, budget).

Usage:
  python experiments/extraction_budget.py
  python experiments/extraction_budget.py --tasks banking77 --seeds 42
"""
import argparse

import numpy as np
import torch

from heoft.attacks.extraction import (Oracle, fit_linear, head_of,
                                      query_boundary, query_random)
from heoft.common import artifacts_dir, results_dir
from heoft.data import TEXT_TASKS

NAME = "extraction_budget"
BUDGETS = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
N_EVAL = 20000
SEEDS = [42, 43, 44]
TASKS = ["ag_news", "dbpedia_14", "banking77"]


def run(task, seed, rows):
    path = artifacts_dir("accuracy_text") / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        print(f"  [skip] no artifact for {task} s{seed}", flush=True)
        return
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])
    rng = np.random.default_rng(seed)

    for arrangement in ("A", "B"):
        try:
            W, b = head_of(art, arrangement)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [skip] {task} s{seed} {arrangement}: {e}", flush=True)
            continue
        d = W.shape[1]
        scale = 1.0 / np.sqrt(d)
        Xe = rng.normal(0, scale, size=(N_EVAL, d))
        ytrue = np.argmax(Xe @ W.T + b, axis=1)
        majority = float(np.bincount(ytrue, minlength=C).max() / len(ytrue))

        for access, q in (("disclosed", 0), ("logits", d + 1)):
            rows.append(dict(task=task, C=C, d=d, seed=seed,
                             arrangement=arrangement, strategy=access,
                             queries=q, fidelity=1.0, majority=round(majority, 4)))

        for strategy in ("random", "boundary"):
            for nq in BUDGETS:
                orc = Oracle(W, b)
                if strategy == "random":
                    X, y = query_random(orc, nq, d, rng, scale)
                else:
                    X, y = query_boundary(orc, nq, d, rng, scale, C)
                if len(np.unique(y)) < 2:
                    continue
                Wh, bh = fit_linear(X, y, C)
                fid = float((np.argmax(Xe @ Wh.T + bh, axis=1) == ytrue).mean())
                rows.append(dict(task=task, C=C, d=d, seed=seed,
                                 arrangement=arrangement, strategy=strategy,
                                 queries=int(orc.n), fidelity=round(fid, 4),
                                 majority=round(majority, 4)))
                print(f"  {task} s{seed} {arrangement} {strategy:<8} "
                      f"q={orc.n:<7} fid={fid:.4f} (majority {majority:.3f})",
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

    cols = ["task", "C", "d", "seed", "arrangement", "strategy", "queries",
            "fidelity", "majority"]
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
