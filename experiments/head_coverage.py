#!/usr/bin/env python
"""How many rows of the shared head the federation covers.

Row c of the shared head is the coverage-weighted mean of the row-c
displacements of the clients holding class c. A class no client holds keeps its
initial row, which is random and still competes in the argmax. A class one
client holds takes that client's displacement with no dilution.

The experiment replays the Dirichlet partition of the accuracy runs, N=10 and
alpha=0.1, which is deterministic in the labels, N, alpha, C and the seed, and
counts the holders of every class among the clients left after the
minimum-shard rule. It trains nothing and needs no GPU.

The output is results.csv, one row per (task, seed), with
  n_clients   clients left after the minimum-shard rule
  holders_0   classes no client holds
  holders_1   classes exactly one client holds
  holders_2   classes exactly two clients hold
  holders_3p  classes three or more clients hold

Usage:
  python experiments/head_coverage.py
  python experiments/head_coverage.py --tasks banking77
"""
import argparse

import numpy as np

from heoft.common import results_dir
from heoft.data import TEXT_TASKS, dirichlet_partition, text_data
from heoft.selection import drop_small_clients
from heoft.train import BACKBONE

NAME = "head_coverage"
TASKS = ["ag_news", "trec", "dbpedia_14", "banking77"]
SEEDS = [42, 43, 44]
N, ALPHA = 10, 0.1


def run(task, seed, rows):
    *_, ytr = text_data(task, BACKBONE, seed)[:3]
    y = np.asarray(ytr)
    C = int(y.max()) + 1
    parts = drop_small_clients(dirichlet_partition(y, N, ALPHA, C, seed))
    holders = np.zeros(C, dtype=int)
    for p_ in parts:
        holders += (np.bincount(y[np.asarray(p_)], minlength=C) > 0).astype(int)
    r = dict(task=task, C=C, seed=seed, n_clients=len(parts),
             holders_0=int((holders == 0).sum()),
             holders_1=int((holders == 1).sum()),
             holders_2=int((holders == 2).sum()),
             holders_3p=int((holders >= 3).sum()))
    rows.append(r)
    print(f"  {task} s{seed}: {len(parts)} clients, {C} classes. "
          f"0 holders {r['holders_0']}, 1 holder {r['holders_1']}, "
          f"2 holders {r['holders_2']}, 3+ {r['holders_3p']}", flush=True)


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
    cols = ["task", "C", "seed", "n_clients", "holders_0", "holders_1",
            "holders_2", "holders_3p"]
    out = results_dir(NAME) / "results.csv"
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {out}\n", flush=True)
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r[c]) for c in cols))
    print("\nmean over seeds")
    for t in tasks:
        v = [r for r in rows if r["task"] == t]
        if v:
            print(f"  {t:<12} C={v[0]['C']:<4} "
                  f"0 holders {np.mean([r['holders_0'] for r in v]):.1f}, "
                  f"1 holder {np.mean([r['holders_1'] for r in v]):.1f}, "
                  f"2 holders {np.mean([r['holders_2'] for r in v]):.1f}")


if __name__ == "__main__":
    main()
