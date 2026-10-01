#!/usr/bin/env python
"""The pooled reference on the text tasks: the same recipe on the union of the shards.

Everything except the partition is held fixed: the frozen backbone, the rank-8
adapter with its down-projection frozen, the head, the learning rate and the
batch size. The optimizer sees the whole training pool. Two step budgets:

  matched_per_client   K steps, what one client spends
  matched_total        N * K steps, what the federation spends in total

The paper's pooled column is matched_total.

Usage:
  python experiments/pooled_text.py --tasks ag_news trec dbpedia_14 banking77
"""
import argparse
import csv
import sys

import numpy as np

from heoft.common import DEVICE, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, text_data
from heoft.models import TextLoRA
from heoft.train import BACKBONE, BS, K, LR, R, evaluate_text, train_text

NAME = "pooled_text"
COLS = ["task", "C", "seed", "mode", "steps", "acc"]
N = 10


def run(task, seed, rows):
    print(f"\n=== {task} seed={seed} ===", flush=True)
    ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C = text_data(task, BACKBONE, seed)

    for tag, steps in (("matched_per_client", K), ("matched_total", N * K)):
        set_seed(seed)
        model = TextLoRA(BACKBONE, C, r=R, freeze_a=True).to(DEVICE)
        train_text(model, ids_tr, mask_tr, ytr, steps=steps, lr=LR, bs=BS)
        acc = float(evaluate_text(model, ids_te, mask_te, yte))
        rows.append(dict(task=task, C=C, seed=seed, mode=tag, steps=steps,
                         acc=round(acc, 4)))
        print(f"  >> {tag:<20} steps={steps:<5} acc={acc:.4f}", flush=True)
        del model
        empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=list(TEXT_TASKS),
                    choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    rows = []
    for t in args.tasks:
        for s in args.seeds:
            run(t, s, rows)

    out = results_dir(NAME) / "results.csv"
    hdr = not out.exists()
    with out.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        if hdr:
            wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {out}", flush=True)
    csv.DictWriter(sys.stdout, fieldnames=COLS).writerows(rows)

    print("\ntask,mode,acc_mean")
    for t in sorted({r["task"] for r in rows}):
        for m in ("matched_per_client", "matched_total"):
            a = [r["acc"] for r in rows if r["task"] == t and r["mode"] == m]
            if a:
                print(f"{t},{m},{np.mean(a):.4f}")


if __name__ == "__main__":
    main()
