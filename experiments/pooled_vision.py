#!/usr/bin/env python
"""The pooled reference on the vision tasks: the same recipe on the union of the shards.

The vision counterpart of experiments/pooled_text.py, with the same two step
budgets. Every trained model is evaluated on two test draws, 2000 images (the
size the federated vision runs use by default) and 10000. The 2000-image draw
is not a prefix of the 10000-image draw, and the training pool is identical in
both loads.

Usage:
  python experiments/pooled_vision.py --datasets cifar100
"""
import argparse
import csv
import sys

import numpy as np

from heoft.common import DEVICE, empty_cache, results_dir, set_seed
from heoft.data import VISION_TASKS, load_vision
from heoft.models import ViTLoRA
from heoft.train import BS, K, LR, R, evaluate_vision, train_vision

NAME = "pooled_vision"
COLS = ["task", "C", "seed", "mode", "steps", "n_test", "acc"]
N = 10
TEST_SIZES = (2000, 10000)


def run(ds, seed, rows):
    print(f"\n=== {ds} seed={seed} ===", flush=True)
    Xtr, ytr, Xte_s, yte_s, C = load_vision(ds, max_test=TEST_SIZES[0], seed=seed)
    Xtr2, ytr2, Xte_f, yte_f, _ = load_vision(ds, max_test=TEST_SIZES[1], seed=seed)
    assert np.array_equal(ytr, ytr2), "training pool differs between the two loads"
    print(f"  train {len(ytr)}, test {len(yte_s)} and {len(yte_f)}, C={C}", flush=True)

    for tag, steps in (("matched_per_client", K), ("matched_total", N * K)):
        set_seed(seed)
        model = ViTLoRA(C, r=R, freeze_a=True).to(DEVICE)
        train_vision(model, Xtr, ytr, steps, LR, BS)
        for X, y in ((Xte_s, yte_s), (Xte_f, yte_f)):
            acc = float(evaluate_vision(model, X, y))
            rows.append(dict(task=ds, C=C, seed=seed, mode=tag, steps=steps,
                             n_test=len(y), acc=round(acc, 4)))
            print(f"  >> {tag:<20} steps={steps:<5} n_test={len(y):<6} "
                  f"acc={acc:.4f}", flush=True)
        del model
        empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=["cifar100"],
                    choices=list(VISION_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    rows = []
    for t in args.datasets:
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

    for t in args.datasets:
        for tag in ("matched_per_client", "matched_total"):
            for n in TEST_SIZES:
                v = [r["acc"] for r in rows if r["task"] == t and r["mode"] == tag
                     and r["n_test"] == n]
                if v:
                    print(f"  {t:<10} {tag:<20} n_test={n:<6} "
                          f"mean={np.mean(v):.4f} over {len(v)} seeds")


if __name__ == "__main__":
    main()
