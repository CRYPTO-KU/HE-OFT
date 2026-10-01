#!/usr/bin/env python
"""Whether the cost of extracting a linear head tracks its parameter count.

The budget is set proportional to the parameter count, q = m * C * d, for m in
0.25, 0.5, 1, 2 and 4. If the cost tracks the parameter count, fidelity depends
on m alone and the curves for different C coincide. The test checks this
directly and fits no law.

The head is synthetic, because extraction attacks the linear map and not the
task. W is Gaussian with variance 1/d, each row scaled by a factor drawn
uniformly from [0.7, 1.3] to give the unequal row norms of a trained head, and
the bias is Gaussian with standard deviation 0.05. The feature dimension is
d = 768, C is 4, 16, 64 or 256, and each C runs under seeds 42 and 43.

The adversary submits random queries drawn from N(0, I/d), receives the labels,
and fits a multinomial logistic regression by L-BFGS. Fidelity is the agreement
of the copy with the head on 20000 held-out points of the same distribution,
and majority is the share of the largest class among them.

The output is results.csv, one row per (C, seed, m), and a table of fidelity
against m, one column per C, on standard output.

Usage:
  python experiments/extraction_scale.py
"""
import argparse

import numpy as np
import torch

from heoft.common import results_dir

NAME = "extraction_scale"
D = 768
CLASSES = [4, 16, 64, 256]
MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_EVAL = 20000
SEEDS = [42, 43]


def make_head(C, d, rng):
    """A linear map with the row-norm spread a trained head shows."""
    W = rng.normal(0, 1.0 / np.sqrt(d), size=(C, d))
    W *= rng.uniform(0.7, 1.3, size=(C, 1))     # unequal row norms
    b = rng.normal(0, 0.05, size=C)
    return W, b


def fit_linear(X, y, C, steps=300):
    """Multinomial logistic regression fitted by L-BFGS, capped at 300 iterations."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    lin = torch.nn.Linear(Xt.shape[1], C)
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=steps,
                            history_size=10, line_search_fn="strong_wolfe")
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(lin(Xt), yt)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:                                            # noqa: BLE001
        pass
    return (lin.weight.detach().numpy().astype(np.float64),
            lin.bias.detach().numpy().astype(np.float64))


def run(C, seed, rows):
    rng = np.random.default_rng(seed * 1000 + C)
    W, b = make_head(C, D, rng)
    scale = 1.0 / np.sqrt(D)
    Xe = rng.normal(0, scale, size=(N_EVAL, D))
    ytrue = np.argmax(Xe @ W.T + b, axis=1)
    majority = float(np.bincount(ytrue, minlength=C).max() / len(ytrue))

    for m in MULTS:
        nq = int(m * C * D)
        if nq < 50:
            continue
        X = rng.normal(0, scale, size=(nq, D))
        y = np.argmax(X @ W.T + b, axis=1)
        if len(np.unique(y)) < 2:
            continue
        Wh, bh = fit_linear(X, y, C)
        fid = float((np.argmax(Xe @ Wh.T + bh, axis=1) == ytrue).mean())
        rows.append(dict(C=C, d=D, seed=seed, params=C * D, mult=m,
                         queries=nq, fidelity=round(fid, 4),
                         majority=round(majority, 4)))
        print(f"  C={C:<5} m={m:<5} q={nq:<9} fid={fid:.4f} "
              f"(majority {majority:.3f})", flush=True)
        del X, y


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    rows = []
    for C in CLASSES:
        for s in SEEDS:
            run(C, s, rows)

    cols = ["C", "d", "seed", "params", "mult", "queries", "fidelity", "majority"]
    out = results_dir(NAME) / "results.csv"
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)\n", flush=True)

    # fidelity against m, one column per C
    print("COLLAPSE TEST: fidelity against queries per parameter")
    hdr = "m".ljust(7)
    for C in CLASSES:
        hdr += ("C=" + str(C)).rjust(10)
    print(hdr)
    for m in MULTS:
        line = str(m).ljust(7)
        for C in CLASSES:
            v = [r["fidelity"] for r in rows if r["C"] == C and r["mult"] == m]
            line += (("%.3f" % np.mean(v)) if v else "-").rjust(10)
        print(line)
    print("\nIf the columns agree, cost tracks the parameter count.")


if __name__ == "__main__":
    main()
