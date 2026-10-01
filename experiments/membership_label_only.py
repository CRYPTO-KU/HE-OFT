#!/usr/bin/env python
"""One-shot label-only membership inference against the serving interface.

The attack is OSLO (Peng et al., 2024). It spends one query per candidate and
reads membership from the returned label alone. The serving interface admits it
in a strong form. A client encrypts its own feature vector, so a coalition may
submit any point of the feature space and not only the features of a real
input.

The adversary is a coalition of the first --coalition clients. It holds their
data and adapters, the public backbone and label-only answers. It does not hold
the head. The candidates are training examples (members) and held-out examples
(non-members) of the clients outside the coalition.

The attack, per arrangement:
  1. Train --surrogates heads, each on a random half (at least 32 examples) of
     the coalition's own pool, its training and held-out examples. No query.
  2. For a candidate (x, y), take the normalized sum over the surrogates of the
     unit direction that most reduces the margin of class y. For a linear head
     it is the difference between row y and the row of the nearest competitor.
  3. Choose one step size on up to 600 examples of the coalition's own pool,
     where the membership of every example in every surrogate is known. The step
     is the one that best separates the survival rate of held-in examples from
     that of held-out examples. Calibrating on the candidates would need the
     answer the attack looks for.
  4. Submit one query at the perturbed feature vector, and predict member if the
     label is still y. Ties are broken by the surrogates' survival, which costs
     no query.
A member sits further from the boundary than a non-member when the head fitted
it more tightly, so the single label carries the membership signal.

Arrangement A computes features with the bare backbone. Arrangement B computes
them with the adapter of the first coalition client.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200. The output is results.csv, one row per (task, seed,
arrangement), with the step chosen, its separation on the surrogates, the
queries spent (one per candidate), the true-positive rates at false-positive
rates of 0.1 and 1 per cent and the area under the ROC curve.

Usage:
  python experiments/membership_label_only.py
  python experiments/membership_label_only.py --tasks ag_news --seeds 42 --coalition 3
"""
import argparse

import numpy as np
import torch

from heoft.attacks.extraction import head_of
from heoft.attacks.membership import features_of, roc_points, train_head
from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, dirichlet_partition, text_data
from heoft.models import TextLoRA, load_trainable
from heoft.selection import drop_small_clients, stratified_holdout
from heoft.train import BACKBONE, R

NAME = "membership_label_only"
TASKS = ["ag_news", "dbpedia_14", "banking77"]
SEEDS = [42, 43, 44]
SURR = 32            # surrogate heads
N_CAND = 1000        # members drawn, and as many non-members
COALITION = 3        # clients the coalition holds
STEPS = np.linspace(0.0, 6.0, 25)


def margin_direction(W, y):
    """For a linear head, the direction that most reduces class y's margin.

    The boundary between y and a competitor c is where (w_y - w_c) . z + (b_y -
    b_c) is zero, so moving against w_y - w_c is the shortest way out of y's
    cell. Averaging over surrogates gives the coalition's best guess at it
    without ever seeing the served head.
    """
    d = W[y][None, :] - W                     # (C, dim)
    n = np.linalg.norm(d, axis=1)
    n[y] = np.inf
    c = int(np.argmin(n))                     # nearest competitor
    v = d[c]
    return v / (np.linalg.norm(v) + 1e-12), c


def run(task, seed, rows, cfg):
    SURR, N_CAND, COALITION = cfg.surrogates, cfg.candidates, cfg.coalition
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
    tr_parts, va_parts = stratified_holdout(parts, seed, y)
    n_cl = len(parts)
    if n_cl <= COALITION + 1:
        print(f"  [skip] {task} s{seed}: {n_cl} clients, too few to split",
              flush=True)
        return
    coal = list(range(COALITION))
    honest = list(range(COALITION, n_cl))

    # candidates come from the clients outside the coalition only, the data
    # the coalition does not hold
    members = np.concatenate([np.asarray(tr_parts[j]) for j in honest])
    nonmembers = np.concatenate([np.asarray(va_parts[j]) for j in honest])
    n = min(len(members), len(nonmembers), N_CAND)
    if n < 50:
        print(f"  [skip] {task} s{seed}: only {n} candidates", flush=True)
        return
    rng = np.random.default_rng(seed)
    members = rng.choice(members, n, replace=False)
    nonmembers = rng.choice(nonmembers, n, replace=False)
    cand = np.concatenate([members, nonmembers])
    label = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    print(f"  {task} s{seed}: coalition holds {len(coal)} of {n_cl} clients, "
          f"{n} members and {n} non-members from the other {len(honest)}",
          flush=True)

    for arrangement in ("A", "B"):
        tag = "r0" if arrangement == "A" else "r8"
        r = 0 if arrangement == "A" else R
        set_seed(seed)
        model = TextLoRA(BACKBONE, C, r=r, freeze_a=True).to(DEVICE)
        if arrangement == "B":
            load_trainable(model, art[f"states_{tag}"][coal[0]])
        F = features_of(model, ids_tr, mask_tr)
        del model
        empty_cache()

        W, b = head_of(art, arrangement)                 # the served head
        theta0 = art[f"theta0_{tag}"]
        hk = [k for k in theta0 if "head" in k]
        wk = [k for k in hk if theta0[k].ndim == 2][0]
        bk = [k for k in hk if theta0[k].ndim == 1][0]
        t0W = theta0[wk].double().numpy()
        t0b = theta0[bk].double().numpy()

        # step 1, surrogates from the coalition's own data, no queries spent
        srng = np.random.default_rng(seed * 31 + 5)
        pool = np.concatenate([np.concatenate([np.asarray(tr_parts[j]),
                                               np.asarray(va_parts[j])])
                               for j in coal])
        surr, surr_in = [], []
        for _ in range(SURR):
            idx = srng.choice(pool, max(32, len(pool) // 2), replace=False)
            surr.append(train_head(F[idx], y[idx], C, t0W, t0b))
            surr_in.append(np.isin(pool, idx))
        surr_in = np.array(surr_in)                     # (SURR, len(pool))
        print(f"    {task} s{seed} {arrangement}: {SURR} surrogates from "
              f"{len(pool)} coalition examples", flush=True)

        # step 2, one direction per candidate, chosen on the surrogates alone
        Fc, yc = F[cand], y[cand]
        dirs = np.zeros_like(Fc)
        for i in range(len(cand)):
            v = np.zeros(Fc.shape[1])
            for Ws, _ in surr:
                d, _ = margin_direction(Ws, int(yc[i]))
                v += d
            dirs[i] = v / (np.linalg.norm(v) + 1e-12)

        # step 3, one step size for every candidate, calibrated on the
        # coalition's own data, where it knows which examples each surrogate
        # trained on
        cal = srng.choice(len(pool), min(600, len(pool)), replace=False)
        Fcal, ycal, incal = F[pool[cal]], y[pool[cal]], surr_in[:, cal]
        dcal = np.zeros_like(Fcal)
        for i in range(len(cal)):
            v = np.zeros(Fcal.shape[1])
            for Ws, _ in surr:
                d, _ = margin_direction(Ws, int(ycal[i]))
                v += d
            dcal[i] = v / (np.linalg.norm(v) + 1e-12)
        best_t, best_sep = STEPS[1], -np.inf
        for t in STEPS[1:]:
            surv = np.array([((Fcal + t * dcal) @ Ws.T + bsv).argmax(1) == ycal
                             for Ws, bsv in surr])          # (SURR, len(cal))
            num_in = (surv & incal).sum(); den_in = incal.sum()
            num_out = (surv & ~incal).sum(); den_out = (~incal).sum()
            if den_in == 0 or den_out == 0:
                continue
            sep = float(num_in / den_in - num_out / den_out)
            if sep > best_sep:
                best_sep, best_t = sep, t
        if not np.isfinite(best_sep):
            print(f"  [skip] {task} s{seed} {arrangement}: no calibration signal",
                  flush=True)
            continue

        # step 4, one query per candidate against the served head
        z = (Fc + best_t * dirs) @ W.T + b
        still = (np.argmax(z, axis=1) == yc).astype(np.float64)
        # break ties with the surrogate-averaged survival, which costs no query
        aux = np.mean([((Fc + best_t * dirs) @ Ws.T + bsv).argmax(1) == yc
                       for Ws, bsv in surr], axis=0)
        score = still + 1e-3 * aux
        t1, t2, auc = roc_points(score, label)
        rows.append(dict(task=task, C=C, seed=seed, arrangement=arrangement,
                         coalition=len(coal), surrogates=SURR,
                         n_candidates=int(len(cand)), queries=int(len(cand)),
                         step=round(float(best_t), 3),
                         surrogate_sep=round(float(best_sep), 4),
                         tpr_at_0_1pct=round(t1, 5), tpr_at_1pct=round(t2, 5),
                         auc=round(auc, 4)))
        print(f"  {task} s{seed} {arrangement} OSLO step={best_t:.2f} "
              f"(surrogate separation {best_sep:+.4f}) "
              f"TPR@0.1%={t1:.4f} TPR@1%={t2:.4f} AUC={auc:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--surrogates", type=int, default=SURR, help="surrogate heads")
    ap.add_argument("--candidates", type=int, default=N_CAND,
                    help="members drawn, and as many non-members")
    ap.add_argument("--coalition", type=int, default=COALITION,
                    help="clients the coalition holds, the first ones")
    cfg = ap.parse_args()
    tasks, seeds = cfg.tasks, cfg.seeds
    rows = []
    for t in tasks:
        for s in seeds:
            run(t, s, rows, cfg)
    if not rows:
        raise SystemExit("no rows produced")
    cols = ["task", "C", "seed", "arrangement", "coalition", "surrogates",
            "n_candidates", "queries", "step", "surrogate_sep",
            "tpr_at_0_1pct", "tpr_at_1pct", "auc"]
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
