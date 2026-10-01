#!/usr/bin/env python
"""Accuracy of every arrangement and every selection rule, on the text tasks.

Each client fine-tunes RoBERTa-base from the shared initialization theta0: a
rank-8 adapter with its down-projection frozen, and the head. The federation
merges the heads only (heoft.merge.merge_coverage_weighted). Every mode is
evaluated on the task's global test set.

  mode               what is served                                     paper
  current            merged adapter and merged head                     disclosed
  local              client j's own adapter and own head                alone
  B_personal         client j's own adapter and the merged head         personal arrangement
  A_headonly         the bare backbone and the merged head (rank 0)     shared head
  sel_globalprior    the arrangement the global-prior estimator picks   selected
  sel_gp_rarefill    the estimator with the rarest-class fill
  sel_federated      the arrangement a sample-weighted held-out vote picks
  sel_fed_balanced   the same vote, each class weighted equally
  sel_perclient      each client picks by its own held-out accuracy

A row reports the mean, minimum and maximum over the n served models (one per
client for B and local, one for A and current). Each run also writes the states,
counts and logits the attack experiments read, to
outputs/accuracy_text/artifacts/<task>_N<N>_a<alpha>_K<K>_s<seed>.pt.

Usage:
  python experiments/accuracy_text.py --tasks ag_news trec --seeds 42 43 44
  python experiments/accuracy_text.py --tasks dbpedia_14 --alpha 0.05
"""
import argparse
import csv
import json
import sys

import numpy as np
import torch

from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, dirichlet_partition, text_data
from heoft.merge import merge_coverage_weighted
from heoft.models import TextLoRA, is_head, load_trainable, trainable_state
from heoft.selection import (drop_small_clients, estimate, global_prior,
                             per_class_nk, stratified_holdout)
from heoft.train import (BACKBONE, BS, LR, R, acc_of, balanced_acc_of,
                         evaluate_text, logits_text, train_clients_text)

NAME = "accuracy_text"
COLS = ["task", "C", "N", "alpha", "K", "seed", "mode", "n",
        "acc_mean", "acc_min", "acc_max", "note"]


def record(rows, cfg, task, C, seed, mode, accs, note=""):
    a = np.asarray(accs, dtype=float)
    rows.append(dict(task=task, C=C, N=cfg.clients, alpha=cfg.alpha, K=cfg.steps,
                     seed=seed, mode=mode, n=len(a),
                     acc_mean=round(a.mean(), 4), acc_min=round(a.min(), 4),
                     acc_max=round(a.max(), 4), note=note))
    print(f"  >> {mode:<12} mean={a.mean():.4f} "
          f"[{a.min():.4f}, {a.max():.4f}]  n={len(a)}  {note}", flush=True)


def run_task(task, seed, rows, cfg):
    N, ALPHA, K = cfg.clients, cfg.alpha, cfg.steps
    print(f"\n=== {task} seed={seed} ===", flush=True)
    ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C = text_data(task, BACKBONE, seed)
    parts = drop_small_clients(dirichlet_partition(ytr, N, ALPHA, C, seed))
    tr_parts, va_parts = stratified_holdout(parts, seed, ytr)
    y = np.asarray(ytr)

    def val_ids(j):
        v = torch.as_tensor(va_parts[j], dtype=torch.long)
        return ids_tr[v], mask_tr[v], y[v.numpy()]

    art = dict(task=task, seed=seed, C=C, N=N, alpha=ALPHA, K=K, r=R,
               va_parts=[np.asarray(v) for v in va_parts], yte=np.asarray(yte))

    # Pass 1, rank 8: each client trains its own adapter and head.
    set_seed(seed)
    model = TextLoRA(BACKBONE, C, r=R, freeze_a=True).to(DEVICE)
    theta0 = trainable_state(model)
    states, w, counts = train_clients_text(model, theta0, tr_parts, ids_tr, mask_tr,
                                           ytr, C, K=K, lr=LR, bs=BS)

    deltas = [{k: s[k] - theta0[k] for k in s} for s in states]
    agg = merge_coverage_weighted(theta0, deltas, w, counts)
    art.update(theta0_r8={k: v.cpu() for k, v in theta0.items()},
               states_r8=[{k: v.cpu() for k, v in s.items()} for s in states],
               w=w, counts=np.stack(counts))

    load_trainable(model, agg)              # the disclosed reference, never served
    art["logits_current_test"] = logits_text(model, ids_te, mask_te)
    record(rows, cfg, task, C, seed, "current",
           [acc_of(art["logits_current_test"], yte)], "agg adapter + agg head")

    accs = []
    for s in states:                        # each client alone
        load_trainable(model, s)
        accs.append(evaluate_text(model, ids_te, mask_te, yte))
    record(rows, cfg, task, C, seed, "local", accs, "own adapter + own head")

    B_test, B_val, B_bal, val_y = [], [], [], []   # B: own adapter, merged head
    art["logits_B_test"], art["logits_B_val"] = [], []
    for j, s in enumerate(states):
        load_trainable(model, {k: (agg[k] if is_head(k) else s[k]) for k in s})
        lt = logits_text(model, ids_te, mask_te)
        vi, vm_, vy = val_ids(j)
        lv = logits_text(model, vi, vm_)
        art["logits_B_test"].append(lt); art["logits_B_val"].append(lv)
        val_y.append(vy)
        B_test.append(acc_of(lt, yte))
        B_val.append(acc_of(lv, vy))
        B_bal.append(balanced_acc_of(lv, vy, C))
    record(rows, cfg, task, C, seed, "B_personal", B_test, "own adapter + agg head")

    del model, states, deltas
    empty_cache()

    # Pass 2, rank 0: each client trains a head only. A serves the merged head.
    set_seed(seed)
    m0 = TextLoRA(BACKBONE, C, r=0, freeze_a=True).to(DEVICE)
    t0 = trainable_state(m0)
    s0, w0, c0 = train_clients_text(m0, t0, tr_parts, ids_tr, mask_tr, ytr, C,
                                    K=K, lr=LR, bs=BS)
    d0 = [{k: s[k] - t0[k] for k in s} for s in s0]
    load_trainable(m0, merge_coverage_weighted(t0, d0, w0, c0))
    art.update(theta0_r0={k: v.cpu() for k, v in t0.items()},
               states_r0=[{k: v.cpu() for k, v in s.items()} for s in s0])

    art["logits_A_test"] = logits_text(m0, ids_te, mask_te)
    A_test = acc_of(art["logits_A_test"], yte)
    A_val, A_bal, art["logits_A_val"] = [], [], []
    for j in range(len(parts)):
        vi, vm_, vy = val_ids(j)
        lv = logits_text(m0, vi, vm_)
        art["logits_A_val"].append(lv)
        A_val.append(acc_of(lv, vy))
        A_bal.append(balanced_acc_of(lv, vy, C))
    record(rows, cfg, task, C, seed, "A_headonly", [A_test], "r=0, head only, federated")

    # Selection rules.
    # Per client, by plain held-out accuracy. A client's held-out set follows
    # its own label skew, so this rule favors the personal arrangement.
    sel = [B_test[j] if B_val[j] >= A_val[j] else A_test for j in range(len(parts))]
    n_B = sum(1 for j in range(len(parts)) if B_val[j] >= A_val[j])
    record(rows, cfg, task, C, seed, "sel_perclient", sel,
           f"per-client vote (biased), {n_B}/{len(parts)} chose B")

    # Federation-wide, sample-weighted: one choice for every client.
    for tag, vB, vA in (("sel_federated", B_val, A_val),
                        ("sel_fed_balanced", B_bal, A_bal)):
        sB = float(np.dot(w, vB)); sA = float(np.dot(w, vA))
        pick_B = sB >= sA
        out = B_test if pick_B else [A_test] * len(parts)
        record(rows, cfg, task, C, seed, tag, out,
               f"federation picked {'B' if pick_B else 'A'} (B={sB:.4f} A={sA:.4f})")

    # The global-prior estimator. A pools its per-class evidence across
    # clients; B cannot, so the classes a client does not hold take a fill.
    pg = global_prior(counts)
    nkA = [per_class_nk(art["logits_A_val"][j], val_y[j], C) for j in range(len(parts))]
    nkB = [per_class_nk(art["logits_B_val"][j], val_y[j], C) for j in range(len(parts))]
    eA = estimate(pg, nkA, w, counts, pooled=True)
    for tag, fill in (("sel_globalprior", "zero"), ("sel_gp_rarefill", "rare")):
        eB = estimate(pg, nkB, w, counts, pooled=False, fill=fill)
        pick_B = eB >= eA
        out = B_test if pick_B else [A_test] * len(parts)
        record(rows, cfg, task, C, seed, tag, out,
               f"picked {'B' if pick_B else 'A'} (E_B={eB:.4f} E_A={eA:.4f}; "
               f"true B={np.mean(B_test):.4f} A={A_test:.4f})")
    art.update(pg=pg, E_A=eA,
               E_B_zero=estimate(pg, nkB, w, counts, pooled=False, fill="zero"),
               E_B_rare=estimate(pg, nkB, w, counts, pooled=False, fill="rare"),
               nkA=np.array(nkA), nkB=np.array(nkB),
               val_y=[np.asarray(v) for v in val_y])
    art.update(B_test=B_test, B_val=B_val, B_bal=B_bal,
               A_test=A_test, A_val=A_val, A_bal=A_bal)
    ap = artifacts_dir(NAME) / f"{task}_N{N}_a{ALPHA}_K{K}_s{seed}.pt"
    torch.save(art, ap)
    print(f"  artifacts -> {ap} ({ap.stat().st_size/1e6:.1f} MB)", flush=True)

    del m0, s0, d0
    empty_cache()


def write_rows(rows, stem):
    out = results_dir(NAME)
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=2))
    with (out / f"{stem}.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=list(TEXT_TASKS),
                    choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--clients", type=int, default=10, help="N")
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration")
    ap.add_argument("--steps", type=int, default=200, help="K, local steps per client")
    cfg = ap.parse_args()

    rows = []
    stem = f"{'_'.join(cfg.tasks)}_N{cfg.clients}_a{cfg.alpha}_K{cfg.steps}"
    for t in cfg.tasks:
        for sd in cfg.seeds:
            try:
                run_task(t, sd, rows, cfg)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {t} s{sd}: {type(e).__name__}: {e}", flush=True)
            write_rows(rows, stem)

    csv.DictWriter(sys.stdout, fieldnames=COLS).writeheader()
    csv.DictWriter(sys.stdout, fieldnames=COLS).writerows(rows)


if __name__ == "__main__":
    main()
