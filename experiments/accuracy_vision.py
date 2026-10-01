#!/usr/bin/env python
"""Accuracy of every arrangement and every selection rule, on the vision tasks.

The protocol of experiments/accuracy_text.py on ViT-B/16: each client fine-tunes
a rank-8 adapter with its down-projection frozen and the head, and the federation
merges the heads only. The modes and their meaning are listed in
experiments/accuracy_text.py.

The default test set is 2000 images drawn at random. --max-test 10000 evaluates
on the full CIFAR-10 test set; the training pool does not change with it.
Each run writes its artifacts to
outputs/accuracy_vision/artifacts/<dataset>_N<N>_a<alpha>_K<K>_s<seed>.pt.

Usage:
  python experiments/accuracy_vision.py --datasets cifar100 --seeds 42 43 44
  python experiments/accuracy_vision.py --datasets cifar10 --clients 20 --alpha 0.04 --max-test 10000
"""
import argparse
import csv
import json
import sys

import numpy as np
import torch

from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import VISION_TASKS, dirichlet_partition, load_vision
from heoft.merge import merge_coverage_weighted
from heoft.models import ViTLoRA, is_head, load_trainable, trainable_state
from heoft.selection import (drop_small_clients, estimate, global_prior,
                             per_class_nk, stratified_holdout)
from heoft.train import (BS, LR, R, acc_of, balanced_acc_of, evaluate_vision,
                         logits_vision, train_clients_vision)

NAME = "accuracy_vision"
COLS = ["dataset", "C", "N", "alpha", "K", "max_test", "seed", "mode", "n",
        "acc_mean", "acc_min", "acc_max", "note"]


def record(rows, cfg, ds, C, seed, mode, accs, note=""):
    a = np.asarray(accs, dtype=float)
    rows.append(dict(dataset=ds, C=C, N=cfg.clients, alpha=cfg.alpha, K=cfg.steps,
                     max_test=cfg.max_test, seed=seed, mode=mode, n=len(a),
                     acc_mean=round(a.mean(), 4), acc_min=round(a.min(), 4),
                     acc_max=round(a.max(), 4), note=note))
    print(f"  >> {mode:<12} mean={a.mean():.4f} "
          f"[{a.min():.4f}, {a.max():.4f}]  n={len(a)}  {note}", flush=True)


def run_ds(ds, seed, rows, cfg):
    N, ALPHA, K = cfg.clients, cfg.alpha, cfg.steps
    print(f"\n=== {ds} seed={seed} ===", flush=True)
    Xtr, ytr, Xte, yte, C = load_vision(ds, max_test=cfg.max_test, seed=seed)
    parts = drop_small_clients(dirichlet_partition(ytr, N, ALPHA, C, seed))
    tr_parts, va_parts = stratified_holdout(parts, seed, ytr)

    def val_of(j):
        v = np.asarray(va_parts[j])
        return Xtr[v], ytr[v]

    art = dict(dataset=ds, seed=seed, C=C, N=N, alpha=ALPHA, K=K, r=R,
               va_parts=[np.asarray(v) for v in va_parts], yte=np.asarray(yte))

    # Pass 1, rank 8: each client trains its own adapter and head.
    set_seed(seed)
    model = ViTLoRA(C, r=R, freeze_a=True).to(DEVICE)
    theta0 = trainable_state(model)
    states, w, counts = train_clients_vision(model, theta0, tr_parts, Xtr, ytr, C,
                                             K=K, lr=LR, bs=BS)

    deltas = [{k: s[k] - theta0[k] for k in s} for s in states]
    agg = merge_coverage_weighted(theta0, deltas, w, counts)
    art.update(theta0_r8={k: v.cpu() for k, v in theta0.items()},
               states_r8=[{k: v.cpu() for k, v in st.items()} for st in states],
               w=w, counts=np.stack(counts))

    load_trainable(model, agg)              # the disclosed reference, never served
    art["logits_current_test"] = logits_vision(model, Xte)
    record(rows, cfg, ds, C, seed, "current",
           [acc_of(art["logits_current_test"], yte)], "agg adapter + agg head")

    accs = []
    for st in states:                       # each client alone
        load_trainable(model, st)
        accs.append(evaluate_vision(model, Xte, yte))
    record(rows, cfg, ds, C, seed, "local", accs, "own adapter + own head")

    B_test, B_val, B_bal, val_y = [], [], [], []   # B: own adapter, merged head
    art["logits_B_test"], art["logits_B_val"] = [], []
    for j, st in enumerate(states):
        load_trainable(model, {k: (agg[k] if is_head(k) else st[k]) for k in st})
        lt = logits_vision(model, Xte)
        vX, vy = val_of(j)
        lv = logits_vision(model, vX)
        art["logits_B_test"].append(lt); art["logits_B_val"].append(lv)
        val_y.append(vy)
        B_test.append(acc_of(lt, yte))
        B_val.append(acc_of(lv, vy))
        B_bal.append(balanced_acc_of(lv, vy, C))
    record(rows, cfg, ds, C, seed, "B_personal", B_test, "own adapter + agg head")

    del model, states, deltas
    empty_cache()

    # Pass 2, rank 0: each client trains a head only. A serves the merged head.
    set_seed(seed)
    m0 = ViTLoRA(C, r=0, freeze_a=True).to(DEVICE)
    t0 = trainable_state(m0)
    s0, w0, c0 = train_clients_vision(m0, t0, tr_parts, Xtr, ytr, C, K=K, lr=LR, bs=BS)
    d0 = [{k: st[k] - t0[k] for k in st} for st in s0]
    load_trainable(m0, merge_coverage_weighted(t0, d0, w0, c0))
    art.update(theta0_r0={k: v.cpu() for k, v in t0.items()},
               states_r0=[{k: v.cpu() for k, v in st.items()} for st in s0])

    art["logits_A_test"] = logits_vision(m0, Xte)
    A_test = acc_of(art["logits_A_test"], yte)
    A_val, A_bal, art["logits_A_val"] = [], [], []
    for j in range(len(parts)):
        vX, vy = val_of(j)
        lv = logits_vision(m0, vX)
        art["logits_A_val"].append(lv)
        A_val.append(acc_of(lv, vy))
        A_bal.append(balanced_acc_of(lv, vy, C))
    record(rows, cfg, ds, C, seed, "A_headonly", [A_test], "r=0, head only, federated")

    # Selection rules, as in experiments/accuracy_text.py.
    sel = [B_test[j] if B_val[j] >= A_val[j] else A_test for j in range(len(parts))]
    n_B = sum(1 for j in range(len(parts)) if B_val[j] >= A_val[j])
    record(rows, cfg, ds, C, seed, "sel_perclient", sel,
           f"per-client vote (biased), {n_B}/{len(parts)} chose B")

    for tag, vB, vA in (("sel_federated", B_val, A_val),
                        ("sel_fed_balanced", B_bal, A_bal)):
        sB = float(np.dot(w, vB)); sA = float(np.dot(w, vA))
        pick_B = sB >= sA
        out = B_test if pick_B else [A_test] * len(parts)
        record(rows, cfg, ds, C, seed, tag, out,
               f"federation picked {'B' if pick_B else 'A'} (B={sB:.4f} A={sA:.4f})")

    pg = global_prior(counts)
    nkA = [per_class_nk(art["logits_A_val"][j], val_y[j], C) for j in range(len(parts))]
    nkB = [per_class_nk(art["logits_B_val"][j], val_y[j], C) for j in range(len(parts))]
    eA = estimate(pg, nkA, w, counts, pooled=True)
    for tag, fill in (("sel_globalprior", "zero"), ("sel_gp_rarefill", "rare")):
        eB = estimate(pg, nkB, w, counts, pooled=False, fill=fill)
        pick_B = eB >= eA
        out = B_test if pick_B else [A_test] * len(parts)
        record(rows, cfg, ds, C, seed, tag, out,
               f"picked {'B' if pick_B else 'A'} (E_B={eB:.4f} E_A={eA:.4f}; "
               f"true B={np.mean(B_test):.4f} A={A_test:.4f})")
    art.update(pg=pg, E_A=eA,
               E_B_zero=estimate(pg, nkB, w, counts, pooled=False, fill="zero"),
               E_B_rare=estimate(pg, nkB, w, counts, pooled=False, fill="rare"),
               nkA=np.array(nkA), nkB=np.array(nkB),
               val_y=[np.asarray(v) for v in val_y])
    art.update(B_test=B_test, B_val=B_val, B_bal=B_bal,
               A_test=A_test, A_val=A_val, A_bal=A_bal)
    ap = artifacts_dir(NAME) / f"{ds}_N{N}_a{ALPHA}_K{K}_s{seed}.pt"
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
    ap.add_argument("--datasets", nargs="+", default=["cifar100"],
                    choices=list(VISION_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--clients", type=int, default=10, help="N")
    ap.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration")
    ap.add_argument("--steps", type=int, default=200, help="K, local steps per client")
    ap.add_argument("--max-test", type=int, default=2000, help="test images drawn")
    cfg = ap.parse_args()

    rows = []
    stem = (f"{'_'.join(cfg.datasets)}_N{cfg.clients}_a{cfg.alpha}"
            f"_K{cfg.steps}_t{cfg.max_test}")
    for ds in cfg.datasets:
        for sd in cfg.seeds:
            try:
                run_ds(ds, sd, rows, cfg)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {ds} s{sd}: {type(e).__name__}: {e}", flush=True)
            write_rows(rows, stem)

    csv.DictWriter(sys.stdout, fieldnames=COLS).writeheader()
    csv.DictWriter(sys.stdout, fieldnames=COLS).writerows(rows)


if __name__ == "__main__":
    main()
