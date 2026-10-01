#!/usr/bin/env python
"""Membership inference against the disclosed reference model.

The disclosed model is the plaintext reference that one-shot federated learning
without encryption would give every participant. Its adapter is the
sample-weighted merge of the clients' adapters and its head the
coverage-weighted merge of their heads, both in the clear. The protocol never
builds it. This experiment runs the attack of experiments/membership_extracted.py
against it, so that its rows read beside the truehead rows there.

The target is mode current of the accuracy experiments, rebuilt from the
rank-8 client states in the artifact, not retrained. Before it attacks, the
script checks three things.
  the partition recomputed from the data equals the artifact's held-out indices
  theta0 rebuilt under the seed equals the artifact's bit for bit, so the
      frozen down-projection, which the artifact does not store, is the same
  the test accuracy of the rebuilt model equals that of the artifact's logits

The shadows follow experiments/membership_extracted.py with one change. The
merged adapter moves the features, so a shadow client trains its adapter and
its head for K steps through the backbone, with the function that trained the
target, and the statistic is the rescaled logit of the whole shadow model. No
feature is cached. The candidate draw and the half-sampling of each shadow,
with their random streams, are those of experiments/membership_extracted.py,
so the candidate set and the membership of every candidate in shadow s are the
same there and here.

Task cifar100 runs the same attack on the vision pipeline, ViT-B/16 with the
artifact of experiments/accuracy_vision.py. The test set enters only the third
check. It is reloaded at the size the artifact stores, and its labels must
equal the artifact's. The loader draws the training images before the test
images, so the partition does not depend on the test size.

Each finished shadow is one file under shadows/<cell>/ in the output
directory. A rerun skips finished shadows and replays the random stream, so the
draws do not shift. With --shard i/n a job trains only the shadows s with
s mod n equal to i, so one cell can run as n jobs at once. The job that finds
every shadow present scores the cell. A run with fewer shadows uses the first
ones.

The output, in outputs/membership_disclosed/:
  cells/cell_<task>_N<N>_a<alpha>_K<K>_s<seed>_t<target>_c<ncand>_S<shadows>.json
  results.csv, rebuilt from every cell file, with the columns of
    experiments/membership_extracted.py, arrangement disclosed, and two surfaces
      disclosed  LiRA on the disclosed model's logits, the analogue of truehead
      gap        member if and only if the disclosed model is right. The
                 stratified held-out set biases it, as explained in
                 experiments/membership_extracted.py.

Usage:
  python experiments/membership_disclosed.py --tasks ag_news --seeds 42
  python experiments/membership_disclosed.py --tasks cifar100 --seeds 42 --shard 0/2
"""
import argparse
import csv
import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from heoft.attacks.membership import (N_CAND, SEEDS, SHADOWS, TARGET, TASKS,
                                      logit_stat, roc_points)
from heoft.common import (DEVICE, REPO_ROOT, RESULTS, artifacts_dir, empty_cache,
                          results_dir, set_seed)
from heoft.data import (TEXT_TASKS, VISION_TASKS, dirichlet_partition,
                        load_vision, text_data)
from heoft.merge import merge_coverage_weighted
from heoft.models import VIT, TextLoRA, ViTLoRA, load_trainable, trainable_state
from heoft.selection import drop_small_clients, stratified_holdout
from heoft.train import (BACKBONE, R, acc_of, logits_text, logits_vision,
                         train_clients_text, train_clients_vision)
from heoft.train import K as K_CLIENT

NAME = "membership_disclosed"
# The tracked accuracy records. The text record names its task column `task`,
# the vision record `dataset`.
RECORD = REPO_ROOT / "results" / "accuracy_text" / "results.csv"
VRECORD = REPO_ROOT / "results" / "accuracy_vision" / "results.csv"

COLS = ["task", "C", "d", "seed", "target", "arrangement", "surface",
        "queries", "shadows", "n_candidates", "tpr_at_0_1pct",
        "tpr_at_1pct", "auc"]


def parse_shard(spec):
    i, n = (int(x) for x in spec.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {spec!r} must be i/n with 0 <= i < n")
    return i, n


# ------------------------- as in experiments/membership_extracted.py ---
def candidates(y, tr_parts, va_parts, seed, TARGET, N_CAND):
    """The candidate draw of experiments/membership_extracted.py, unchanged."""
    if TARGET < 0:
        members = np.concatenate([np.asarray(p) for p in tr_parts])
        nonmembers = np.concatenate([np.asarray(p) for p in va_parts])
    else:
        if TARGET >= len(tr_parts):
            return None
        members = np.asarray(tr_parts[TARGET])
        nonmembers = np.asarray(va_parts[TARGET])
    n = min(len(members), len(nonmembers), N_CAND)
    if n < 50:
        return None
    pools = [np.concatenate([np.asarray(a), np.asarray(v)])
             for a, v in zip(tr_parts, va_parts)]
    rng = np.random.default_rng(seed)
    members = rng.choice(members, n, replace=False)
    nonmembers = rng.choice(nonmembers, n, replace=False)
    cand = np.concatenate([members, nonmembers])
    label = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    return cand, label, pools


def shadow_halves(pools, rng):
    """The half-sampling of membership_extracted.shadow_federation, unchanged."""
    halves = []
    for p in pools:
        take = rng.random(len(p)) < 0.5
        idx = p[take]
        if len(idx) < 4:
            idx = p[:max(4, len(p) // 2)]
        halves.append(idx)
    return halves


# ------------------------------------------------------------- statistics ---
def rescaled_logit(logits, y):
    """logit_stat on logits the model already produced (identity head)."""
    z = np.asarray(logits, dtype=np.float64)
    C = z.shape[1]
    return logit_stat(np.eye(C), np.zeros(C), z, np.asarray(y))


def lira_scores(stat_target, S, Min):
    """Per-candidate Gaussians over IN and OUT shadows, as in membership_extracted.

    S and Min are (shadows, candidates). Mean and population standard deviation
    plus 1e-6 on each side, the same estimator as there.
    """
    def moments(m):
        cnt = np.maximum(m.sum(0), 1)
        mu = (S * m).sum(0) / cnt
        sd = np.sqrt((((S - mu) ** 2) * m).sum(0) / cnt) + 1e-6
        return mu, sd
    mu_i, sd_i = moments(Min)
    mu_o, sd_o = moments(~Min)
    return ((-0.5 * ((stat_target - mu_i) / sd_i) ** 2 - np.log(sd_i))
            - (-0.5 * ((stat_target - mu_o) / sd_o) ** 2 - np.log(sd_o)))


# ------------------------------------------------------------- provenance ---
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def recorded_disclosed(task, seed, record=RECORD, key="task"):
    """The `current` accuracy in the tracked accuracy record, for the log only."""
    try:
        with record.open() as f:
            for r in csv.DictReader(f):
                if (r[key] == task and r["seed"] == str(seed)
                        and r["mode"] == "current"):
                    return float(r["acc_mean"])
    except Exception:  # noqa: BLE001
        return None
    return None


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def rebuild_csv():
    rows = []
    for f in sorted((results_dir(NAME) / "cells").glob("cell_*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    rows.sort(key=lambda r: (r["task"], r["seed"], r["shadows"], r["surface"]))
    lines = [",".join(COLS)] + [",".join(str(r[c]) for c in COLS) for r in rows]
    atomic_write(results_dir(NAME) / "results.csv", "\n".join(lines) + "\n")
    return len(rows)


# ------------------------------------------------------------- the worlds ---
# One namespace per modality, so run() reads the same for text and vision. Each
# field makes the call the accuracy experiment of that modality makes, with the
# same arguments.
def text_world(task, seed, art):
    """The text tasks as experiments/accuracy_text.py loads, trains and scores them."""
    ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C = text_data(task, BACKBONE, seed)
    return SimpleNamespace(
        C=C, y=np.asarray(ytr), yte=yte,
        build=lambda: TextLoRA(BACKBONE, C, r=R, freeze_a=True),
        train=lambda model, th0, parts, y: train_clients_text(
            model, th0, parts, ids_tr, mask_tr, y, C),
        logits_train=lambda model, idx: logits_text(model, ids_tr[idx], mask_tr[idx]),
        logits_test=lambda model: logits_text(model, ids_te, mask_te),
        record=RECORD, record_key="task", extra={})


def vision_world(task, seed, art):
    """CIFAR-100 as experiments/accuracy_vision.py loads, trains and scores it.

    The test set is reloaded at the artifact's size, so the accuracy check
    compares the same images. The training draw precedes the test draw in
    load_vision, so the training set does not depend on that size.
    """
    n_test = len(art["yte"])
    Xtr, ytr, Xte, yte, C = load_vision(task, max_test=n_test, seed=seed)
    if not np.array_equal(np.asarray(yte), np.asarray(art["yte"])):
        raise RuntimeError("the reloaded test set differs from the artifact's")
    print(f"  vision: {len(ytr)} training images, {n_test} test images "
          f"(the artifact's), backbone {VIT}", flush=True)
    return SimpleNamespace(
        C=C, y=np.asarray(ytr), yte=yte,
        build=lambda: ViTLoRA(C, r=R, freeze_a=True),
        train=lambda model, th0, parts, y: train_clients_vision(
            model, th0, parts, Xtr, y, C),
        logits_train=lambda model, idx: logits_vision(model, Xtr[idx]),
        logits_test=lambda model: logits_vision(model, Xte),
        record=VRECORD, record_key="dataset",
        extra=dict(backbone=VIT, n_test=n_test))


# -------------------------------------------------------------------- run ---
def run(task, seed, rows, cfg):
    SHADOWS, TARGET, N_CAND = cfg.shadows, cfg.target, cfg.candidates
    SHARD_I, SHARD_N = cfg.shard
    vision = task in VISION_TASKS
    artdir = artifacts_dir("accuracy_vision" if vision else "accuracy_text")
    path = artdir / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        raise RuntimeError(f"no artifact for {task} s{seed} under {artdir}")
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])
    N, ALPHA = int(art.get("N", 10)), float(art.get("alpha", 0.1))
    K = int(art.get("K", K_CLIENT))
    if K != K_CLIENT:
        raise RuntimeError(f"artifact K={K} but the client training "
                           f"runs {K_CLIENT} steps")
    stem = f"{task}_N{N}_a{ALPHA}_K{K}_s{seed}_t{TARGET}_c{N_CAND}"
    out = results_dir(NAME) / "cells" / f"cell_{stem}_S{SHADOWS}.json"
    if out.exists():
        cell = json.loads(out.read_text())
        rows.extend(cell["rows"])
        print(f"  skip (done): {out.name}", flush=True)
        return
    print(f"\n=== {task} seed={seed} artifact={path.name} shadows={SHADOWS} "
          f"shard={SHARD_I}/{SHARD_N} ===", flush=True)
    t_start = time.time()

    W = (vision_world if vision else text_world)(task, seed, art)
    assert W.C == C, (W.C, C)
    y = W.y
    parts = drop_small_clients(dirichlet_partition(y, N, ALPHA, C, seed))
    tr_parts, va_parts = stratified_holdout(parts, seed, y)
    art_va = art["va_parts"]
    if len(art_va) != len(va_parts) or not all(
            np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(art_va, va_parts)):
        raise RuntimeError("the recomputed partition differs from the artifact's")

    got = candidates(y, tr_parts, va_parts, seed, TARGET, N_CAND)
    if got is None:
        raise RuntimeError(f"too few candidates for target {TARGET}")
    cand, label, pools = got
    n = int(label.sum())
    ckdir = results_dir(NAME) / "shadows" / stem
    ckdir.mkdir(parents=True, exist_ok=True)
    cf = ckdir / "cand.npy"
    if cf.exists():
        if not np.array_equal(np.load(cf), cand):
            raise RuntimeError(f"candidate set differs from the one in {cf}")
    else:                                   # atomic, since shards may start together
        tmp = cf.with_name(cf.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, cand)
        os.replace(tmp, cf)
    print(f"  {n} members and {n} non-members over {len(parts)} clients", flush=True)

    # --- the target, rebuilt and checked ------------------------------------
    set_seed(seed)
    model = W.build().to(DEVICE)
    t0_art = {k: v.float() for k, v in art["theta0_r8"].items()}
    t0_new = trainable_state(model)
    if set(t0_new) != set(t0_art) or not all(
            torch.equal(t0_new[k].cpu(), t0_art[k]) for k in t0_art):
        raise RuntimeError("theta0 rebuilt under the seed differs from the artifact's, "
                           "so the frozen down-projection cannot be trusted")
    deltas = [{k: s[k].float() - t0_art[k] for k in s} for s in art["states_r8"]]
    load_trainable(model, merge_coverage_weighted(t0_art, deltas, art["w"], art["counts"]))
    del deltas
    acc_rebuilt = acc_of(W.logits_test(model), W.yte)
    acc_art = acc_of(art["logits_current_test"], W.yte)
    acc_rec = recorded_disclosed(task, seed, W.record, W.record_key)
    print(f"  disclosed model: rebuilt acc={acc_rebuilt:.4f}  artifact acc={acc_art:.4f}  "
          f"record acc={acc_rec}", flush=True)
    if abs(acc_rebuilt - acc_art) > 0.005:
        raise RuntimeError("the rebuilt disclosed model does not reproduce the artifact")
    z_target = W.logits_train(model, cand).double().numpy()
    st_target = rescaled_logit(z_target, y[cand])
    d = int(t0_art["head.weight"].shape[1])

    # --- shadow federations, one file each ------------------------------------
    th0 = {k: v.to(DEVICE) for k, v in t0_art.items()}
    srng = np.random.default_rng(seed * 1000 + 7)    # the stream membership_extracted uses
    shadow_walls = []
    for s in range(SHADOWS):
        halves = shadow_halves(pools, srng)            # always drawn, so resume replays
        f = ckdir / f"shadow_{s:03d}.npz"
        if f.exists() or s % SHARD_N != SHARD_I:
            continue
        t = time.time()
        inmask = np.zeros(len(y), dtype=bool)
        for h in halves:
            inmask[h] = True
        states, w, counts = W.train(model, th0, halves, y)
        deltas = [{k: st[k] - th0[k] for k in st} for st in states]
        load_trainable(model, merge_coverage_weighted(th0, deltas, w, counts))
        z = W.logits_train(model, cand).double().numpy()
        tmp = f.with_name(f.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, inmask=inmask[cand], stat=rescaled_logit(z, y[cand]))
        os.replace(tmp, f)
        del states, deltas
        empty_cache()
        shadow_walls.append(time.time() - t)
        print(f"    {task} s{seed} shadow {s + 1}/{SHADOWS} "
              f"({shadow_walls[-1]:.0f}s)", flush=True)

    files = [ckdir / f"shadow_{s:03d}.npz" for s in range(SHADOWS)]
    missing = sum(1 for f in files if not f.exists())
    if missing:
        print(f"  {missing}/{SHADOWS} shadows belong to other shards, "
              f"not scoring yet", flush=True)
        return
    loaded = [np.load(f) for f in files]
    S = np.stack([z["stat"] for z in loaded])
    Min = np.stack([z["inmask"] for z in loaded]).astype(bool)
    keep = (Min.sum(0) >= 2) & ((~Min).sum(0) >= 2)
    if keep.sum() < 20:
        raise RuntimeError(f"only {keep.sum()} candidates with both sides")

    cell_rows = []
    sc = lira_scores(st_target, S, Min)
    t1, t2, auc = roc_points(sc[keep], label[keep])
    cell_rows.append(dict(task=task, C=C, d=d, seed=seed, target=TARGET,
                          arrangement="disclosed", surface="disclosed", queries=0,
                          shadows=SHADOWS, n_candidates=int(keep.sum()),
                          tpr_at_0_1pct=round(t1, 5), tpr_at_1pct=round(t2, 5),
                          auc=round(auc, 4)))
    pred = z_target.argmax(1)
    g1, g2, gauc = roc_points((pred == y[cand]).astype(np.float64)[keep], label[keep])
    cell_rows.append(dict(task=task, C=C, d=d, seed=seed, target=TARGET,
                          arrangement="disclosed", surface="gap", queries=1,
                          shadows=0, n_candidates=int(keep.sum()),
                          tpr_at_0_1pct=round(g1, 5), tpr_at_1pct=round(g2, 5),
                          auc=round(gauc, 4)))
    for r in cell_rows:
        print(f"  {task} s{seed} disclosed {r['surface']:<10} "
              f"TPR@0.1%={r['tpr_at_0_1pct']:.4f} TPR@1%={r['tpr_at_1pct']:.4f} "
              f"AUC={r['auc']:.4f}", flush=True)

    cell = dict(task=task, seed=seed, N=N, alpha=ALPHA, K=K, target=TARGET,
                n_members=n, n_nonmembers=n, n_clients=len(parts), shadows=SHADOWS,
                artifact=path.name, acc_rebuilt=round(acc_rebuilt, 4),
                acc_artifact=round(acc_art, 4), acc_record=acc_rec,
                shadow_wall_s_this_job=[round(x, 1) for x in shadow_walls],
                wall_this_job_s=round(time.time() - t_start, 1),
                git=git_commit(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                torch=torch.__version__, env=os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                deterministic_algorithms=False)
    cell.update(W.extra)                    # empty for text
    cell["rows"] = cell_rows
    atomic_write(out, json.dumps(cell, indent=2))
    rows.extend(cell_rows)
    del model
    empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=TASKS,
                    choices=list(TEXT_TASKS) + list(VISION_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--shadows", type=int, default=SHADOWS,
                    help="shadow federations per (task, seed)")
    ap.add_argument("--candidates", type=int, default=N_CAND,
                    help="members drawn, and as many non-members")
    ap.add_argument("--target", type=int, default=TARGET,
                    help="client attacked, -1 pools every client")
    ap.add_argument("--shard", type=parse_shard, default="0/1",
                    help="i/n, train only the shadows s with s mod n = i")
    cfg = ap.parse_args()
    (results_dir(NAME) / "cells").mkdir(parents=True, exist_ok=True)
    head = RESULTS / "membership_extracted" / "results.csv"
    if head.exists():
        first = head.read_text().splitlines()[0].strip()
        assert first == ",".join(COLS), f"column mismatch with {head}: {first}"
    tasks, seeds = cfg.tasks, cfg.seeds
    print(f"[membership_disclosed] tasks={tasks} seeds={seeds} shadows={cfg.shadows} "
          f"ncand={cfg.candidates} target={cfg.target} "
          f"shard={cfg.shard[0]}/{cfg.shard[1]} K={K_CLIENT} "
          f"git={git_commit()}", flush=True)
    rows, failed = [], 0
    for t in tasks:
        for s in seeds:
            try:
                run(t, s, rows, cfg)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {t} s{s}: {type(e).__name__}: {e}", flush=True)
    if any((results_dir(NAME) / "cells").glob("cell_*.json")):
        print(f"\nwrote {results_dir(NAME) / 'results.csv'} ({rebuild_csv()} rows)",
              flush=True)
    print("\n" + ",".join(COLS))
    for r in rows:
        print(",".join(str(r[c]) for c in COLS))
    if failed:
        raise SystemExit(f"{failed} cell(s) failed")


if __name__ == "__main__":
    main()
