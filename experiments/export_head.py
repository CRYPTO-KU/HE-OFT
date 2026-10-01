#!/usr/bin/env python
"""Export a shared head, real query features and their plaintext answers.

The Go code in fhe/ runs one real query through the encrypted serving path.
This script writes its input and the plaintext answers the encrypted path must
reproduce, as JSON that the Go standard library reads. Nothing here is
encrypted.

For each (task, seed, arrangement) it writes <task>_s<seed>_<arrangement>.json
with
  W, b           the shared head, the coverage-weighted merge rebuilt from the
                 artifact (heoft.attacks.extraction.head_of)
  queries        for a random sample of test examples, the features phi(x)
                 under the backbone the arrangement serves (the bare backbone
                 for A, the adapter of client --client for B), the logits
                 W phi(x) + b in float64, their argmax, the true label, and the
                 margin between the two largest logits, which the encrypted
                 argmax must resolve
  logit_abs_max  the largest absolute logit over the queries. The sign
                 approximation of the encrypted argmax works on [-1, 1], and a
                 public scale set from this value maps the logits into it.
  min_margin     the smallest margin over the queries
  plain_accuracy the plaintext accuracy on the queries
and the configuration (task, seed, arrangement, backbone, artifact, client, N,
alpha, K, r, C, d). Computing phi(x) needs the backbone, so a GPU helps.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200.

Usage:
  python experiments/export_head.py
  python experiments/export_head.py --tasks ag_news --seeds 42 --arrangements A
"""
import argparse
import json

import numpy as np
import torch

from heoft.attacks.extraction import head_of
from heoft.attacks.membership import features_of
from heoft.common import DEVICE, artifacts_dir, empty_cache, results_dir, set_seed
from heoft.data import TEXT_TASKS, text_data
from heoft.models import TextLoRA, load_trainable
from heoft.train import BACKBONE, R

NAME = "export_head"
TASKS = ["ag_news"]
SEEDS = [42]
NQ = 16              # queries per export
ARRS = ["A", "B"]
CLIENT = 0           # the client whose adapter serves arrangement B


def export(task, seed, arrangement, cfg):
    NQ, CLIENT = cfg.queries, cfg.client
    path = artifacts_dir("accuracy_text") / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        print(f"  [skip] no artifact for {task} s{seed}", flush=True)
        return
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])

    ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C2 = text_data(task, BACKBONE, seed)
    assert C2 == C, (C2, C)
    yte = np.asarray(yte)

    tag = "r0" if arrangement == "A" else "r8"
    r = 0 if arrangement == "A" else R
    set_seed(seed)
    model = TextLoRA(BACKBONE, C, r=r, freeze_a=True).to(DEVICE)
    if arrangement == "B":
        load_trainable(model, art[f"states_{tag}"][CLIENT])

    rng = np.random.default_rng(seed * 100 + 1)
    q = rng.choice(len(yte), size=min(NQ, len(yte)), replace=False)
    q = np.sort(q)
    F = features_of(model, ids_te[torch.as_tensor(q, dtype=torch.long)],
                    mask_te[torch.as_tensor(q, dtype=torch.long)])
    del model
    empty_cache()

    W, b = head_of(art, arrangement)
    logits = F @ W.T + b
    pred = np.argmax(logits, axis=1)
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    queries = []
    for i in range(len(q)):
        queries.append(dict(
            test_index=int(q[i]),
            true_label=int(yte[q[i]]),
            plain_label=int(pred[i]),
            margin=float(margin[i]),
            features=[float(v) for v in F[i]],
            logits=[float(v) for v in logits[i]],
        ))

    out = dict(
        task=task, seed=seed, arrangement=arrangement, backbone=BACKBONE,
        artifact=path.name, client=(CLIENT if arrangement == "B" else -1),
        N=int(art.get("N", 10)), alpha=float(art.get("alpha", 0.1)),
        K=int(art.get("K", 200)), r=r, C=C, d=int(W.shape[1]),
        W=[[float(v) for v in row] for row in W],
        b=[float(v) for v in b],
        logit_abs_max=float(np.abs(logits).max()),
        min_margin=float(margin.min()),
        plain_accuracy=float((pred == yte[q]).mean()),
        queries=queries,
    )
    dst = results_dir(NAME) / f"{task}_s{seed}_{arrangement}.json"
    dst.write_text(json.dumps(out))
    print(f"  {task} s{seed} {arrangement}: C={C} d={W.shape[1]} "
          f"{len(queries)} queries, |logit|max={out['logit_abs_max']:.4f}, "
          f"min margin={out['min_margin']:.4f}, plaintext acc "
          f"{out['plain_accuracy']:.3f} -> {dst} "
          f"({dst.stat().st_size/1e6:.2f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=list(TEXT_TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--arrangements", nargs="+", default=ARRS, choices=["A", "B"])
    ap.add_argument("--queries", type=int, default=NQ, help="queries per export")
    ap.add_argument("--client", type=int, default=CLIENT,
                    help="client whose adapter serves arrangement B")
    cfg = ap.parse_args()
    tasks, seeds = cfg.tasks, cfg.seeds
    for task in tasks:
        for seed in seeds:
            for arrangement in cfg.arrangements:
                export(task, seed, arrangement, cfg)


if __name__ == "__main__":
    main()
