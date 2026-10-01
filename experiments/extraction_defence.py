#!/usr/bin/env python
"""What the answer format and noise on the answer do to extraction of the shared head.

The study rebuilds the shared head of arrangement A from the artifact and
attacks it directly, with no backbone and no GPU. Four interfaces:
  label          the argmax only, as the protocol specifies
  probs          the softmax of the logits
  label + noise  the argmax through randomised response with parameter eps
  probs + noise  the logits clipped to [-B, B] and perturbed by the Gaussian
                 mechanism at (eps, 1e-5), then the softmax
B is the 99th percentile of the absolute test logits. The Gaussian mechanism
takes the L2 sensitivity 2B sqrt(C).

For each interface and each eps it reports
  fidelity  the agreement of the extracted copy with the head on 20000
            held-out feature vectors, the standard extraction metric
  task_acc  the accuracy on the real test set of the head behind the same
            mechanism, applied to the stored test logits
The two columns are read together, since a mechanism can lower fidelity by
lowering accuracy.

The adversary is a participating client. It submits arbitrary feature vectors
and is charged one query per answer. The synthetic query distribution is scaled
so that its logits have the standard deviation of the real test logits, so one
clipping range serves both the attack and the accuracy measurement. Label
answers are fitted by multinomial logistic regression. Probability answers
expose the logits up to a per-query additive constant and are fitted by least
squares.

The input is the artifact experiments/accuracy_text.py writes for (task, seed)
at N=10, alpha=0.1, K=200. The output is results.csv, one row per (task, seed,
access, eps, budget), where eps is inf for the interfaces without noise.

Usage:
  python experiments/extraction_defence.py
  python experiments/extraction_defence.py --tasks ag_news --seeds 42
"""
import argparse

import numpy as np
import torch

from heoft.common import artifacts_dir, results_dir
from heoft.data import TEXT_TASKS

NAME = "extraction_defence"
BUDGETS = [1000, 5000, 20000, 50000, 100000]
EPSILONS = [None, 8.0, 4.0, 2.0, 1.0, 0.5]     # None = no perturbation
N_EVAL = 20000
DELTA = 1e-5
SEEDS = [42, 43, 44]
TASKS = ["ag_news", "dbpedia_14", "banking77"]


# --------------------------------------------------------------------------
# the target


def head_of(art, arrangement):
    """The shared head (W, b) of one arrangement, the coverage-weighted merge.

    The same map as heoft.attacks.extraction.head_of, without its check for a
    missing head weight.
    """
    tag = "r0" if arrangement == "A" else "r8"
    theta0, states = art[f"theta0_{tag}"], art[f"states_{tag}"]
    counts = np.asarray(art["counts"], dtype=np.float64)

    hk = [k for k in theta0 if "head" in k or "classifier" in k or "score" in k]
    wk = [k for k in hk if theta0[k].ndim == 2][0]
    bl = [k for k in hk if theta0[k].ndim == 1]
    bk = bl[0] if bl else None

    den = torch.as_tensor(np.where(counts.sum(0) > 0, counts.sum(0), 1.0))
    W = theta0[wk].double().clone()
    num = torch.zeros_like(W)
    for j, s in enumerate(states):
        g = torch.as_tensor(counts[j], dtype=torch.float64).unsqueeze(1)
        num += g * (s[wk].double() - theta0[wk].double())
    W = W + num / den.unsqueeze(1)

    if bk is None:
        b = torch.zeros(W.shape[0], dtype=torch.float64)
    else:
        b = theta0[bk].double().clone()
        nb = torch.zeros_like(b)
        for j, s in enumerate(states):
            nb += torch.as_tensor(counts[j], dtype=torch.float64) * \
                  (s[bk].double() - theta0[bk].double())
        b = b + nb / den
    return W.numpy(), b.numpy()


# --------------------------------------------------------------------------
# the mechanisms


def gaussian_sigma(eps, sens, delta=DELTA):
    """Classical Gaussian mechanism calibration."""
    return sens * np.sqrt(2.0 * np.log(1.25 / delta)) / eps


class Oracle:
    """The served head behind a given answer format and perturbation.

    access='label'  returns argmax, optionally through randomised response,
                    which is a standard eps-local-DP randomiser on the answer.
    access='probs'  returns softmax of the logits, optionally clipped to
                    [-B, B] and perturbed by the Gaussian mechanism.
    """

    def __init__(self, W, b, access, eps, rng, clip):
        self.W, self.b, self.access, self.eps = W, b, access, eps
        self.rng, self.clip, self.n = rng, clip, 0
        self.C = W.shape[0]
        if access == "probs" and eps is not None:
            # changing the answer vector moves it by at most 2B in each of C
            # coordinates, so the L2 sensitivity is 2B*sqrt(C)
            self.sigma = gaussian_sigma(eps, 2.0 * clip * np.sqrt(self.C))

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n += len(X)
        logits = X @ self.W.T + self.b
        if self.access == "label":
            y = np.argmax(logits, axis=1)
            if self.eps is not None:
                keep = np.exp(self.eps) / (np.exp(self.eps) + self.C - 1)
                flip = self.rng.random(len(y)) > keep
                if flip.any():
                    rnd = self.rng.integers(0, self.C - 1, size=int(flip.sum()))
                    yf = y[flip]
                    y[flip] = rnd + (rnd >= yf)     # uniform over the others
            return y
        z = np.clip(logits, -self.clip, self.clip)
        if self.eps is not None:
            z = z + self.rng.normal(0, self.sigma, size=z.shape)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# the attacks


def fit_from_labels(X, y, C):
    """Multinomial logistic regression fitted by L-BFGS, capped at 400 iterations."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    lin = torch.nn.Linear(Xt.shape[1], C)
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=400,
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


def fit_from_probs(X, P):
    """Probabilities expose the logits up to a per-query additive constant,
    so the head follows from an ordinary least-squares solve."""
    L = np.log(np.clip(P, 1e-12, None))
    L = L - L.mean(axis=1, keepdims=True)          # remove the free constant
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, L, rcond=None)
    return sol[:-1].T, sol[-1]


# --------------------------------------------------------------------------


def run(task, seed, rows):
    path = artifacts_dir("accuracy_text") / f"{task}_N10_a0.1_K200_s{seed}.pt"
    if not path.exists():
        print(f"  [skip] no artifact for {task} s{seed}", flush=True)
        return
    art = torch.load(path, map_location="cpu", weights_only=False)
    C = int(art["C"])
    rng = np.random.default_rng(seed)
    W, b = head_of(art, "A")
    d = W.shape[1]

    # real test logits, for the accuracy side of the trade-off
    real = art.get("logits_A_test")
    real = np.asarray(real, dtype=np.float64) if real is not None else None
    yte = np.asarray(art["yte"])

    # Match the synthetic query distribution to the real one in logit space.
    # Otherwise the attack and the accuracy measurement sit at different
    # operating points, one clipping range cannot serve both, and a noise
    # level calibrated on one has no meaning on the other.
    scale = 1.0 / np.sqrt(d)
    if real is not None:
        probe = rng.normal(0, scale, size=(4096, d)) @ W.T + b
        s_syn = float(probe.std())
        s_real = float(real.std())
        if s_syn > 0:
            scale *= s_real / s_syn
    clip = float(np.percentile(np.abs(real), 99)) if real is not None \
        else float(np.percentile(np.abs(rng.normal(0, scale, (4096, d)) @ W.T + b), 99))

    Xe = rng.normal(0, scale, size=(N_EVAL, d))
    ytrue = np.argmax(Xe @ W.T + b, axis=1)

    for access in ("label", "probs"):
        for eps in EPSILONS:
            # ---- accuracy under the mechanism, the cost to an honest querier ----
            if real is None:
                acc = float("nan")
            else:
                if access == "label":
                    yhat = np.argmax(real, axis=1)    # argmax is clip-invariant
                    if eps is not None:
                        keep = np.exp(eps) / (np.exp(eps) + C - 1)
                        r2 = np.random.default_rng(seed + 2)
                        flip = r2.random(len(yhat)) > keep
                        rnd = r2.integers(0, C - 1, size=int(flip.sum()))
                        yf = yhat[flip]
                        yhat[flip] = rnd + (rnd >= yf)
                else:
                    lg = np.clip(real, -clip, clip)
                    if eps is not None:
                        sig = gaussian_sigma(eps, 2.0 * clip * np.sqrt(C))
                        lg = lg + np.random.default_rng(seed + 2).normal(
                            0, sig, size=lg.shape)
                    yhat = np.argmax(lg, axis=1)
                acc = float((yhat == yte).mean())

            # ---- extraction ----
            for nq in BUDGETS:
                orc = Oracle(W, b, access, eps, rng, clip)
                X = rng.normal(0, scale, size=(nq, d))
                ans = orc(X)
                if access == "label":
                    if len(np.unique(ans)) < 2:
                        continue
                    Wh, bh = fit_from_labels(X, ans, C)
                else:
                    Wh, bh = fit_from_probs(X, ans)
                fid = float((np.argmax(Xe @ Wh.T + bh, axis=1) == ytrue).mean())
                rows.append(dict(task=task, C=C, seed=seed, access=access,
                                 eps=("inf" if eps is None else eps),
                                 queries=nq, fidelity=round(fid, 4),
                                 task_acc=round(acc, 4)))
                print(f"  {task:<11} s{seed} {access:<5} eps="
                      f"{'inf' if eps is None else eps:<5} q={nq:<7} "
                      f"fid={fid:.4f} acc={acc:.4f}", flush=True)


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
    cols = ["task", "C", "seed", "access", "eps", "queries", "fidelity", "task_acc"]
    out = results_dir(NAME) / "results.csv"
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
