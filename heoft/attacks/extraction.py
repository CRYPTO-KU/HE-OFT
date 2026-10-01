"""Extraction of the shared head from label-only answers.

The served head maps a feature vector phi to the label argmax_c (W phi + b)_c.
A client computes its query features itself, so it may submit any vector of the
feature space, and it receives one label per query. Extraction is therefore the
problem of learning a linear classifier from label queries. These helpers
rebuild the head from an accuracy artifact, answer queries against it, draw the
queries and fit the adversary's copy.
"""
import numpy as np
import torch

BISECT = 12          # halvings per boundary probe under the 'boundary' strategy


def head_of(art, arrangement):
    """The shared head (W, b) of one arrangement, rebuilt from an artifact.

    Row c is theta0 plus the n_{j,c}-weighted mean of the clients' row-c
    displacements, the coverage-weighted merge, computed in float64. A row no
    client covers keeps its value in theta0. Arrangement A reads the rank-0
    client states, arrangement B the rank-8 client states.
    """
    tag = "r0" if arrangement == "A" else "r8"
    theta0, states = art[f"theta0_{tag}"], art[f"states_{tag}"]
    counts = np.asarray(art["counts"], dtype=np.float64)

    hk = [k for k in theta0 if "head" in k or "classifier" in k or "score" in k]
    wk = [k for k in hk if theta0[k].ndim == 2]
    bk = [k for k in hk if theta0[k].ndim == 1]
    if not wk:
        raise SystemExit(f"no head weight among {list(theta0)}")
    wk, bk = wk[0], (bk[0] if bk else None)

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


class Oracle:
    """The served head behind the label-only interface.

    Returns one label per query and nothing else, and counts the queries.
    """

    def __init__(self, W, b):
        self.W, self.b, self.n = W, b, 0

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n += len(X)
        return np.argmax(X @ self.W.T + self.b, axis=1)


def fit_linear(X, y, C, steps=400, lr=0.5):
    """Multinomial logistic regression fitted by L-BFGS, the adversary's copy."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    lin = torch.nn.Linear(Xt.shape[1], C)
    opt = torch.optim.LBFGS(lin.parameters(), lr=lr, max_iter=steps,
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


def query_random(orc, n, d, rng, scale):
    """n queries drawn from N(0, scale^2 I_d), with the labels the oracle returns."""
    X = rng.normal(0, scale, size=(n, d))
    return X, orc(X)


def query_boundary(orc, n, d, rng, scale, C):
    """Half the budget on random queries, half bisecting toward a decision boundary.

    The random queries are visited in a random order. Each probe pairs a query
    with the first of the next 39 whose label differs and bisects the segment
    between them for BISECT steps. Every midpoint costs one query and joins
    the training set. Points close to a decision boundary carry the most
    information about a linear classifier.
    """
    half = max(2, n // 2)
    X0, y0 = query_random(orc, half, d, rng, scale)
    Xs, ys = [X0], [y0]

    spent, budget = 0, n - half
    order = rng.permutation(len(X0))
    i = 0
    while spent < budget and i + 1 < len(order):
        a, bnd = X0[order[i]], None
        ya = int(y0[order[i]])
        for j in range(i + 1, min(i + 40, len(order))):
            if int(y0[order[j]]) != ya:
                bnd = X0[order[j]]
                break
        i += 1
        if bnd is None:
            continue
        xa, xb = a.copy(), bnd.copy()
        for _ in range(min(BISECT, budget - spent)):
            xm = 0.5 * (xa + xb)
            ym = int(orc(xm)[0])          # one query, used both to steer and to keep
            spent += 1
            if ym == ya:
                xa = xm
            else:
                xb = xm
            Xs.append(xm[None, :])
            ys.append(np.array([ym]))
            if spent >= budget:
                break
    return np.vstack(Xs), np.concatenate(ys)
