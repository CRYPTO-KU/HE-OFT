"""Membership-inference helpers shared by the attack experiments.

The adversary works in the feature space of the frozen backbone. It computes
phi(x) itself, trains linear heads on cached features to build shadow or
surrogate federations, and scores a candidate by the rescaled logit of Carlini
et al. (IEEE S&P 2022). An attack is reported as its true-positive rate at
false-positive rates of 0.1 and 1 per cent, with the area under the ROC curve
for completeness.
"""
import numpy as np
import torch

from heoft.common import DEVICE
from heoft.models import mean_pool

# The defaults of the likelihood-ratio attack, shared by the attack on the
# shared head and the attack on the disclosed reference model.
TASKS = ["ag_news", "dbpedia_14", "banking77"]
SEEDS = [42, 43, 44]
SHADOWS = 64         # shadow federations per (task, seed)
N_CAND = 1000        # members drawn, and as many non-members
TARGET = -1          # the client whose data is attacked, -1 pools every client

# Local head training of a shadow or surrogate client.
K_STEPS, LR = 200, 5e-4

# numpy 2 renamed trapz to trapezoid. The two compute the same rule.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def features_of(model, ids, mask, bs=64):
    """phi(x) for every row: the backbone output the head consumes.

    For the rank-0 model this is the bare public backbone. For the rank-8 model
    the adapter is inside the backbone, so this is the feature map of the client
    whose adapter is loaded.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            i = ids[s:s + bs].to(DEVICE)
            m = mask[s:s + bs].to(DEVICE)
            h = model.backbone(input_ids=i, attention_mask=m).last_hidden_state
            out.append(mean_pool(h, m).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def train_head(F, y, C, theta0_W, theta0_b, steps=K_STEPS, lr=LR):
    """One client's local head training on cached features.

    AdamW from theta0, minibatches of 32 drawn by a generator seeded with 0.
    """
    W = torch.as_tensor(theta0_W, dtype=torch.float32).clone().requires_grad_(True)
    b = torch.as_tensor(theta0_b, dtype=torch.float32).clone().requires_grad_(True)
    Xt = torch.as_tensor(F, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.AdamW([W, b], lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(Xt)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(32, n),), generator=g)
        opt.zero_grad()
        lossf(Xt[idx] @ W.T + b, yt[idx]).backward()
        opt.step()
    return W.detach().numpy().astype(np.float64), b.detach().numpy().astype(np.float64)


def logit_stat(W, b, F, y):
    """Carlini's rescaled logit, log p_y - log(1 - p_y), in a stable form."""
    z = F @ W.T + b
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    py = np.clip(p[np.arange(len(y)), y], 1e-12, 1 - 1e-12)
    return np.log(py) - np.log1p(-py)


def roc_points(score, label):
    """TPR at 0.1 and 1 per cent FPR, and the AUC. A higher score means member."""
    o = np.argsort(-score)
    lab = label[o]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    P, N = lab.sum(), (~lab).sum()
    if P == 0 or N == 0:
        return 0.0, 0.0, 0.5
    tpr, fpr = tp / P, fp / N
    def at(t):
        k = np.searchsorted(fpr, t, side="right") - 1
        return float(tpr[k]) if k >= 0 else 0.0
    auc = float(_trapezoid(tpr, fpr))
    return at(0.001), at(0.01), auc
