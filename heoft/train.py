"""Local fine-tuning, evaluation and logits, for the text and the vision model.

Every client starts from the same initialization theta0 and runs K steps of
AdamW on minibatches drawn by a generator with a fixed seed, so a client's
trajectory depends only on its data and theta0.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .common import DEVICE
from .data import prepare_images
from .models import load_trainable, trainable_state

# The recipe of every run in the paper.
BACKBONE = "roberta_base"
R = 8          # adapter rank
K = 200        # local steps per client
LR = 5e-4
BS = 32


def train_text(model, ids, mask, y, steps, lr, bs):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n = len(y); yt = torch.as_tensor(y, device=DEVICE)
    g = torch.Generator().manual_seed(0)
    steps = max(1, steps)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        lo = model(ids[idx].to(DEVICE), mask[idx].to(DEVICE))
        loss = F.cross_entropy(lo, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate_text(model, ids, mask, y, bs=256):
    model.eval()
    n = len(y); yt = torch.as_tensor(y); correct = 0
    for s in range(0, n, bs):
        lo = model(ids[s:s + bs].to(DEVICE), mask[s:s + bs].to(DEVICE))
        correct += (lo.argmax(1).cpu() == yt[s:s + bs]).sum().item()
    return correct / max(n, 1)


@torch.no_grad()
def logits_text(model, ids, mask, bs=256):
    model.eval()
    out = []
    for s in range(0, len(ids), bs):
        out.append(model(ids[s:s + bs].to(DEVICE),
                         mask[s:s + bs].to(DEVICE)).float().cpu())
    return torch.cat(out)


def train_vision(m, X, y, steps, lr, bs):
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    n = len(y); yt = torch.as_tensor(y, device=DEVICE)
    g = torch.Generator().manual_seed(0)
    for _ in range(max(1, steps)):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        loss = F.cross_entropy(m(prepare_images(X[idx])), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def evaluate_vision(m, X, y, bs=64):
    m.eval()
    n = len(y); yt = torch.as_tensor(y); c = 0
    for s in range(0, n, bs):
        c += (m(prepare_images(X[s:s + bs])).argmax(1).cpu() == yt[s:s + bs]).sum().item()
    return c / max(n, 1)


@torch.no_grad()
def logits_vision(model, X, bs=64):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        out.append(model(prepare_images(X[s:s + bs])).float().cpu())
    return torch.cat(out)


def train_clients_text(model, theta0, parts, ids_tr, mask_tr, ytr, C,
                       K=K, lr=LR, bs=BS):
    """Each client runs its own K-step trajectory from theta0.

    Returns the trained states, the sample weights n_j / n and the per-class
    counts n_{j,c} of every client.
    """
    states, ws, counts = [], [], []
    y = np.asarray(ytr)
    for j, idx in enumerate(parts):
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        load_trainable(model, theta0)
        train_text(model, ids_tr[idx], mask_tr[idx], y[idx.numpy()], K, lr, bs)
        states.append(trainable_state(model))
        ws.append(len(idx))
        counts.append(np.bincount(y[idx.numpy()], minlength=C))
        print(f"  client {j+1}/{len(parts)} done", flush=True)
    ws = np.asarray(ws, dtype=float)
    return states, ws / ws.sum(), counts


def train_clients_vision(model, theta0, parts, X, y, C, K=K, lr=LR, bs=BS):
    """The vision counterpart of train_clients_text."""
    states, ws, counts = [], [], []
    for j, idx in enumerate(parts):
        idx = np.asarray(idx)
        load_trainable(model, theta0)
        train_vision(model, X[idx], y[idx], K, lr, bs)
        states.append(trainable_state(model))
        ws.append(len(idx))
        counts.append(np.bincount(y[idx], minlength=C))
        print(f"  client {j+1}/{len(parts)} done", flush=True)
    ws = np.asarray(ws, dtype=float)
    return states, ws / ws.sum(), counts


def acc_of(logits, y):
    return float((logits.argmax(1).numpy() == np.asarray(y)).mean())


def balanced_acc_of(logits, y, C):
    """Per-class accuracy averaged over the classes present in y."""
    p = logits.argmax(1).numpy(); y = np.asarray(y)
    per = [(p[y == c] == c).mean() for c in range(C) if (y == c).sum() > 0]
    return float(np.mean(per)) if per else 0.0
