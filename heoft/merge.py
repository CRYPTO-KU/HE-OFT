"""The head merge, in plaintext.

Under the protocol the server forms the merge on ciphertexts: it adds the
clients' encrypted products n_{j,c} * Delta_j[c] and multiplies the sum by an
encrypted reciprocal of the per-class totals (fhe/). These functions compute the
same map in plaintext, which is what the accuracy experiments evaluate.
"""
import numpy as np
import torch

from .common import HEAD_KEYS


def merge_sample_weighted(theta0, deltas, w):
    """theta0 + sum_j w_j * Delta_j, tensor by tensor, with w_j = n_j / n."""
    out = {}
    for k in theta0:
        acc = torch.zeros_like(theta0[k])
        for i in range(len(deltas)):
            acc += w[i] * deltas[i][k]
        out[k] = theta0[k] + acc
    return out


def merge_coverage_weighted(theta0, deltas, w, class_counts, head_keys=HEAD_KEYS):
    """The shared head: row c is the n_{j,c}-weighted mean of the row-c displacements.

    Only the clients holding class c decide row c, and a row no client covers
    keeps its value in theta0. Every tensor outside the head takes the
    sample-weighted merge, which matters only for the disclosed reference model,
    because the protocol never merges the adapters.
    """
    out = merge_sample_weighted(theta0, deltas, w)
    counts = torch.as_tensor(np.stack(class_counts), dtype=torch.float32)  # (N, C)
    den = counts.sum(0).clamp_min(1e-9)                                    # (C,)
    for k in head_keys:
        if k not in theta0:
            continue
        num = torch.zeros_like(theta0[k])
        for i in range(len(deltas)):
            ci = counts[i].to(theta0[k].device)
            num += (ci.unsqueeze(-1) if theta0[k].dim() == 2 else ci) * deltas[i][k]
        d = den.to(theta0[k].device)
        out[k] = theta0[k] + num / (d.unsqueeze(-1) if theta0[k].dim() == 2 else d)
    return out
