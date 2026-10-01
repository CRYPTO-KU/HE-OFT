"""The stratified holdout and the global-prior estimator."""
import numpy as np
import torch

from heoft.selection import (estimate, global_prior, per_class_nk,
                             stratified_holdout)


def test_holdout_reserves_every_held_class():
    y = np.array([0] * 30 + [1] * 2 + [2] * 1)
    parts = [np.arange(len(y))]
    tr, va = stratified_holdout(parts, seed=0, y=y)
    assert set(np.unique(y[va[0]])) == {0, 1, 2}
    assert len(np.intersect1d(tr[0], va[0])) == 0
    assert len(tr[0]) + len(va[0]) == len(y)
    # The single example of class 2 is held out, so the client never trains on it.
    assert 2 not in set(y[tr[0]])


def test_global_prior_is_a_distribution():
    pg = global_prior([np.array([3, 1, 0]), np.array([1, 1, 4])])
    assert np.isclose(pg.sum(), 1.0)
    assert np.allclose(pg, np.array([4, 2, 4]) / 10)


def test_per_class_nk_counts():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    n, k = per_class_nk(logits, [0, 0, 1], C=2)
    assert n.tolist() == [2, 1] and k.tolist() == [1, 0]


def test_personal_estimate_with_zero_fill_never_exceeds_full_evidence():
    pg = np.array([0.5, 0.5])
    # Client 0 holds only class 0 and is perfect there. With fill="zero" its
    # unmeasured class counts as wrong, so the estimate is a lower bound.
    nk = [(np.array([4.0, 0.0]), np.array([4.0, 0.0]))]
    assert estimate(pg, nk, [1.0], [np.array([40, 0])], pooled=False) == 0.5
    assert estimate(pg, nk, [1.0], [np.array([40, 0])], pooled=True) == 0.5
