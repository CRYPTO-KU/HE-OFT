"""Properties of the coverage-weighted head merge that the paper relies on."""
import numpy as np
import torch

from heoft.merge import merge_coverage_weighted, merge_sample_weighted


def _setup(C=4, d=3, N=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta0 = {"head.weight": torch.randn(C, d, generator=g),
              "head.bias": torch.randn(C, generator=g),
              "adapter": torch.randn(5, generator=g)}
    deltas = [{k: torch.randn(v.shape, generator=g) for k, v in theta0.items()}
              for _ in range(N)]
    return theta0, deltas


def test_row_is_count_weighted_mean_of_holders():
    theta0, deltas = _setup()
    counts = [np.array([5, 0, 1, 0]), np.array([1, 0, 0, 0]), np.array([0, 0, 3, 0])]
    w = [0.5, 0.25, 0.25]
    out = merge_coverage_weighted(theta0, deltas, w, counts)
    want = theta0["head.weight"][0] + (5 * deltas[0]["head.weight"][0]
                                       + 1 * deltas[1]["head.weight"][0]) / 6
    assert torch.allclose(out["head.weight"][0], want, atol=1e-6)


def test_single_holder_row_equals_its_displacement():
    theta0, deltas = _setup()
    counts = [np.array([0, 0, 0, 7]), np.array([1, 1, 1, 0]), np.array([1, 1, 1, 0])]
    out = merge_coverage_weighted(theta0, deltas, [1 / 3] * 3, counts)
    want = theta0["head.weight"][3] + deltas[0]["head.weight"][3]
    assert torch.allclose(out["head.weight"][3], want, atol=1e-6)


def test_uncovered_row_keeps_initialization():
    theta0, deltas = _setup()
    counts = [np.array([1, 0, 1, 1])] * 3
    out = merge_coverage_weighted(theta0, deltas, [1 / 3] * 3, counts)
    assert torch.equal(out["head.weight"][1], theta0["head.weight"][1])
    assert torch.equal(out["head.bias"][1], theta0["head.bias"][1])


def test_non_head_tensors_take_the_sample_weighted_merge():
    theta0, deltas = _setup()
    counts = [np.array([1, 1, 1, 1])] * 3
    w = [0.2, 0.3, 0.5]
    out = merge_coverage_weighted(theta0, deltas, w, counts)
    ref = merge_sample_weighted(theta0, deltas, w)
    assert torch.equal(out["adapter"], ref["adapter"])
