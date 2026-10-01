"""Device, seeding and output locations shared by every experiment.

Runs write to outputs/<experiment>/, which git ignores. The records behind the
paper's numbers are tracked separately in results/<experiment>/, so a rerun can
be compared against them without overwriting them. HEOFT_OUTPUTS overrides the
output location.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The shared head. Every other trainable tensor (the adapter) stays with its client.
HEAD_KEYS = ("head.weight", "head.bias")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("HEOFT_OUTPUTS", REPO_ROOT / "outputs"))


def is_head(k):
    return k in HEAD_KEYS


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def results_dir(name):
    """outputs/<name>/, created on first use."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(name):
    """outputs/<name>/artifacts/, the per-run tensors the attack experiments read.

    One run writes tens to hundreds of megabytes, and every file is
    regenerated from its seed.
    """
    d = results_dir(name) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
