#!/usr/bin/env python
"""Download every backbone and dataset the experiments use into the local cache.

Run once on a machine with internet access (on a cluster, the login node).
Afterwards the experiments run with HF_HUB_OFFLINE=1.
"""
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from heoft.data import TEXT_TASKS, VISION_TASKS, load_hf
from heoft.models import BACKBONES, VIT


def main():
    for b in BACKBONES.values():
        AutoTokenizer.from_pretrained(b["hf"]); AutoModel.from_pretrained(b["hf"])
        print("backbone", b["hf"], flush=True)
    AutoModel.from_pretrained(VIT)
    print("backbone", VIT, flush=True)
    for cfg in TEXT_TASKS.values():
        load_hf(cfg["hf"])
        print("dataset", cfg["hf"], flush=True)
    for cfg in VISION_TASKS.values():
        load_dataset(cfg["hf"][0])
        print("dataset", cfg["hf"][0], flush=True)


if __name__ == "__main__":
    main()
