"""Datasets, tokenization and the Dirichlet label-skew partition.

Every loader draws its training and test subsets with a generator seeded by the
run's seed, so a (task, seed) pair fixes the data exactly. The training draw
precedes the test draw in the same stream, so changing the test-set size leaves
the training pool unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from .common import DEVICE
from .models import BACKBONES

TEXT_TASKS = {
    "ag_news":    dict(hf="fancyzhx/ag_news",    text="text",    label="label",        C=4),
    "dbpedia_14": dict(hf="fancyzhx/dbpedia_14", text="content", label="label",        C=14),
    "trec":       dict(hf="CogComp/trec",        text="text",    label="coarse_label", C=6),
    "banking77":  dict(hf="PolyAI/banking77",    text="text",    label="label",        C=77),
}

VISION_TASKS = {
    "cifar10":  dict(hf=("uoft-cs/cifar10", "cifar10"), C=10),
    "cifar100": dict(hf=("uoft-cs/cifar100", "cifar100"), C=100),
}

# Normalization of the ViT backbone (google/vit-base-patch16-224-in21k).
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def load_hf(hf_id):
    """load_dataset, falling back to the parquet conversion of the repository."""
    last = None
    for kw in ({}, {"revision": "refs/convert/parquet"}):
        try:
            return load_dataset(hf_id, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def load_text(task, max_train=20000, max_test=5000, seed=0):
    cfg = TEXT_TASKS[task]
    ds = load_hf(cfg["hf"])
    rng = np.random.default_rng(seed)
    def take(split, n):
        n = min(n, len(split))
        idx = rng.choice(len(split), n, replace=False)
        sub = split.select(idx.tolist())
        return list(sub[cfg["text"]]), np.array(sub[cfg["label"]], dtype=np.int64)
    Xtr, ytr = take(ds["train"], max_train)
    Xte, yte = take(ds["test"], max_test)
    return Xtr, ytr, Xte, yte, cfg["C"]


_TOK_CACHE = {}
def tokenize(backbone, texts, max_len=128):
    tok = _TOK_CACHE.setdefault("tok:" + backbone,
                                AutoTokenizer.from_pretrained(BACKBONES[backbone]["hf"]))
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


_TEXT_CACHE = {}
def text_data(task, backbone, seed):
    """Tokenized training and test sets of one text task, cached per process.

    Returns (ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C).
    """
    key = (task, backbone, seed)
    if key not in _TEXT_CACHE:
        Xtr, ytr, Xte, yte, C = load_text(task, seed=seed)
        ids_tr, mask_tr = tokenize(backbone, Xtr)
        ids_te, mask_te = tokenize(backbone, Xte)
        _TEXT_CACHE[key] = (ids_tr, mask_tr, ytr, ids_te, mask_te, yte, C)
    return _TEXT_CACHE[key]


def load_vision(task, max_train=10000, max_test=2000, seed=0):
    """Training and test images as uint8 tensors (n, 3, H, W), with labels."""
    cfg = VISION_TASKS[task]
    last = None
    for hf in cfg["hf"]:
        try:
            ds = load_dataset(hf); break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise last
    test_split = "test" if "test" in ds else "valid"
    rng = np.random.default_rng(seed)
    cols = ds["train"].column_names
    img_col = "img" if "img" in cols else "image"
    lbl_col = ("fine_label" if "fine_label" in cols
               else "label" if "label" in cols else "labels")

    def take(split, n):
        n = min(n, len(split))
        idx = rng.choice(len(split), n, replace=False)
        sub = split.select(idx.tolist())
        imgs = np.stack([np.array(im.convert("RGB")) for im in sub[img_col]])
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # uint8
        return x, np.array(sub[lbl_col], dtype=np.int64)

    Xtr, ytr = take(ds["train"], max_train)
    Xte, yte = take(ds[test_split], max_test)
    return Xtr, ytr, Xte, yte, cfg["C"]


def prepare_images(x_uint8):
    """uint8 images to normalized 224x224 float tensors on the device."""
    x = x_uint8.to(DEVICE).float().div_(255.0)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    return (x - MEAN.to(DEVICE)) / STD.to(DEVICE)


def dirichlet_partition(y, N, alpha, C, seed):
    """Split example indices over N clients with Dirichlet(alpha) label skew.

    For each class, the class's examples are shuffled and cut in the
    proportions of one Dirichlet(alpha) draw over the N clients.
    """
    rng = np.random.default_rng(seed)
    client = [[] for _ in range(N)]
    for c in range(C):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        if len(idx) == 0:
            continue
        props = rng.dirichlet([alpha] * N)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx, cuts)):
            client[i].extend(part.tolist())
    return [np.array(c, dtype=np.int64) for c in client]
