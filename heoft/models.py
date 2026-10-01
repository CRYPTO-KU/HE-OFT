"""The client model: a frozen public backbone, a LoRA adapter and a linear head.

The adapter's down-projection (lora_A) is frozen at its seeded initialization,
which every client shares, and only the up-projection (lora_B) and the head are
trained. With rank 0 there is no adapter and the client trains the head alone.
The head is the only part the federation shares.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from .common import HEAD_KEYS, is_head  # noqa: F401

BACKBONES = {
    "roberta_base": dict(hf="roberta-base", targets=["query", "value"]),
}
VIT = "google/vit-base-patch16-224-in21k"
VIT_TARGETS = ["query", "value"]


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)


def _freeze_down_projection(base):
    for n, p in base.named_parameters():
        if "lora_A" in n:
            p.requires_grad = False


class TextLoRA(nn.Module):
    """RoBERTa-base, LoRA on the query and value projections, mean-pooled head."""
    def __init__(self, backbone, C, r=8, freeze_a=True):
        super().__init__()
        base = AutoModel.from_pretrained(BACKBONES[backbone]["hf"])
        hidden = base.config.hidden_size
        for p in base.parameters():
            p.requires_grad = False
        if r > 0:
            lcfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                              target_modules=BACKBONES[backbone]["targets"])
            base = get_peft_model(base, lcfg)
            if freeze_a:
                _freeze_down_projection(base)
        self.backbone = base
        self.head = nn.Linear(hidden, C)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(mean_pool(out, mask))


class ViTLoRA(nn.Module):
    """ViT-B/16, LoRA on the query and value projections, head on the CLS token."""
    def __init__(self, C, r=8, freeze_a=True):
        super().__init__()
        base = AutoModel.from_pretrained(VIT)
        hidden = base.config.hidden_size
        for p in base.parameters():
            p.requires_grad = False
        if r > 0:
            base = get_peft_model(base, LoraConfig(
                r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                target_modules=VIT_TARGETS))
            if freeze_a:
                _freeze_down_projection(base)
        self.backbone = base
        self.head = nn.Linear(hidden, C)

    def forward(self, x):
        out = self.backbone(pixel_values=x).last_hidden_state
        return self.head(out[:, 0])


def trainable_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.device))


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
