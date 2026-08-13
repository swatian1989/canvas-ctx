"""CANVAS classification head and focal loss.

Head architecture, verbatim from the paper:
    z1 = Dropout(ReLU(W1 h  + b1))  -> 256
    z2 = Dropout(ReLU(W2 z1 + b2))  -> 128
    o  = W3 z2 + b3                 -> K+1 logits
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanvasHead(nn.Module):
    """Three-layer classifier over encoder features."""

    def __init__(self, embed_dim: int, n_classes: int = 11, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z1 = self.drop(F.relu(self.fc1(h)))
        z2 = self.drop(F.relu(self.fc2(z1)))
        return self.fc3(z2)


class FocalLoss(nn.Module):
    """L(p_t) = -alpha (1 - p_t)^gamma log(p_t).

    Used together with weighted random sampling, not instead of it. The paper
    applies both: the sampler balances exposure across classes, the loss
    down-weights easy examples within each batch.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weight", weight if weight is not None else torch.tensor([])
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.class_weight if self.class_weight.numel() else None
        ce = F.cross_entropy(logits, target, weight=w, reduction="none")
        p_t = torch.exp(-ce)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


def class_sample_weights(labels) -> torch.Tensor:
    """Weight = 1 / class frequency, for WeightedRandomSampler. [PAPER]"""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(float)
    counts[counts == 0] = 1.0
    return torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
