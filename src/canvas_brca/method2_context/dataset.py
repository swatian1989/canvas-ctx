"""Dataset and training loop for CANVAS-CTX. Operates on cached embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .context_model import ContextConfig, build_patch_neighbour_index

logger = logging.getLogger(__name__)


class ContextPatchDataset(Dataset):
    """Cached patch embeddings plus a precomputed spatial neighbour index.

    Neighbour indices are built PER SLIDE, so context never crosses slides.
    Getting this wrong leaks information across samples and inflates accuracy.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        slide_ids: np.ndarray,
        cfg: ContextConfig,
    ):
        self.emb = torch.as_tensor(embeddings, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.cfg = cfg

        n = len(embeddings)
        idx = np.zeros((n, cfg.k_neighbours), dtype=np.int64)
        dist = np.full((n, cfg.k_neighbours), np.inf, dtype=np.float32)
        mask = np.zeros((n, cfg.k_neighbours), dtype=bool)

        for sid in np.unique(slide_ids):
            sel = np.flatnonzero(slide_ids == sid)
            i, d, m = build_patch_neighbour_index(
                coords[sel], cfg.k_neighbours, cfg.radius_um
            )
            if cfg.k_neighbours > 0:
                idx[sel] = sel[i]      # map back to global rows
                dist[sel] = d
                mask[sel] = m

        self.idx = torch.as_tensor(idx)
        self.dist = torch.as_tensor(dist)
        self.mask = torch.as_tensor(mask)
        logger.info(
            "dataset: %d patches, %d slides, mean %.1f valid neighbours",
            n, len(np.unique(slide_ids)), float(self.mask.sum(1).float().mean()),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self.cfg.k_neighbours == 0:
            empty = torch.zeros(0, self.emb.shape[1])
            return (self.emb[i], empty, torch.zeros(0, dtype=torch.bool),
                    torch.zeros(0), self.labels[i])
        return (
            self.emb[i], self.emb[self.idx[i]], self.mask[i],
            self.dist[i], self.labels[i],
        )


def make_loaders(
    ds_train: ContextPatchDataset,
    ds_val: ContextPatchDataset,
    batch_size: int = 64,
    balanced: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Weighted random sampling on train. [PAPER] weight = 1 / class frequency."""
    if balanced:
        y = ds_train.labels.numpy()
        counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
        counts[counts == 0] = 1.0
        w = torch.as_tensor(1.0 / counts[y], dtype=torch.double)
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler)
    else:
        train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    return train, DataLoader(ds_val, batch_size=batch_size, shuffle=False)


@dataclass
class TrainResult:
    history: pd.DataFrame
    best_state: dict
    best_macro_f1: float


def train_context_model(
    model, train_loader, val_loader,
    epochs: int = 30, lr: float = 1e-4, lr_decay: float = 0.95,
    alpha: float = 0.25, gamma: float = 2.0, device: str = "cpu",
) -> TrainResult:
    """Train with focal loss and exponential LR decay. [PAPER] Adam, 1e-4, 0.95.

    No two-stage freezing here: the encoder is already frozen by construction
    since we train on cached embeddings. That is the trade for CPU feasibility,
    and it must be stated. Method 1 as reported in the paper fine-tunes the last
    two encoder layers; if you compare against a fine-tuned Method 1, the
    comparison is confounded. Compare frozen-vs-frozen.
    """
    from copy import deepcopy

    from sklearn.metrics import f1_score

    from ..stage3_model.head import FocalLoss

    model = model.to(device)
    crit = FocalLoss(alpha=alpha, gamma=gamma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)

    rows, best_f1, best_state = [], -1.0, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for centre, nbr, mask, dist, y in train_loader:
            opt.zero_grad()
            out = model(centre.to(device), nbr.to(device),
                        mask.to(device), dist.to(device))
            loss = crit(out, y.to(device))
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for centre, nbr, mask, dist, y in val_loader:
                out = model(centre.to(device), nbr.to(device),
                            mask.to(device), dist.to(device))
                preds.append(out.argmax(1).cpu().numpy())
                trues.append(y.numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        f1 = f1_score(t, p, average="macro", zero_division=0)
        acc = float((p == t).mean())
        rows.append({"epoch": ep, "train_loss": tot / len(train_loader.dataset),
                     "val_acc": acc, "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_state = f1, deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == epochs - 1:
            logger.info("epoch %2d  loss %.4f  val_acc %.3f  val_f1 %.3f",
                        ep, rows[-1]["train_loss"], acc, f1)

    return TrainResult(pd.DataFrame(rows), best_state, best_f1)
