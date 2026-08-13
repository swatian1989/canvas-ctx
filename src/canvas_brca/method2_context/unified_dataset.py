"""Dataset serving any context mode from the same cached embeddings."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .context_model import build_patch_neighbour_index
from .grid import GridConfig, build_grid_index
from .habitat_net import HabitatNetConfig

logger = logging.getLogger(__name__)


class UnifiedPatchDataset(Dataset):
    """Cached embeddings + per-slide context index for the chosen mode."""

    def __init__(self, embeddings, coords, labels, slide_ids,
                 cfg: HabitatNetConfig):
        self.cfg = cfg
        self.emb = torch.as_tensor(embeddings, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.mode = cfg.context_mode
        n = len(embeddings)

        if self.mode == "none":
            return

        if self.mode == "graph":
            idx = np.zeros((n, cfg.k_neighbours), np.int64)
            dist = np.full((n, cfg.k_neighbours), np.inf, np.float32)
            mask = np.zeros((n, cfg.k_neighbours), bool)
            for sid in np.unique(slide_ids):
                sel = np.flatnonzero(slide_ids == sid)
                i, d, m = build_patch_neighbour_index(
                    coords[sel], cfg.k_neighbours, cfg.radius_um)
                if cfg.k_neighbours:
                    idx[sel], dist[sel], mask[sel] = sel[i], d, m
            self.idx = torch.as_tensor(idx)
            self.dist = torch.as_tensor(dist)
            self.mask = torch.as_tensor(mask)
            return

        gcfg = GridConfig(window=cfg.window,
                          n_scales=1 if self.mode == "grid2d" else cfg.n_scales)
        g = build_grid_index(embeddings, coords, slide_ids, gcfg)
        self.banks = [torch.as_tensor(e) for e in g["scale_embeddings"]]
        self.gidx = torch.as_tensor(g["idx"].astype(np.int64))   # (n, S, W, W)
        self.gmask = torch.as_tensor(g["mask"])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        c, y = self.emb[i], self.labels[i]

        if self.mode == "none":
            return c, {}, y

        if self.mode == "graph":
            return c, {"nbr": self.emb[self.idx[i]], "mask": self.mask[i],
                       "dist": self.dist[i]}, y

        # gather (S, W, W, D) from the per-scale banks, then move D to front
        cube = torch.stack([
            self.banks[s][self.gidx[i, s]] for s in range(self.gidx.shape[1])
        ])                                              # (S, W, W, D)
        cube = cube.permute(3, 0, 1, 2).contiguous()    # (D, S, W, W)

        if self.mode == "grid2d":
            return c, {"grid": cube[:, 0], "mask": self.gmask[i, 0]}, y
        return c, {"cube": cube, "mask": self.gmask[i]}, y


def collate(batch):
    centres = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    keys = batch[0][1].keys()
    ctx = {k: torch.stack([b[1][k] for b in batch]) for k in keys}
    return centres, ctx, labels


def make_loaders(ds_train, ds_val, batch_size=64, balanced=True):
    """[PAPER] weighted random sampling, weight = 1 / class frequency."""
    if balanced:
        y = ds_train.labels.numpy()
        cnt = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
        cnt[cnt == 0] = 1.0
        w = torch.as_tensor(1.0 / cnt[y], dtype=torch.double)
        tr = DataLoader(ds_train, batch_size=batch_size, collate_fn=collate,
                        sampler=WeightedRandomSampler(w, len(w), replacement=True))
    else:
        tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                        collate_fn=collate)
    va = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return tr, va


def train_unified(model, train_loader, val_loader, epochs=30, lr=1e-4,
                  lr_decay=0.95, alpha=0.25, gamma=2.0, device="cpu"):
    """Focal loss, Adam 1e-4, exponential decay 0.95. [PAPER]

    The encoder is frozen by construction (we train on cached embeddings), so
    there is no two-stage unfreezing. Compare modes frozen-vs-frozen.
    """
    from copy import deepcopy

    import pandas as pd
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
        for c, ctx, y in train_loader:
            opt.zero_grad()
            loss = crit(model(c.to(device),
                              {k: v.to(device) for k, v in ctx.items()}),
                        y.to(device))
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
        sched.step()

        model.eval()
        p, t = [], []
        with torch.no_grad():
            for c, ctx, y in val_loader:
                out = model(c.to(device), {k: v.to(device) for k, v in ctx.items()})
                p.append(out.argmax(1).cpu().numpy())
                t.append(y.numpy())
        p, t = np.concatenate(p), np.concatenate(t)
        f1 = f1_score(t, p, average="macro", zero_division=0)
        rows.append({"epoch": ep, "train_loss": tot / len(train_loader.dataset),
                     "val_acc": float((p == t).mean()), "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_state = f1, deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == epochs - 1:
            logger.info("epoch %2d  loss %.4f  acc %.3f  f1 %.3f", ep,
                        rows[-1]["train_loss"], rows[-1]["val_acc"], f1)

    return pd.DataFrame(rows), best_state, best_f1
