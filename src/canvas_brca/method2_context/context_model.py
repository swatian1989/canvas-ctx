"""Method 2: context-aware habitat inference (CANVAS-CTX).

Motivation
----------
CANVAS defines habitats from 40 um cellular neighbourhoods, then predicts them
from H&E with a classifier that sees one 224x224 patch at a time. The label is
inherently contextual; the predictor is not. Current spatial-omics-from-H&E
methods have moved the other way, aggregating neighbourhood morphology (deep
sets in DeepSpot, graph message passing in SEPAL/EGGN, 2D feature grids in
TITAN).

CANVAS-CTX predicts the habitat of patch i from patch i AND its k spatial
neighbours, via two parallel aggregators over the same cached embeddings:

    deep-set branch   permutation-invariant mean+max pool of neighbour features
    attention branch  single-head dot-product attention with a distance prior

Both feed the unchanged CANVAS head, so the comparison against Method 1 is
clean: same encoder, same labels, same splits, same loss, same head.

Why this is cheap
-----------------
It consumes CACHED embeddings, never images. Encoding is done once by Method 1.
Training CANVAS-CTX on 100 K cached patch embeddings takes minutes on CPU. No
PyTorch Geometric dependency; attention is dense over a fixed-k index tensor.

The built-in ablation
---------------------
Setting k = 0 reduces CANVAS-CTX exactly to the CANVAS head on the raw
embedding. That is Method 1. So the k-sweep IS the ablation, and any gain is
attributable to spatial context alone rather than to extra parameters or a
different backbone. Run k in {0, 4, 8, 16} and report the curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for CANVAS-CTX."""

    k_neighbours: int = 8          # 0 reduces exactly to Method 1 (CANVAS)
    radius_um: float = 300.0       # spatial cap on what counts as a neighbour
    proj_dim: int = 256            # shared projection before aggregation
    n_classes: int = 11            # 10 habitats + background
    dropout: float = 0.5           # [PAPER] same as CANVAS
    use_deepset: bool = True
    use_attention: bool = True
    distance_prior: bool = True    # bias attention by inverse neighbour distance
    seed: int = 42


# ------------------------------------------------------------------ neighbours


def build_patch_neighbour_index(
    coords: np.ndarray,
    k: int,
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-NN index over patch centroids within one slide.

    Returns
    -------
    idx
        (n, k) int array of neighbour row indices. Self is excluded.
    dist
        (n, k) float array of distances in microns.
    mask
        (n, k) bool array, True where the neighbour is real (within radius and
        the slide has enough patches). Padded slots are False and must be
        masked out of both aggregators, otherwise small tissue fragments get
        their features diluted by fake zero-neighbours.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    if k == 0 or n < 2:
        return (
            np.zeros((n, 0), dtype=np.int64),
            np.zeros((n, 0), dtype=np.float32),
            np.zeros((n, 0), dtype=bool),
        )

    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k_eff + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]           # drop self

    mask = dist <= radius_um
    idx = np.where(mask, idx, 0)                  # park padded slots on row 0

    if k_eff < k:                                 # pad to fixed k
        pad = k - k_eff
        idx = np.pad(idx, ((0, 0), (0, pad)))
        dist = np.pad(dist, ((0, 0), (0, pad)), constant_values=np.inf)
        mask = np.pad(mask, ((0, 0), (0, pad)), constant_values=False)

    return idx.astype(np.int64), dist.astype(np.float32), mask


# --------------------------------------------------------------------- model


class DeepSetAggregator(nn.Module):
    """Permutation-invariant mean+max pooling over masked neighbours."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.out_dim = out_dim * 2

    def forward(self, nbr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """nbr (B, k, D), mask (B, k) -> (B, 2*out_dim)."""
        h = self.phi(nbr)
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (h * m).sum(dim=1) / denom
        mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        mx = torch.nan_to_num(mx, neginf=0.0)     # rows with zero neighbours
        return torch.cat([mean, mx], dim=-1)


class SpatialAttention(nn.Module):
    """Single-head dot-product attention with an optional inverse-distance bias.

    The distance prior encodes the assumption that morphologically adjacent
    tissue is more informative than distant tissue. It is a bias on the logits,
    not a hard constraint, so the model can override it where the data says a
    far-off region matters.
    """

    def __init__(self, in_dim: int, out_dim: int, distance_prior: bool = True):
        super().__init__()
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.scale = out_dim ** -0.5
        self.distance_prior = distance_prior
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.out_dim = out_dim

    def forward(
        self, centre: torch.Tensor, nbr: torch.Tensor,
        mask: torch.Tensor, dist: torch.Tensor,
    ) -> torch.Tensor:
        """centre (B, D), nbr (B, k, D) -> (B, out_dim)."""
        q = self.q(centre).unsqueeze(1)                    # (B, 1, d)
        k = self.k(nbr)                                    # (B, k, d)
        v = self.v(nbr)
        logits = (q * k).sum(-1) * self.scale              # (B, k)

        if self.distance_prior:
            safe = torch.nan_to_num(dist, posinf=1e6)
            logits = logits - self.beta * torch.log1p(safe / 100.0)

        logits = logits.masked_fill(~mask, float("-inf"))
        no_nbr = ~mask.any(dim=1)
        weights = torch.softmax(logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        out = (weights.unsqueeze(-1) * v).sum(dim=1)
        return out.masked_fill(no_nbr.unsqueeze(-1), 0.0)


class ContextHabitatNet(nn.Module):
    """CANVAS-CTX. Set k_neighbours=0 to recover CANVAS exactly."""

    def __init__(self, embed_dim: int, cfg: ContextConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ContextConfig()
        torch.manual_seed(cfg.seed)

        fused = embed_dim
        self.deepset = None
        self.attn = None

        if cfg.k_neighbours > 0:
            if cfg.use_deepset:
                self.deepset = DeepSetAggregator(embed_dim, cfg.proj_dim)
                fused += self.deepset.out_dim
            if cfg.use_attention:
                self.attn = SpatialAttention(
                    embed_dim, cfg.proj_dim, cfg.distance_prior
                )
                fused += self.attn.out_dim

        # unchanged CANVAS head: 256 -> 128 -> K+1  [PAPER]
        self.fc1 = nn.Linear(fused, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, cfg.n_classes)
        self.drop = nn.Dropout(cfg.dropout)
        logger.info("ContextHabitatNet: fused dim %d, k=%d", fused, cfg.k_neighbours)

    def forward(
        self, centre: torch.Tensor,
        nbr: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [centre]
        if self.cfg.k_neighbours > 0 and nbr is not None and nbr.shape[1] > 0:
            if self.deepset is not None:
                parts.append(self.deepset(nbr, mask))
            if self.attn is not None:
                parts.append(self.attn(centre, nbr, mask, dist))
        h = torch.cat(parts, dim=-1)
        z1 = self.drop(F.relu(self.fc1(h)))
        z2 = self.drop(F.relu(self.fc2(z1)))
        return self.fc3(z2)

    def attention_map(
        self, centre: torch.Tensor, nbr: torch.Tensor,
        mask: torch.Tensor, dist: torch.Tensor,
    ) -> torch.Tensor:
        """Return attention weights for interpretability figures.

        Which neighbouring patches the model consults to call a habitat is the
        interpretable output that Method 1 cannot produce, and it is the natural
        figure for the paper: an H&E region with arrows to the context it used.
        """
        with torch.no_grad():
            q = self.attn.q(centre).unsqueeze(1)
            k = self.attn.k(nbr)
            logits = (q * k).sum(-1) * self.attn.scale
            if self.attn.distance_prior:
                safe = torch.nan_to_num(dist, posinf=1e6)
                logits = logits - self.attn.beta * torch.log1p(safe / 100.0)
            logits = logits.masked_fill(~mask, float("-inf"))
            return torch.nan_to_num(torch.softmax(logits, dim=1), nan=0.0)
