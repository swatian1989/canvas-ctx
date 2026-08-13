"""Unified habitat classifier: one head, four context encoders.

    context_mode = "none"     Method 1. CANVAS exactly.
    context_mode = "graph"    k-NN deep-set + distance-biased attention
    context_mode = "grid2d"   local W x W feature grid, 2D conv + positional enc
    context_mode = "grid3d"   local S x W x W multi-scale cube, 3D conv

Every mode feeds the same unchanged CANVAS head (256 -> 128 -> K+1) and is
trained with the same focal loss, weighted sampler and sample-level splits. The
only thing that varies is how neighbourhood evidence reaches the head, so the
four-way benchmark is a controlled ablation rather than a model zoo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_model import DeepSetAggregator, SpatialAttention

logger = logging.getLogger(__name__)

CONTEXT_MODES = ("none", "graph", "grid2d", "grid3d")


@dataclass
class HabitatNetConfig:
    context_mode: str = "grid2d"
    n_classes: int = 11
    dropout: float = 0.5          # [PAPER]
    proj_dim: int = 128
    # graph
    k_neighbours: int = 8
    radius_um: float = 300.0
    distance_prior: bool = True
    # grid
    window: int = 7
    n_scales: int = 3             # grid3d only
    grid_channels: int = 96
    seed: int = 42

    def __post_init__(self) -> None:
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if self.context_mode == "grid2d":
            self.n_scales = 1


class Grid2DEncoder(nn.Module):
    """CNN over a local W x W feature grid with learned positional encoding.

    Design notes.

    A 1x1 convolution first projects D-dimensional embeddings down to
    ``grid_channels``; running 3x3 convolutions directly on a 768- or 2048-dim
    embedding grid is wasteful and overfits immediately at pilot scale.

    Empty lattice slots are zeroed via the mask *before* convolution and the
    mask is carried as an extra input channel, so the network can distinguish
    "stroma here" from "nothing here". Without that channel, tissue edges look
    identical to low-signal tissue.

    The learned positional embedding is what makes this differ from k-NN
    pooling: it is not permutation invariant, so the model can represent
    directional structure such as an invasive front or a duct wall.
    """

    def __init__(self, in_dim: int, window: int, channels: int = 96,
                 out_dim: int = 128):
        super().__init__()
        self.window = window
        self.proj = nn.Conv2d(in_dim + 1, channels, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, channels, window, window))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(8, channels),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """grid (B, D, W, W), mask (B, W, W) -> (B, out_dim)."""
        m = mask.unsqueeze(1).float()
        x = torch.cat([grid * m, m], dim=1)
        x = self.proj(x) + self.pos
        x = self.block(x)

        denom = m.sum(dim=(2, 3)).clamp(min=1.0)
        pooled = (x * m).sum(dim=(2, 3)) / denom
        c = self.window // 2
        centre = x[:, :, c, c]
        return self.head(torch.cat([pooled, centre], dim=-1))


class Grid3DEncoder(nn.Module):
    """3D CNN over a local S x W x W multi-scale cube.

    The depth axis is magnification scale, not physical z. Kernels are
    (3, 3, 3) so a unit mixes evidence across adjacent scales at nearby spatial
    positions, which is the whole point: a region that looks like stroma at
    high magnification but sits inside a glandular structure at low
    magnification is a different habitat from one that is stroma at every scale.

    Padding on the scale axis is 'replicate' rather than zero, because zero
    padding would tell the network there is empty tissue beyond the coarsest
    scale, which is meaningless.
    """

    def __init__(self, in_dim: int, window: int, n_scales: int,
                 channels: int = 96, out_dim: int = 128):
        super().__init__()
        self.window, self.n_scales = window, n_scales
        self.proj = nn.Conv3d(in_dim + 1, channels, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, channels, n_scales, window, window))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1))
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.head = nn.Linear(channels * 2, out_dim)
        self.out_dim = out_dim

    def _pad_scale(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")

    def forward(self, cube: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """cube (B, D, S, W, W), mask (B, S, W, W) -> (B, out_dim)."""
        m = mask.unsqueeze(1).float()
        x = torch.cat([cube * m, m], dim=1)
        x = self.proj(x) + self.pos
        x = F.relu(self.norm1(self.conv1(self._pad_scale(x))))
        x = F.relu(self.norm2(self.conv2(self._pad_scale(x))))

        denom = m.sum(dim=(2, 3, 4)).clamp(min=1.0)
        pooled = (x * m).sum(dim=(2, 3, 4)) / denom
        c = self.window // 2
        centre = x[:, :, 0, c, c]        # scale 0 is native resolution
        return self.head(torch.cat([pooled, centre], dim=-1))


class HabitatNet(nn.Module):
    """The unified model. All context modes, one head."""

    def __init__(self, embed_dim: int, cfg: HabitatNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or HabitatNetConfig()
        torch.manual_seed(cfg.seed)

        fused = embed_dim
        self.deepset = self.attn = self.grid2d = self.grid3d = None

        if cfg.context_mode == "graph":
            self.deepset = DeepSetAggregator(embed_dim, cfg.proj_dim)
            self.attn = SpatialAttention(embed_dim, cfg.proj_dim, cfg.distance_prior)
            fused += self.deepset.out_dim + self.attn.out_dim
        elif cfg.context_mode == "grid2d":
            self.grid2d = Grid2DEncoder(embed_dim, cfg.window,
                                        cfg.grid_channels, cfg.proj_dim)
            fused += self.grid2d.out_dim
        elif cfg.context_mode == "grid3d":
            self.grid3d = Grid3DEncoder(embed_dim, cfg.window, cfg.n_scales,
                                        cfg.grid_channels, cfg.proj_dim)
            fused += self.grid3d.out_dim

        # unchanged CANVAS head  [PAPER]
        self.fc1 = nn.Linear(fused, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, cfg.n_classes)
        self.drop = nn.Dropout(cfg.dropout)

        n_par = sum(p.numel() for p in self.parameters())
        logger.info("HabitatNet[%s]: fused %d, %.2fM params",
                    cfg.context_mode, fused, n_par / 1e6)

    def forward(self, centre: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        """centre (B, D). context carries mode-specific tensors."""
        parts = [centre]
        ctx = context or {}

        if self.deepset is not None:
            parts.append(self.deepset(ctx["nbr"], ctx["mask"]))
            parts.append(self.attn(centre, ctx["nbr"], ctx["mask"], ctx["dist"]))
        elif self.grid2d is not None:
            parts.append(self.grid2d(ctx["grid"], ctx["mask"]))
        elif self.grid3d is not None:
            parts.append(self.grid3d(ctx["cube"], ctx["mask"]))

        h = torch.cat(parts, dim=-1)
        z1 = self.drop(F.relu(self.fc1(h)))
        z2 = self.drop(F.relu(self.fc2(z1)))
        return self.fc3(z2)
