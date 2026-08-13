"""CANVAS-CTX contract tests. The k=0 equivalence is the important one."""

import numpy as np
import torch

from canvas_brca.method2_context.context_model import (
    ContextConfig, ContextHabitatNet, build_patch_neighbour_index,
)
from canvas_brca.method2_context.dataset import ContextPatchDataset


def _grid(n_side=10, spacing=112.0):
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return np.column_stack([gx.ravel() * spacing, gy.ravel() * spacing]).astype(float)


def test_k_zero_recovers_canvas_head():
    """k=0 must give a plain 3-layer head on the raw embedding. This is the
    ablation baseline; if it silently differs, the comparison is invalid."""
    cfg = ContextConfig(k_neighbours=0, n_classes=11)
    m = ContextHabitatNet(768, cfg)
    assert m.deepset is None and m.attn is None
    assert m.fc1.in_features == 768
    out = m(torch.randn(4, 768))
    assert out.shape == (4, 11)


def test_context_changes_output():
    cfg = ContextConfig(k_neighbours=8, n_classes=11, proj_dim=64)
    m = ContextHabitatNet(128, cfg).eval()
    c = torch.randn(4, 128)
    nbr = torch.randn(4, 8, 128)
    mask = torch.ones(4, 8, dtype=torch.bool)
    dist = torch.full((4, 8), 100.0)
    a = m(c, nbr, mask, dist)
    b = m(c, torch.randn(4, 8, 128), mask, dist)
    assert not torch.allclose(a, b), "neighbours must influence the output"


def test_neighbours_never_cross_slides():
    """Context leaking across slides would inflate accuracy. Must not happen."""
    coords = np.vstack([_grid(6), _grid(6) + 1e5])
    slides = np.array(["A"] * 36 + ["B"] * 36)
    cfg = ContextConfig(k_neighbours=4, radius_um=500.0)
    ds = ContextPatchDataset(
        np.random.randn(72, 32).astype(np.float32), coords,
        np.random.randint(0, 5, 72), slides, cfg,
    )
    for i in range(72):
        valid = ds.idx[i][ds.mask[i]].numpy()
        assert all(slides[j] == slides[i] for j in valid)


def test_padded_neighbours_masked():
    """A 3-patch fragment asking for k=8 must not invent neighbours."""
    coords = _grid(2)[:3]
    idx, dist, mask = build_patch_neighbour_index(coords, k=8, radius_um=500.0)
    assert idx.shape == (3, 8) and mask.shape == (3, 8)
    assert mask.sum(axis=1).max() <= 2      # at most n-1 real neighbours


def test_isolated_patch_survives():
    """A patch with zero valid neighbours must still produce finite logits."""
    cfg = ContextConfig(k_neighbours=4, n_classes=11, proj_dim=32)
    m = ContextHabitatNet(64, cfg).eval()
    out = m(torch.randn(2, 64), torch.randn(2, 4, 64),
            torch.zeros(2, 4, dtype=torch.bool), torch.full((2, 4), np.inf))
    assert torch.isfinite(out).all()


def test_radius_excludes_distant_patches():
    coords = _grid(10, spacing=112.0)
    _, _, near = build_patch_neighbour_index(coords, k=8, radius_um=120.0)
    _, _, far = build_patch_neighbour_index(coords, k=8, radius_um=1000.0)
    assert near.sum() < far.sum()


def test_attention_weights_sum_to_one():
    cfg = ContextConfig(k_neighbours=6, n_classes=11, proj_dim=32)
    m = ContextHabitatNet(64, cfg).eval()
    w = m.attention_map(torch.randn(3, 64), torch.randn(3, 6, 64),
                        torch.ones(3, 6, dtype=torch.bool), torch.full((3, 6), 50.0))
    assert torch.allclose(w.sum(1), torch.ones(3), atol=1e-5)
