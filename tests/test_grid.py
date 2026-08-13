"""2D/3D grid contract tests. Correctness of masking and slide isolation."""

import numpy as np
import pytest
import torch

from canvas_brca.method2_context.grid import (
    EMPTY, GridConfig, build_grid_index, build_slide_lattice, extract_windows,
    infer_stride, pyramid_pool_embeddings,
)
from canvas_brca.method2_context.habitat_net import HabitatNet, HabitatNetConfig
from canvas_brca.method2_context.unified_dataset import UnifiedPatchDataset, make_loaders


def _lattice_coords(side=12, stride=112.0):
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    return np.column_stack([gx.ravel() * stride, gy.ravel() * stride]).astype(float)


def test_stride_inferred_correctly():
    assert abs(infer_stride(_lattice_coords(stride=112.0)) - 112.0) < 6.0


def test_even_window_rejected():
    with pytest.raises(ValueError, match="odd"):
        GridConfig(window=6)


def test_centre_slot_always_valid():
    """The target patch must occupy the window centre. Non-negotiable."""
    coords = _lattice_coords(10)
    lat, r, c = build_slide_lattice(coords, 112.0)
    _, mask = extract_windows(lat, r, c, window=7)
    assert mask[:, 3, 3].all()


def test_edge_patches_masked_not_wrapped():
    """A corner patch has ~1/4 of its window off-slide. Those must be masked,
    not silently filled with patch 0."""
    coords = _lattice_coords(10)
    lat, r, c = build_slide_lattice(coords, 112.0)
    idx, mask = extract_windows(lat, r, c, window=7)
    corner = int(np.argmin(coords.sum(axis=1)))
    assert mask[corner].sum() < 49
    assert (idx[corner][~mask[corner]] == 0).all()   # parked, and masked out


def test_holes_in_tissue_masked():
    """Artefact-filtered patches leave lattice holes that must read as empty."""
    coords = _lattice_coords(8)
    keep = np.ones(len(coords), bool)
    keep[[20, 21, 28, 29]] = False
    lat, r, c = build_slide_lattice(coords[keep], 112.0)
    assert (lat == EMPTY).sum() >= 4


def test_pyramid_pool_shrinks_lattice():
    coords = _lattice_coords(16)
    lat, _, _ = build_slide_lattice(coords, 112.0)
    emb = np.random.randn(len(coords), 32).astype(np.float32)
    pyr = pyramid_pool_embeddings(emb, lat, n_scales=3)
    assert len(pyr) == 3
    assert pyr[1][1].shape[0] == int(np.ceil(lat.shape[0] / 2))
    assert pyr[2][1].shape[0] == int(np.ceil(lat.shape[0] / 4))


def test_pooled_embedding_is_block_mean():
    """Scale 1 slot must equal the mean of its 2x2 fine-grid parents."""
    coords = _lattice_coords(4)
    lat, _, _ = build_slide_lattice(coords, 112.0)
    emb = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
    pooled, plat = pyramid_pool_embeddings(emb, lat, 2)[1]
    parents = [lat[r, c] for r in (0, 1) for c in (0, 1)]
    assert np.allclose(pooled[plat[0, 0]], emb[parents].mean(0), atol=1e-5)


def test_grid_context_never_crosses_slides():
    """Two slides far apart must never appear in one another's windows."""
    coords = np.vstack([_lattice_coords(6), _lattice_coords(6) + 1e5])
    slides = np.array(["A"] * 36 + ["B"] * 36)
    emb = np.random.randn(72, 16).astype(np.float32)
    g = build_grid_index(emb, coords, slides, GridConfig(window=5, n_scales=1))
    banks = g["scale_embeddings"][0]
    assert len(banks) == 72          # each slide contributes its own patches
    # slide A occupies bank rows 0..35, slide B rows 36..71
    for i in range(36):
        valid = g["idx"][i, 0][g["mask"][i, 0]]
        assert (valid < 36).all()
    for i in range(36, 72):
        valid = g["idx"][i, 0][g["mask"][i, 0]]
        assert (valid >= 36).all()


@pytest.mark.parametrize("mode", ["none", "graph", "grid2d", "grid3d"])
def test_all_modes_forward(mode):
    coords = _lattice_coords(14)
    n = len(coords)
    X = np.random.randn(n, 48).astype(np.float32)
    y = np.random.randint(0, 11, n)
    sl = np.array(["S0"] * n)
    cfg = HabitatNetConfig(context_mode=mode, n_classes=11, proj_dim=32,
                           grid_channels=32, window=5, n_scales=2)
    ds = UnifiedPatchDataset(X, coords, y, sl, cfg)
    tr, _ = make_loaders(ds, ds, batch_size=8)
    c, ctx, _ = next(iter(tr))
    out = HabitatNet(48, cfg)(c, ctx)
    assert out.shape == (8, 11) and torch.isfinite(out).all()


def test_grid2d_forces_single_scale():
    """grid2d must ignore n_scales; otherwise the 2D/3D comparison is muddled."""
    cfg = HabitatNetConfig(context_mode="grid2d", n_scales=5)
    assert cfg.n_scales == 1


def test_grid_is_not_permutation_invariant():
    """The whole point of the grid over k-NN: it sees orientation."""
    cfg = HabitatNetConfig(context_mode="grid2d", n_classes=11, proj_dim=16,
                           grid_channels=16, window=5)
    m = HabitatNet(24, cfg).eval()
    grid = torch.randn(2, 24, 5, 5)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)
    c = torch.randn(2, 24)
    a = m(c, {"grid": grid, "mask": mask})
    b = m(c, {"grid": torch.rot90(grid, 1, dims=(2, 3)), "mask": mask})
    assert not torch.allclose(a, b), "rotating the grid must change the output"
