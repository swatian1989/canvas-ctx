"""Tests for stage3_model/extract_embeddings.py.

Every test uses a synthetic in-memory RGB array. No WSI file, no openslide
package, and no network access anywhere in this file: `_FakeSlide` duck-types
the handful of openslide.OpenSlide methods `iter_slide_patches` calls, and the
stub encoder below stands in for a real foundation model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from canvas_brca.stage3_model.extract_embeddings import (
    ExtractConfig,
    artefact_flags,
    build_tissue_mask,
    encode_slide,
    extract_embeddings,
    is_artefact,
    iter_slide_patches,
    macenko_normalize,
)


# --------------------------------------------------------------- test doubles


class _FakeSlide:
    """Duck-types the openslide.OpenSlide surface iter_slide_patches uses."""

    def __init__(self, arr: np.ndarray, mpp: float = 0.50, slide_id: str = "fake01"):
        self.arr = arr
        self.properties = {"openslide.mpp-x": str(mpp)}
        self.dimensions = (arr.shape[1], arr.shape[0])  # (w, h)
        self.slide_id = slide_id
        self.closed = False

    def get_thumbnail(self, size):
        return Image.fromarray(self.arr).resize(size)

    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        tile = np.zeros((h, w, 4), dtype=np.uint8)
        H, W = self.arr.shape[:2]
        y1, x1 = min(y + h, H), min(x + w, W)
        if y < H and x < W:
            sub = self.arr[y:y1, x:x1]
            tile[: sub.shape[0], : sub.shape[1], :3] = sub
            tile[: sub.shape[0], : sub.shape[1], 3] = 255
        return Image.fromarray(tile, mode="RGBA")

    def close(self):
        self.closed = True


def _tissue_like_patch(rng, size=224, base=(205, 110, 185)):
    """Saturated pink/purple-ish patch, roughly H&E-toned, with texture.

    Base sits solidly in eosin-pink/hematoxylin-purple hue territory, well
    clear of the green/blue hue band `artefact_flags` treats as pen ink.
    """
    noise = rng.integers(-15, 15, size=(size, size, 3))
    img = np.clip(np.array(base) + noise, 0, 255).astype(np.uint8)
    return img


def _background_patch(size=224):
    """Near-white, low-saturation: what a tissue mask should reject."""
    return np.full((size, size, 3), 245, dtype=np.uint8)


class _StubPreprocess:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, 8, 8)


class _StubEncoder(torch.nn.Module):
    """Trivial per-channel mean -> a deterministic 3-d "embedding"."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=[2, 3])


# ------------------------------------------------------------- tissue mask


def test_tissue_mask_separates_blob_from_background():
    rng = np.random.default_rng(0)
    thumb = np.full((64, 64, 3), 245, dtype=np.uint8)
    thumb[16:48, 16:48] = _tissue_like_patch(rng, size=32)
    mask = build_tissue_mask(thumb)
    assert mask[24:40, 24:40].mean() > 0.8      # blob center: tissue
    assert mask[:8, :8].mean() < 0.2            # corner: background


# ---------------------------------------------------------------- artefacts


def test_flat_patch_is_blurry():
    cfg = ExtractConfig()
    flat = np.full((224, 224, 3), 200, dtype=np.uint8)
    flags = artefact_flags(flat, cfg)
    assert flags["blur_var"] < cfg.blur_laplacian_min
    assert is_artefact(flags, cfg)


def test_sharp_checkerboard_not_blurry():
    cfg = ExtractConfig()
    board = (np.indices((224, 224)).sum(axis=0) % 16 < 8).astype(np.uint8) * 255
    sharp = np.stack([board] * 3, axis=-1)
    flags = artefact_flags(sharp, cfg)
    assert flags["blur_var"] > cfg.blur_laplacian_min


def test_pen_ink_detected():
    cfg = ExtractConfig()
    img = _tissue_like_patch(np.random.default_rng(1))
    img = img.copy()
    img[50:150, 50:150] = [0, 200, 0]  # saturated green blob, like marker ink
    flags = artefact_flags(img, cfg)
    assert flags["pen_frac"] > cfg.pen_ink_max_frac
    assert is_artefact(flags, cfg)


# ------------------------------------------------------------------ macenko


def test_macenko_normalize_shape_dtype_range():
    rng = np.random.default_rng(2)
    patch = _tissue_like_patch(rng)
    out = macenko_normalize(patch)
    assert out.shape == patch.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_macenko_normalize_passthrough_on_low_signal():
    blank = np.full((224, 224, 3), 255, dtype=np.uint8)
    out = macenko_normalize(blank)
    np.testing.assert_array_equal(out, blank)


def test_macenko_normalize_changes_a_real_patch():
    rng = np.random.default_rng(3)
    patch = _tissue_like_patch(rng)
    out = macenko_normalize(patch)
    assert not np.array_equal(out, patch)


# ------------------------------------------------------------------ tiling


def _checkerboard_slide(rng, tiles=4, tile_px=224, mpp=0.50):
    """tiles x tiles grid, alternating tissue / background blocks."""
    side = tiles * tile_px
    arr = np.full((side, side, 3), 245, dtype=np.uint8)
    for ty in range(tiles):
        for tx in range(tiles):
            if (tx + ty) % 2 == 0:
                y0, x0 = ty * tile_px, tx * tile_px
                arr[y0:y0 + tile_px, x0:x0 + tile_px] = _tissue_like_patch(rng, size=tile_px)
    return _FakeSlide(arr, mpp=mpp)


def test_iter_slide_patches_respects_tissue_mask():
    rng = np.random.default_rng(4)
    slide = _checkerboard_slide(rng, tiles=4)
    cfg = ExtractConfig(patch_px=224, stride_px=224, target_mpp=0.50,
                        min_tissue_frac=0.40, max_patches_per_slide=None,
                        blur_laplacian_min=0.0)  # noise-textured tiles pass blur trivially
    patches = list(iter_slide_patches(slide, cfg, slide_id="check01"))
    # half the 4x4 grid is tissue -> up to 8 candidates (fewer if edge tiles
    # get masked partially); every yielded patch must be a "tissue" tile
    assert 0 < len(patches) <= 8
    for p in patches:
        assert p.slide_id == "check01"


def test_iter_slide_patches_grid_subsample_caps_and_spreads():
    rng = np.random.default_rng(5)
    slide = _checkerboard_slide(rng, tiles=6)  # 18 tissue tiles available
    cfg = ExtractConfig(patch_px=224, stride_px=224, target_mpp=0.50,
                        min_tissue_frac=0.40, max_patches_per_slide=4,
                        blur_laplacian_min=0.0)
    patches = list(iter_slide_patches(slide, cfg, slide_id="cap01"))
    assert len(patches) <= 4
    xs = sorted(p.x_um for p in patches)
    # spread across the slide width, not clustered at the start
    assert xs[-1] - xs[0] > 0


def test_iter_slide_patches_missing_mpp_raises():
    slide = _FakeSlide(_background_patch(64), mpp=0.5)
    slide.properties = {}  # no mpp property at all
    cfg = ExtractConfig()
    with pytest.raises(ValueError):
        list(iter_slide_patches(slide, cfg))


# --------------------------------------------------------------- encode + i/o


def test_encode_slide_schema():
    rng = np.random.default_rng(6)
    slide = _checkerboard_slide(rng, tiles=3)
    cfg = ExtractConfig(min_tissue_frac=0.40, max_patches_per_slide=None,
                        blur_laplacian_min=0.0)
    df = encode_slide(slide, _StubEncoder(), _StubPreprocess(), cfg, slide_id="enc01")
    assert not df.empty
    expected = {"slide_id", "x_um", "y_um", "patch_x", "patch_y",
                "emb_0", "emb_1", "emb_2"}
    assert set(df.columns) == expected
    assert (df["slide_id"] == "enc01").all()


def test_extract_embeddings_resumable_skip(tmp_path):
    rng = np.random.default_rng(7)
    slide = _checkerboard_slide(rng, tiles=3)
    cfg = ExtractConfig(min_tissue_frac=0.40, max_patches_per_slide=None,
                        blur_laplacian_min=0.0)
    outdir = tmp_path / "emb"

    extract_embeddings([slide], _StubEncoder(), _StubPreprocess(), cfg, str(outdir))
    shard = outdir / f"{slide.slide_id}.parquet"
    assert shard.exists()
    first = pd.read_parquet(shard)

    slide.arr[:] = 0  # mutate the slide; a re-encode would produce different rows
    extract_embeddings([slide], _StubEncoder(), _StubPreprocess(), cfg, str(outdir))
    second = pd.read_parquet(shard)
    pd.testing.assert_frame_equal(first, second)  # untouched: it was skipped


def test_extract_embeddings_force_reencodes(tmp_path):
    rng = np.random.default_rng(8)
    slide = _checkerboard_slide(rng, tiles=3)
    cfg = ExtractConfig(min_tissue_frac=0.40, max_patches_per_slide=None,
                        blur_laplacian_min=0.0)
    outdir = tmp_path / "emb"

    extract_embeddings([slide], _StubEncoder(), _StubPreprocess(), cfg, str(outdir))
    shard = outdir / f"{slide.slide_id}.parquet"
    first = pd.read_parquet(shard)

    # Re-tint the SAME tissue footprint a different colour (still saturated,
    # so the tissue mask still fires) -- an all-zero mutation would make the
    # whole slide read as background and produce no shard at all, which would
    # make this test pass for the wrong reason (nothing re-written).
    tiles, tile_px = 3, 224
    for ty in range(tiles):
        for tx in range(tiles):
            if (tx + ty) % 2 == 0:
                y0, x0 = ty * tile_px, tx * tile_px
                slide.arr[y0:y0 + tile_px, x0:x0 + tile_px] = _tissue_like_patch(
                    rng, size=tile_px, base=(230, 150, 210))
    extract_embeddings([slide], _StubEncoder(), _StubPreprocess(), cfg, str(outdir),
                       force=True)
    second = pd.read_parquet(shard)
    assert not first["emb_0"].equals(second["emb_0"])
