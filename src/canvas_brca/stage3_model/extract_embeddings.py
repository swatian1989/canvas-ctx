"""Stage 3: WSI -> cached patch embeddings.

The bridge between raw whole-slide images and everything in Method 2.
Streams tiles, never holds a WSI in RAM, and writes one parquet shard per
slide so a run can be killed and resumed without re-encoding finished slides.

Pipeline per slide
-------------------
1. Read WSI properties (native mpp); refuse to guess if the property is
   missing rather than assume 40x.
2. Low-res tissue mask from the thumbnail (HSV saturation + Otsu + a
   morphological close) -- a simplified stand-in for CLAM's segmentation,
   adequate at laptop scale; swap for real CLAM at hpc scale.
3. Non-overlapping candidate grid at `patch_px` and the profile's target mpp,
   gated by tissue fraction. If more candidates remain than
   `max_patches_per_slide`, subsample uniformly across the grid (not the
   first N encountered) so coverage stays spread across the whole slide.
4. Per surviving patch: colour-based artefact filtering (blur via Laplacian
   variance, pen-ink via an HSV hue/saturation band, folds via a near-black
   pixel fraction).
5. Macenko stain normalisation against a fixed reference stain matrix, so
   every patch is normalised to the same reference regardless of slide.
6. Batch through the encoder's OWN preprocessing + forward pass.
7. Append rows to `{outdir}/{slide_id}.parquet`: slide_id, x_um, y_um,
   patch_x, patch_y, emb_0..emb_{D-1}.

Schema matches the embeddings half of what scripts/run_final_benchmark.py
expects. That script additionally expects a `label` column -- that is joined
on separately during stage2 label transfer, not produced here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Standard Macenko (2009) H&E reference stain matrix and max concentrations,
# used across the field as a fixed normalisation TARGET (not fit per-slide).
# Columns are [haematoxylin, eosin] optical-density unit vectors.
REFERENCE_STAIN_MATRIX = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581],
])
REFERENCE_MAX_CONC = np.array([1.9705, 1.0308])

_MPP_PROPERTY = "openslide.mpp-x"  # == openslide.PROPERTY_NAME_MPP_X; avoids
                                    # importing the openslide binding just to
                                    # read a constant, so duck-typed/fake
                                    # slide objects (tests) need no dependency
                                    # on the openslide package at all.


@dataclass
class ExtractConfig:
    """Everything tunable for tiling/filtering/encoding. From config/*.yaml."""

    patch_px: int = 224                    # [PAPER]
    stride_px: int = 224                   # [PAPER] non-overlapping
    target_mpp: float = 0.50               # laptop: 20x. hpc: 0.25 (40x) [PAPER]
    min_tissue_frac: float = 0.40
    max_patches_per_slide: int | None = 4000
    batch_size: int = 16
    blur_laplacian_min: float = 15.0
    pen_ink_max_frac: float = 0.05
    fold_dark_max_frac: float = 0.20
    stain_norm: str = "macenko"            # macenko | none
    seed: int = 42


@dataclass
class SlidePatch:
    slide_id: str
    x_um: float
    y_um: float
    patch_x: int
    patch_y: int
    image: np.ndarray  # (patch_px, patch_px, 3) uint8, RGB


# --------------------------------------------------------------- tissue mask


def build_tissue_mask(thumb_rgb: np.ndarray) -> np.ndarray:
    """Binary tissue mask from an RGB thumbnail. Simplified CLAM stand-in.

    HSV saturation, Otsu-thresholded, median-blurred first to suppress
    salt-and-pepper noise, then morphologically closed to fill small holes.
    """
    import cv2

    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 7)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask > 0


# ----------------------------------------------------------- artefact filter


def artefact_flags(patch_rgb: np.ndarray, cfg: ExtractConfig) -> dict[str, float]:
    """Colour-based blur / pen-ink / fold signal on one RGB patch."""
    import cv2

    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # pen ink: strongly saturated green/blue marker hues (OpenCV hue 0-179).
    # Stops at 130, short of ~140-170 where H&E purple/pink tissue itself
    # lives (hematoxylin nuclei skew violet) -- a wider band would flag
    # nucleus-dense tissue as "pen ink".
    pen_frac = float(((sat > 60) & (hue > 35) & (hue < 130)).mean())
    # folds and heavy ink often show as very dark, low-value regions
    fold_frac = float((val < 40).mean())

    return {"blur_var": blur_var, "pen_frac": pen_frac, "fold_frac": fold_frac}


def is_artefact(flags: dict[str, float], cfg: ExtractConfig) -> bool:
    return (
        flags["blur_var"] < cfg.blur_laplacian_min
        or flags["pen_frac"] > cfg.pen_ink_max_frac
        or flags["fold_frac"] > cfg.fold_dark_max_frac
    )


# --------------------------------------------------------- stain normalisation


def macenko_normalize(
    patch_rgb: np.ndarray,
    beta: float = 0.15,
    alpha_percentile: float = 1.0,
    reference_stain: np.ndarray = REFERENCE_STAIN_MATRIX,
    reference_max_conc: np.ndarray = REFERENCE_MAX_CONC,
) -> np.ndarray:
    """Macenko (2009) SVD-based stain normalisation against a fixed reference.

    Optical density -> drop near-white pixels -> plane fit on the top two
    OD eigenvectors -> robust angular extremes give the two stain vectors ->
    deconvolve concentrations -> rescale to the reference's max
    concentrations -> reconstruct in the reference stain basis.

    Patches with too little tissue signal to estimate stains (e.g. mostly
    background) pass through unchanged rather than raise or fabricate stains.
    """
    img = patch_rgb.astype(np.float64)
    od = -np.log((img.reshape(-1, 3) + 1.0) / 256.0)
    tissue = ~np.any(od < beta, axis=1)
    od_thresh = od[tissue]
    if od_thresh.shape[0] < 10:
        return patch_rgb

    cov = np.cov(od_thresh.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, np.argsort(eigvals)[-2:]]
    proj = od_thresh @ top2
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    lo = np.percentile(angles, alpha_percentile)
    hi = np.percentile(angles, 100 - alpha_percentile)

    def _angle_to_stain(theta: float) -> np.ndarray:
        v = top2 @ np.array([np.cos(theta), np.sin(theta)])
        return v / (np.linalg.norm(v) + 1e-12)

    v1, v2 = _angle_to_stain(lo), _angle_to_stain(hi)
    # consistent channel ordering across all patches, not a biological claim
    # about which vector is "truly" haematoxylin vs eosin
    stain = np.column_stack([v1, v2])
    if stain[0, 0] < stain[0, 1]:
        stain = stain[:, ::-1]
    stain = np.abs(stain)

    conc, *_ = np.linalg.lstsq(stain, od.T, rcond=None)
    max_conc = np.percentile(conc, 99, axis=1)
    max_conc[max_conc == 0] = 1e-8
    conc_norm = conc * (reference_max_conc / max_conc)[:, None]

    od_norm = reference_stain @ conc_norm
    rgb_norm = 256.0 * np.exp(-od_norm) - 1.0
    return np.clip(rgb_norm, 0, 255).reshape(patch_rgb.shape).astype(np.uint8)


# --------------------------------------------------------------------- tiling


def _open_openslide(path: str):
    import openslide

    return openslide.OpenSlide(path)


def _candidate_grid(
    slide, cfg: ExtractConfig, native_patch_px: int, native_stride_px: int,
) -> list[tuple[int, int]]:
    """Cheap, mask-only pass: tissue-gated tile origins, no pixel reads yet."""
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((max(1, w // 32), max(1, h // 32))).convert("RGB")
    tissue_mask = build_tissue_mask(np.array(thumb))
    scale_y = tissue_mask.shape[0] / h
    scale_x = tissue_mask.shape[1] / w

    coords = []
    for y0 in range(0, h - native_patch_px + 1, native_stride_px):
        my0 = int(y0 * scale_y)
        my1 = max(my0 + 1, int((y0 + native_patch_px) * scale_y))
        for x0 in range(0, w - native_patch_px + 1, native_stride_px):
            mx0 = int(x0 * scale_x)
            mx1 = max(mx0 + 1, int((x0 + native_patch_px) * scale_x))
            tile_mask = tissue_mask[my0:my1, mx0:mx1]
            if tile_mask.size and tile_mask.mean() >= cfg.min_tissue_frac:
                coords.append((x0, y0))
    return coords


def _grid_subsample(
    coords: list[tuple[int, int]], max_n: int | None,
) -> list[tuple[int, int]]:
    """Uniformly subsample across the ordered grid, not the first N found.

    Truncating a raster-order list biases coverage toward one corner of the
    slide; an evenly-spaced index selection keeps spatial coverage intact.
    """
    if max_n is None or len(coords) <= max_n:
        return coords
    idx = sorted(set(np.linspace(0, len(coords) - 1, max_n).round().astype(int)))
    return [coords[i] for i in idx]


def iter_slide_patches(
    slide_source, cfg: ExtractConfig, slide_id: str | None = None,
) -> Iterator[SlidePatch]:
    """Stream tissue patches from one WSI. Never holds the WSI in RAM.

    `slide_source` is a path (opened via openslide) or an already-opened
    slide-like object exposing `.properties`, `.dimensions`,
    `.get_thumbnail(size)`, `.read_region(loc, level, size)`, `.close()` --
    the same surface openslide.OpenSlide exposes. Tests inject a duck-typed
    fake backed by a synthetic array so no WSI file or the openslide package
    itself is required to exercise the tiling/filtering logic.
    """
    is_path = isinstance(slide_source, (str, Path))
    slide = _open_openslide(slide_source) if is_path else slide_source
    sid = slide_id or (Path(slide_source).stem if is_path else "slide")

    native_mpp = slide.properties.get(_MPP_PROPERTY) or slide.properties.get("aperio.MPP")
    if native_mpp is None:
        raise ValueError(f"{sid}: no MPP property, refusing to guess resolution")
    native_mpp = float(native_mpp)

    w, h = slide.dimensions
    native_patch_px = max(1, round(cfg.patch_px * cfg.target_mpp / native_mpp))
    native_stride_px = max(1, round(cfg.stride_px * cfg.target_mpp / native_mpp))

    coords = _candidate_grid(slide, cfg, native_patch_px, native_stride_px)
    coords = _grid_subsample(coords, cfg.max_patches_per_slide)
    logger.info("%s: %d candidate tiles after tissue gate and grid-subsample",
                sid, len(coords))

    n_yielded = 0
    for x0, y0 in coords:
        region = slide.read_region((x0, y0), 0, (native_patch_px, native_patch_px))
        patch = np.array(region.convert("RGB"))
        if native_patch_px != cfg.patch_px:
            patch = np.array(
                Image.fromarray(patch).resize((cfg.patch_px, cfg.patch_px), Image.BILINEAR)
            )

        flags = artefact_flags(patch, cfg)
        if is_artefact(flags, cfg):
            continue

        yield SlidePatch(
            slide_id=sid,
            x_um=x0 * native_mpp,
            y_um=y0 * native_mpp,
            patch_x=x0 // native_stride_px,
            patch_y=y0 // native_stride_px,
            image=patch,
        )
        n_yielded += 1

    if is_path:
        slide.close()
    logger.info("%s: %d patches survived artefact filtering", sid, n_yielded)


# --------------------------------------------------------------- encode + i/o


def encode_slide(
    slide_source, model, preprocess: Callable, cfg: ExtractConfig,
    slide_id: str | None = None,
) -> pd.DataFrame:
    """Tile, filter, normalise and encode every qualifying patch on one slide."""
    rows: list[dict] = []
    batch_imgs: list[np.ndarray] = []
    batch_meta: list[SlidePatch] = []

    def _flush() -> None:
        if not batch_imgs:
            return
        tensors = torch.stack([preprocess(Image.fromarray(im)) for im in batch_imgs])
        with torch.no_grad():
            emb = model(tensors)
        emb = emb.detach().cpu().numpy()
        for meta, vec in zip(batch_meta, emb):
            row = {
                "slide_id": meta.slide_id, "x_um": meta.x_um, "y_um": meta.y_um,
                "patch_x": meta.patch_x, "patch_y": meta.patch_y,
            }
            row.update({f"emb_{i}": float(v) for i, v in enumerate(vec)})
            rows.append(row)
        batch_imgs.clear()
        batch_meta.clear()

    for patch in iter_slide_patches(slide_source, cfg, slide_id=slide_id):
        img = macenko_normalize(patch.image) if cfg.stain_norm == "macenko" else patch.image
        batch_imgs.append(img)
        batch_meta.append(patch)
        if len(batch_imgs) >= cfg.batch_size:
            _flush()
    _flush()

    return pd.DataFrame(rows)


def extract_embeddings(
    slide_sources: list,
    model,
    preprocess: Callable,
    cfg: ExtractConfig,
    outdir: str,
    force: bool = False,
) -> None:
    """Resumable, per-slide streaming extraction to `{outdir}/{slide_id}.parquet`.

    Skips any slide whose shard already exists unless `force`. Downstream
    code reads the whole cohort at once: `pd.read_parquet(outdir)` transparently
    concatenates every shard in the directory.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for source in slide_sources:
        sid = (Path(source).stem if isinstance(source, (str, Path))
               else getattr(source, "slide_id", None))
        if sid is None:
            raise ValueError("slide_sources must be paths, or objects with a .slide_id")
        shard = out / f"{sid}.parquet"
        if shard.exists() and not force:
            logger.info("%s: shard exists, skipping (--force to redo)", sid)
            continue

        logger.info("encoding %s", sid)
        df = encode_slide(source, model, preprocess, cfg, slide_id=sid)
        if df.empty:
            logger.warning("%s: no qualifying patches, no shard written", sid)
            continue

        # Write to a temp file then atomically rename. A shard is the marker
        # this slide is done -- if the process is killed mid-write, a plain
        # `to_parquet(shard, ...)` would leave a truncated file AT the final
        # path, and the next run would see `shard.exists()` and skip it,
        # silently treating a corrupt/partial file as complete.
        tmp = shard.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(shard)
        logger.info("%s: wrote %d patches -> %s", sid, len(df), shard)
