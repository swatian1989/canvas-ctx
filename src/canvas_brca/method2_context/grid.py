"""2D and 3D feature grids over patch embeddings.

Three ways to give a patch classifier spatial context, all on cached embeddings:

    graph    k-NN deep-set + attention        (context_model.py)
    grid2d   local W x W lattice window       TITAN-style 2D feature grid
    grid3d   local S x W x W multi-scale cube

The 2D grid
-----------
Patches sit on a regular stride lattice, so patch embeddings can be arranged
into a (D, H, W) tensor replicating their tissue positions, which is what
TITAN does at slide level. Here we keep it patch-level for a like-for-like
comparison against CANVAS: each target patch gets a **local** W x W window
centred on itself. A whole-slide grid would change the task from patch
classification to slide classification and break the comparison.

Why a grid beats k-NN even at matched neighbour count: the grid preserves
*orientation*. k-NN with k=48 and a 7x7 window see the same 48 neighbours, but
the grid knows which one is north. Tumour-stroma interfaces, duct walls and
invasive fronts are directional structures; a permutation-invariant pooler
throws that away by construction.

The 3D grid
-----------
The third axis is **magnification scale**, not physical depth. Habitats are
hierarchical: nuclear detail at 40x, glandular architecture at 10x, compartment
structure at 2.5x. CANVAS looks at one scale only. A (D, S, W, W) cube lets a
3D convolution mix evidence across scales at matched spatial positions.

Two ways to fill the scale axis:

    scale_mode="pool"     average-pool the fine grid in 2x2, 4x4 blocks.
                          FREE, no extra encoding. Approximate: pooled
                          embeddings are not the same as embeddings of a
                          downsampled larger region, because the encoder is
                          non-linear. Default on the laptop profile.

    scale_mode="encode"   encode 448 and 896 px regions downsampled to 224 and
                          embed them separately. Faithful, costs 3x encoding
                          time. Use when you have GPU time.

State which mode you used. They are not equivalent and a reviewer will ask.

True z-axis 3D (serial sections stacked into a tissue volume) is a different
thing and is not available for TCGA, which is single-section. If you ever get
serial sections or a 3D imaging cohort, the same Grid3D code applies with
scale_axis replaced by z_axis; only the grid builder changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

EMPTY = -1  # lattice slot with no patch (off tissue, artefact-filtered, or edge)


@dataclass
class GridConfig:
    """Configuration for local grid windows."""

    window: int = 7            # W x W patches, must be odd so a centre exists
    n_scales: int = 3          # 1 => 2D grid only; >1 => 3D cube
    scale_mode: str = "pool"   # "pool" (free) or "encode" (faithful)
    stride_um: float | None = None   # inferred from coordinates if None

    def __post_init__(self) -> None:
        if self.window % 2 == 0:
            raise ValueError(f"window must be odd, got {self.window}")
        if self.scale_mode not in ("pool", "encode"):
            raise ValueError("scale_mode must be 'pool' or 'encode'")


def infer_stride(coords: np.ndarray) -> float:
    """Infer the tiling stride in microns from patch centroids.

    Uses the modal nearest-neighbour distance rather than the minimum, which is
    robust to a few patches sitting off-lattice after artefact filtering.
    """
    from scipy.spatial import cKDTree

    if len(coords) < 3:
        raise ValueError("need >=3 patches to infer stride")
    d, _ = cKDTree(coords).query(coords, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    hist, edges = np.histogram(nn, bins=50)
    stride = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
    logger.debug("inferred stride %.1f um (median NN %.1f)", stride, np.median(nn))
    return stride


def to_lattice(coords: np.ndarray, stride: float) -> tuple[np.ndarray, np.ndarray]:
    """Map patch centroids to integer (row, col) lattice positions."""
    origin = coords.min(axis=0)
    rc = np.rint((coords - origin) / stride).astype(np.int64)
    return rc[:, 1], rc[:, 0]   # row from y, col from x


def build_slide_lattice(
    coords: np.ndarray, stride: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense lattice for one slide mapping (row, col) -> patch row index.

    Returns
    -------
    lattice
        (H, W) int32 array of patch indices, EMPTY where no patch exists.
    rows, cols
        Lattice position of each patch, aligned to ``coords``.
    """
    rows, cols = to_lattice(coords, stride)
    h, w = int(rows.max()) + 1, int(cols.max()) + 1
    lattice = np.full((h, w), EMPTY, dtype=np.int32)

    collisions = 0
    for i, (r, c) in enumerate(zip(rows, cols)):
        if lattice[r, c] != EMPTY:
            collisions += 1
        lattice[r, c] = i
    if collisions:
        logger.warning(
            "%d lattice collisions (%.1f%%). Two patches mapped to one slot, so "
            "the stride is probably wrong or the tiling was overlapping. The "
            "grid assumes non-overlapping tiles.",
            collisions, 100 * collisions / len(coords))
    occupancy = float((lattice != EMPTY).mean())
    logger.info("lattice %dx%d, %.0f%% occupied", h, w, occupancy * 100)
    return lattice, rows, cols


def extract_windows(
    lattice: np.ndarray, rows: np.ndarray, cols: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Local W x W index windows centred on each patch.

    Returns
    -------
    idx
        (n, W, W) int32 patch indices. Out-of-bounds and empty slots hold 0 and
        must be masked; never gather them unmasked or edge patches get context
        from patch 0 of the slide.
    mask
        (n, W, W) bool, True where the slot holds a real patch. The centre is
        always True by construction.
    """
    n, half = len(rows), window // 2
    h, w = lattice.shape
    padded = np.full((h + 2 * half, w + 2 * half), EMPTY, dtype=np.int32)
    padded[half:half + h, half:half + w] = lattice

    idx = np.zeros((n, window, window), dtype=np.int32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        idx[i] = padded[r:r + window, c:c + window]

    mask = idx != EMPTY
    idx = np.where(mask, idx, 0)

    centre = window // 2
    assert mask[:, centre, centre].all(), "centre slot must always be valid"
    logger.info("windows %dx%d, mean %.1f/%d valid slots",
                window, window, float(mask.sum(axis=(1, 2)).mean()), window ** 2)
    return idx, mask


def pyramid_pool_embeddings(
    embeddings: np.ndarray, lattice: np.ndarray, n_scales: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build coarser embedding grids by average pooling the fine lattice.

    Scale 0 is the native lattice. Scale s pools 2^s x 2^s lattice blocks,
    averaging over occupied slots only.

    This is the free path for the 3D cube: no additional encoding. It
    approximates "the embedding of a 2^s-larger region" by "the average of the
    embeddings of its parts", which is wrong in general because encoders are
    non-linear, but empirically captures coarse tissue composition well. Use
    scale_mode="encode" when fidelity matters more than compute.

    Returns
    -------
    list of (pooled_embeddings, pooled_lattice) per scale.
    """
    out = [(embeddings, lattice)]
    dim = embeddings.shape[1]

    for s in range(1, n_scales):
        f = 2 ** s
        h, w = lattice.shape
        ph, pw = int(np.ceil(h / f)), int(np.ceil(w / f))

        acc = np.zeros((ph * pw, dim), dtype=np.float64)
        cnt = np.zeros(ph * pw, dtype=np.int64)

        occ_r, occ_c = np.nonzero(lattice != EMPTY)
        flat = (occ_r // f) * pw + (occ_c // f)
        np.add.at(acc, flat, embeddings[lattice[occ_r, occ_c]])
        np.add.at(cnt, flat, 1)

        keep = cnt > 0
        pooled = np.zeros((int(keep.sum()), dim), dtype=np.float32)
        pooled[:] = (acc[keep] / cnt[keep, None]).astype(np.float32)

        p_lat = np.full((ph, pw), EMPTY, dtype=np.int32)
        p_lat.ravel()[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        out.append((pooled, p_lat))
        logger.debug("scale %d: %dx%d lattice, %d occupied", s, ph, pw, int(keep.sum()))

    return out


def build_grid_index(
    embeddings: np.ndarray,
    coords: np.ndarray,
    slide_ids: np.ndarray,
    cfg: GridConfig,
) -> dict:
    """Build 2D or 3D grid windows for every patch, per slide.

    Returns a dict with:
        scale_embeddings  list of (n_s, D) arrays, one per scale, concatenated
                          across slides
        idx               (n, S, W, W) int32 into the corresponding scale array
        mask              (n, S, W, W) bool
        offsets           (S,) list of per-scale row offsets per slide

    Context never crosses slides: lattices are built independently per slide.
    """
    n, dim = embeddings.shape
    W, S = cfg.window, cfg.n_scales

    scale_banks: list[list[np.ndarray]] = [[] for _ in range(S)]
    scale_counts = [0] * S
    idx = np.zeros((n, S, W, W), dtype=np.int32)
    mask = np.zeros((n, S, W, W), dtype=bool)

    for sid in np.unique(slide_ids):
        sel = np.flatnonzero(slide_ids == sid)
        if len(sel) < 3:
            mask[sel, :, W // 2, W // 2] = True
            continue

        stride = cfg.stride_um or infer_stride(coords[sel])
        lattice, rows, cols = build_slide_lattice(coords[sel], stride)
        pyramid = pyramid_pool_embeddings(embeddings[sel], lattice, S)

        for s, (emb_s, lat_s) in enumerate(pyramid):
            f = 2 ** s
            r_s, c_s = rows // f, cols // f
            w_idx, w_mask = extract_windows(lat_s, r_s, c_s, W)
            idx[sel, s] = w_idx + scale_counts[s]      # offset into global bank
            mask[sel, s] = w_mask
            scale_banks[s].append(emb_s)
            scale_counts[s] += len(emb_s)

    scale_embeddings = [
        np.concatenate(b).astype(np.float32) if b else np.zeros((1, dim), np.float32)
        for b in scale_banks
    ]
    logger.info("grid: %d patches, %d scales, window %d, bank sizes %s",
                n, S, W, [len(e) for e in scale_embeddings])
    return {
        "scale_embeddings": scale_embeddings,
        "idx": idx,
        "mask": mask,
        "window": W,
        "n_scales": S,
    }
