"""Stage 2: transfer spatial-omics CN labels onto H&E patches.

This replaces the CANVAS CODEX/H&E co-registration. It is the highest-risk
stage in the pipeline: if registration is off by more than a few microns, patch
labels become noise and the classifier learns nothing real while still
reporting a plausible-looking accuracy.

Pipeline
--------
1. Affine-register the spatial-omics reference channel (Xenium DAPI /
   morphology_focus) to the H&E hematoxylin channel with PALOM.
2. Project spatial cell centroids into H&E pixel space:  xdst = A @ [x, y, 1].
3. Segment nuclei on H&E (CellViT, or StarDist on the laptop profile).
4. Assign each H&E nucleus the CN of its nearest projected spatial cell,
   rejecting matches beyond 5 um.
5. Tile the slide and label each patch by the CANVAS purity rules.

Check registration residuals before step 4. Always.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_CENTROID_DISTANCE_UM = 5.0      # [PAPER]
BACKGROUND_MAX_CELLS = 5            # [PAPER] <=5 annotated cells -> background
INFORMATIVE_MIN_CELLS = 15          # [PAPER] >15 required
DOMINANT_FRAC = 0.60                # [PAPER] dominant CN >= 60%


@dataclass
class AffineTransform:
    """2x3 affine mapping spatial-omics space to H&E space."""

    matrix: np.ndarray   # shape (2, 3)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        """Project (n, 2) source coordinates into destination space."""
        homogeneous = np.column_stack([xy, np.ones(len(xy))])
        return homogeneous @ self.matrix.T


def estimate_affine_palom(spatial_ref_path: str, he_path: str) -> AffineTransform:
    """Register with PALOM and return the affine.

    TODO: implement with `palom` (https://pypi.org/project/palom/). PALOM aligns
    the reference channel to the H&E hematoxylin channel and exposes the
    transform. For 10x Xenium the supplied image alignment CSV is a reasonable
    initialisation but should be refined: 10x documents it as suitable for
    region annotation rather than sub-cellular correspondence.

    Always report the median residual on manually picked landmarks. If it
    exceeds 5 um, stop. Do not proceed to label transfer.
    """
    raise NotImplementedError("implement PALOM registration")


def assign_cn_to_he_nuclei(
    he_nuclei: pd.DataFrame,
    spatial_cells: pd.DataFrame,
    transform: AffineTransform,
    mpp: float,
    max_distance_um: float = MAX_CENTROID_DISTANCE_UM,
) -> pd.DataFrame:
    """Nearest-neighbour CN transfer from spatial cells to H&E nuclei.

    Parameters
    ----------
    he_nuclei
        Columns x_px, y_px. Output of CellViT/StarDist on the H&E.
    spatial_cells
        Columns x_um, y_um, cn.
    transform
        Affine from spatial space to H&E pixel space.
    mpp
        Microns per pixel of the H&E, for converting the distance threshold.

    Returns
    -------
    he_nuclei with added ``cn`` and ``match_distance_um``. Unmatched -> cn = -1.
    """
    from scipy.spatial import cKDTree

    projected = transform.apply(spatial_cells[["x_um", "y_um"]].to_numpy())
    tree = cKDTree(projected)
    dist_px, idx = tree.query(he_nuclei[["x_px", "y_px"]].to_numpy(), k=1)
    dist_um = dist_px * mpp

    out = he_nuclei.copy()
    cn = spatial_cells["cn"].to_numpy()[idx]
    out["cn"] = np.where(dist_um <= max_distance_um, cn, -1)
    out["match_distance_um"] = dist_um

    matched = float((out["cn"] >= 0).mean())
    logger.info("matched %.1f%% of H&E nuclei within %.1f um",
                matched * 100, max_distance_um)
    if matched < 0.30:
        logger.warning(
            "only %.1f%% matched. Registration is probably wrong, or the "
            "sections are not the same tissue plane.", matched * 100)
    return out


def label_patches(
    labelled_nuclei: pd.DataFrame,
    patch_size_px: int,
    n_habitats: int = 10,
) -> pd.DataFrame:
    """Apply the CANVAS patch purity rules exactly.

    - <= BACKGROUND_MAX_CELLS annotated cells       -> background class
    - >  INFORMATIVE_MIN_CELLS and dominant >= 60%  -> dominant CN
    - anything else                                 -> discarded
    """
    df = labelled_nuclei[labelled_nuclei["cn"] >= 0].copy()
    df["patch_x"] = (df["x_px"] // patch_size_px).astype(int)
    df["patch_y"] = (df["y_px"] // patch_size_px).astype(int)

    records = []
    for (px, py), grp in df.groupby(["patch_x", "patch_y"]):
        n = len(grp)
        counts = grp["cn"].value_counts()
        dominant = int(counts.index[0])
        frac = float(counts.iloc[0] / n)

        if n <= BACKGROUND_MAX_CELLS:
            label = n_habitats          # background class
        elif n > INFORMATIVE_MIN_CELLS and frac >= DOMINANT_FRAC:
            label = dominant
        else:
            continue                    # ambiguous, discard

        records.append({"patch_x": px, "patch_y": py, "label": label,
                        "n_cells": n, "dominant_frac": frac})

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        logger.info("retained %d patches (%d background, %d habitat-labelled)",
                    len(out), int((out["label"] == n_habitats).sum()),
                    int((out["label"] < n_habitats).sum()))
    return out
