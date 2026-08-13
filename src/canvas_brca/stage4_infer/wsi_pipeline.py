"""Stage 4: WSI preprocessing, tumour detection, compartments, habitat inference.

Order of operations on a TCGA-BRCA diagnostic slide:

  1. Read slide properties, confirm native magnification. Never assume 40x.
  2. Artefact filtering: pen marks, tissue folds, blur, by colour-based rules.
  3. Tissue segmentation (CLAM) and non-overlapping 224x224 tiling at the
     target mpp.
  4. Stain normalisation (Macenko).
  5. Tumour probability per patch from a fine-tuned ViT.
  6. Spatially aggregate tumour probability, cluster into tumour bulk /
     leading edge / other.
  7. Encode patches, predict habitats, write a patch-level habitat map.

Everything streams. Nothing loads a whole WSI into RAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SlideInfo:
    slide_id: str
    path: str
    native_mpp: float
    native_mag: float
    dimensions: tuple[int, int]


def read_slide_info(path: str) -> SlideInfo:
    """Read mpp and magnification from the slide, do not assume.

    TCGA slides are a mix of 40x and 20x. `openslide.PROPERTY_NAME_MPP_X` is
    usually present for Aperio SVS. If it is missing, fall back to the
    `aperio.MPP` property, and if that is missing too, skip the slide and log
    it rather than guessing.
    """
    import openslide

    slide = openslide.OpenSlide(path)
    props = slide.properties
    mpp = props.get(openslide.PROPERTY_NAME_MPP_X) or props.get("aperio.MPP")
    if mpp is None:
        raise ValueError(f"{path}: no MPP property. Cannot standardise resolution.")
    mpp = float(mpp)
    return SlideInfo(
        slide_id=path.split("/")[-1].split(".")[0],
        path=path,
        native_mpp=mpp,
        native_mag=round(10.0 / mpp) if mpp else np.nan,
        dimensions=slide.dimensions,
    )


def is_diagnostic_slide(barcode: str) -> bool:
    """TCGA slide barcodes: DX = diagnostic FFPE, TS/BS = frozen top/bottom.

    Frozen sections have freezing artefact that destroys the morphological
    signal this whole method depends on. Use DX only.
    """
    return "-DX" in barcode.upper()


def assign_compartments(
    patches: pd.DataFrame,
    n_clusters: int = 3,
    knn_k: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Partition patches into tumour bulk, leading edge, and other.

    Patch-level tumour probability is spatially smoothed over a k-NN
    neighbourhood, then clustered. Clusters are labelled by their mean tumour
    probability and by the spatial gradient of that probability:

      - high mean, low gradient   -> tumour_bulk
      - intermediate mean, high gradient -> leading_edge
      - low mean                  -> other

    Parameters
    ----------
    patches
        Columns x_um, y_um, tumour_prob.
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans

    coords = patches[["x_um", "y_um"]].to_numpy()
    prob = patches["tumour_prob"].to_numpy()

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(knn_k + 1, len(coords)))
    neigh_prob = prob[idx[:, 1:]]
    smoothed = neigh_prob.mean(axis=1)
    gradient = neigh_prob.std(axis=1)

    x = np.column_stack([smoothed, gradient])
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x)

    out = patches.copy()
    stats = pd.DataFrame({"cluster": labels, "mean": smoothed, "grad": gradient})
    summary = stats.groupby("cluster").mean()

    bulk = int(summary["mean"].idxmax())
    remaining = summary.drop(index=bulk)
    edge = int(remaining["grad"].idxmax())

    names = {bulk: "tumour_bulk", edge: "leading_edge"}
    out["compartment"] = [names.get(int(c), "other") for c in labels]
    logger.info("compartments: %s", out["compartment"].value_counts().to_dict())
    return out
