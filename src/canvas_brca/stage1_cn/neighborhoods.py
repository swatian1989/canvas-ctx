"""Cellular neighbourhood (CN) discovery from single-cell spatial data.

Implements the CANVAS stage-1 procedure:
  1. For each index cell, define a local neighbourhood of all cells within
     ``radius_um`` (paper: 40 um, which captures ~25 neighbours).
  2. Represent each neighbourhood as a cell-type composition vector.
  3. Fit spatial-LDA to decompose compositions into latent motifs.
  4. K-means on motif proportions -> cellular neighbourhoods.
  5. Sweep k = 5..20 and score with silhouette, Davies-Bouldin, and adjacent-k
     adjusted Rand index.

Input is a tidy single-cell table, one row per cell:
    image_id, cell_id, x_um, y_um, cell_type

Works on CPU. The Danenberg cohort (~1.1 M cells) fits in 16 GB if you keep the
composition matrix sparse and process images in chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("image_id", "cell_id", "x_um", "y_um", "cell_type")


@dataclass
class NeighbourhoodConfig:
    """Parameters for neighbourhood construction and CN clustering."""

    radius_um: float = 40.0
    expected_neighbours: int = 25
    min_cells_per_image: int = 200
    n_topics: int = 20
    k_min: int = 5
    k_max: int = 20
    final_k: int = 10
    n_init: int = 25
    seed: int = 42
    silhouette_sample: int = 20_000
    cell_type_order: list[str] = field(default_factory=list)
    max_span_um: float = 20_000.0   # raise only for whole-slide data; see validate_cells


def validate_cells(cells: pd.DataFrame, max_span_um: float = 20_000.0) -> None:
    """Fail loudly on the mistakes that silently ruin everything downstream.

    The most common one is coordinate units. If the table is in pixels and you
    apply a 40 um radius, every neighbourhood collapses to the index cell and
    the CNs become noise that still clusters cleanly. Check first.

    Parameters
    ----------
    max_span_um
        Upper bound on a plausible coordinate span. The 20 mm default suits
        TMA cores and Xenium regions. A whole-slide resection is legitimately
        larger (an Orion CRC section spans ~25 mm), so callers working at
        whole-slide scale must raise this DELIBERATELY rather than have the
        guard silently widened for everyone: the check exists to catch
        pixel-for-micron confusion, and that failure mode is real at every
        scale.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in cells.columns]
    if missing:
        raise ValueError(f"single-cell table missing columns: {missing}")

    if cells["cell_id"].duplicated().any() and not cells.duplicated(
        subset=["image_id", "cell_id"]
    ).any():
        logger.info("cell_id is only unique within image_id; that is fine")
    elif cells.duplicated(subset=["image_id", "cell_id"]).any():
        raise ValueError("duplicate (image_id, cell_id) pairs found")

    span = float(
        np.nanmax(
            [
                cells["x_um"].max() - cells["x_um"].min(),
                cells["y_um"].max() - cells["y_um"].min(),
            ]
        )
    )
    if span > max_span_um:
        raise ValueError(
            f"coordinate span is {span:.0f} units. That is implausible for "
            "microns on a TMA core or Xenium region. The table is probably in "
            "pixels. Convert with the acquisition pixel size before proceeding."
        )
    if span < 50:
        raise ValueError(
            f"coordinate span is only {span:.1f} units. Check units, a 40 um "
            "radius would cover the entire image."
        )
    logger.info("coordinate span %.0f um, %d cells, %d images, %d cell types",
                span, len(cells), cells["image_id"].nunique(),
                cells["cell_type"].nunique())


def build_composition_matrix(
    cells: pd.DataFrame,
    cfg: NeighbourhoodConfig,
) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    """Build the index-cell x cell-type neighbourhood composition matrix.

    Parameters
    ----------
    cells
        Tidy single-cell table. See ``REQUIRED_COLUMNS``.
    cfg
        Neighbourhood configuration.

    Returns
    -------
    counts
        Sparse (n_index_cells, n_cell_types) count matrix. Includes the index
        cell itself, matching the CANVAS definition of the local neighbourhood.
    index
        DataFrame aligned to ``counts`` rows with image_id, cell_id, x_um, y_um.
    types
        Ordered cell-type vocabulary matching ``counts`` columns.
    """
    validate_cells(cells, max_span_um=cfg.max_span_um)

    types = cfg.cell_type_order or sorted(cells["cell_type"].unique())
    type_to_col = {t: i for i, t in enumerate(types)}

    rows: list[sparse.csr_matrix] = []
    index_frames: list[pd.DataFrame] = []
    neighbour_counts: list[float] = []

    for image_id, block in cells.groupby("image_id", sort=True):
        if len(block) < cfg.min_cells_per_image:
            logger.debug("skipping %s: only %d cells", image_id, len(block))
            continue

        coords = block[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        codes = block["cell_type"].map(type_to_col).to_numpy()
        if np.isnan(codes).any():
            unknown = set(block["cell_type"]) - set(type_to_col)
            raise ValueError(f"cell types not in vocabulary: {unknown}")
        codes = codes.astype(np.int32)

        nn = NearestNeighbors(radius=cfg.radius_um, algorithm="kd_tree")
        nn.fit(coords)
        neighbours = nn.radius_neighbors(coords, return_distance=False)

        n = len(block)
        indptr = np.zeros(n + 1, dtype=np.int64)
        col_idx: list[np.ndarray] = []
        for i, nb in enumerate(neighbours):
            col_idx.append(codes[nb])
            indptr[i + 1] = indptr[i] + len(nb)
            neighbour_counts.append(len(nb))

        flat = np.concatenate(col_idx) if col_idx else np.array([], dtype=np.int32)
        data = np.ones(len(flat), dtype=np.float32)
        block_mat = sparse.csr_matrix(
            (data, flat, indptr), shape=(n, len(types))
        )
        block_mat.sum_duplicates()
        rows.append(block_mat)
        index_frames.append(
            block[["image_id", "cell_id", "x_um", "y_um", "cell_type"]].reset_index(drop=True)
        )

    if not rows:
        raise ValueError("no images passed the min_cells_per_image filter")

    counts = sparse.vstack(rows, format="csr")
    index = pd.concat(index_frames, ignore_index=True)

    median_nb = float(np.median(neighbour_counts))
    logger.info(
        "median neighbourhood size %.1f cells at r=%.0f um (paper expects ~%d)",
        median_nb, cfg.radius_um, cfg.expected_neighbours,
    )
    if not 0.4 * cfg.expected_neighbours <= median_nb <= 2.5 * cfg.expected_neighbours:
        logger.warning(
            "neighbourhood size %.1f is far from the expected %d. Either the "
            "tissue density differs from NSCLC TMAs or the radius/units are "
            "wrong. Investigate before trusting the CNs.",
            median_nb, cfg.expected_neighbours,
        )
    return counts, index, types


def fit_spatial_lda(
    counts: sparse.csr_matrix,
    cfg: NeighbourhoodConfig,
) -> np.ndarray:
    """Decompose neighbourhood compositions into latent motif proportions.

    The paper uses spatial-LDA (Chen et al.), which adds a spatial smoothness
    penalty over neighbouring index cells to vanilla LDA. If the ``spatial_lda``
    package is installed, use it. Otherwise fall back to scikit-learn's
    LatentDirichletAllocation, which is the same generative model without the
    smoothness prior.

    The fallback is acceptable for a first pass but changes the result. Record
    which path was taken in the run manifest.
    """
    try:
        from spatial_lda.online_lda import LatentDirichletAllocation as SpatialLDA  # noqa

        logger.info("using spatial_lda with smoothness penalty")
        model = SpatialLDA(
            n_components=cfg.n_topics,
            random_state=cfg.seed,
        )
        return model.fit_transform(counts)
    except ImportError:
        from sklearn.decomposition import LatentDirichletAllocation

        logger.warning(
            "spatial_lda not installed, falling back to sklearn LDA without the "
            "spatial smoothness prior. Record this deviation."
        )
        model = LatentDirichletAllocation(
            n_components=cfg.n_topics,
            learning_method="online",
            batch_size=4096,
            max_iter=20,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        return model.fit_transform(counts)


def sweep_k(
    topics: np.ndarray,
    cfg: NeighbourhoodConfig,
) -> pd.DataFrame:
    """Evaluate k = k_min..k_max with the three CANVAS selection metrics.

    Returns a DataFrame with columns k, silhouette, davies_bouldin,
    ari_vs_previous. The paper reports all three rather than optimising one, so
    present the table and choose k with biology in mind. Silhouette and
    Davies-Bouldin often disagree; a plateau in adjacent-k ARI is the strongest
    stability signal.
    """
    rng = np.random.default_rng(cfg.seed)
    sample_idx = (
        rng.choice(len(topics), cfg.silhouette_sample, replace=False)
        if len(topics) > cfg.silhouette_sample
        else np.arange(len(topics))
    )

    records: list[dict[str, float]] = []
    previous: np.ndarray | None = None

    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.seed)
        labels = km.fit_predict(topics)
        records.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(topics[sample_idx], labels[sample_idx])),
                "davies_bouldin": float(davies_bouldin_score(topics, labels)),
                "ari_vs_previous": (
                    float(adjusted_rand_score(previous, labels))
                    if previous is not None
                    else np.nan
                ),
                "inertia": float(km.inertia_),
            }
        )
        previous = labels
        logger.info("k=%d  sil=%.3f  db=%.3f  ari=%.3f",
                    k, records[-1]["silhouette"], records[-1]["davies_bouldin"],
                    records[-1]["ari_vs_previous"])

    return pd.DataFrame.from_records(records)


def assign_cns(
    topics: np.ndarray,
    index: pd.DataFrame,
    cfg: NeighbourhoodConfig,
) -> pd.DataFrame:
    """Cluster motif proportions into final CNs and attach labels to cells."""
    km = KMeans(n_clusters=cfg.final_k, n_init=cfg.n_init, random_state=cfg.seed)
    labels = km.fit_predict(topics)
    out = index.copy()
    out["cn"] = labels
    out["cn_label"] = [f"CN{i + 1:02d}" for i in labels]
    return out


def cn_lineage_enrichment(
    cn_assignments: pd.DataFrame,
    counts: sparse.csr_matrix,
    types: list[str],
) -> pd.DataFrame:
    """Mean neighbourhood composition per CN, z-scored across CNs.

    This reproduces the paper's Figure 3B heatmap and is how you name the CNs.
    In NSCLC the ten came out as tumour core, macrophage niche, B cell niche,
    stromal-fibrotic, plasma cell cluster, neutrophil-rich, tumour-immune
    interface, T cell compartment, pan-immune activation, and vasculature.
    Breast will differ. Name yours from this table, do not assume the lung names
    transfer.
    """
    dense = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    row_sums = np.maximum(dense.sum(axis=1, keepdims=True), 1.0)
    frac = dense / row_sums
    df = pd.DataFrame(frac, columns=types)
    df["cn_label"] = cn_assignments["cn_label"].to_numpy()
    profile = df.groupby("cn_label").mean()
    z = (profile - profile.mean(axis=0)) / profile.std(axis=0).replace(0, np.nan)
    return z
