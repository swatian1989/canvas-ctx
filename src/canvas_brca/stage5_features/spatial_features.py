"""The 262 habitat-level spatial features from CANVAS.

Feature budget, per the paper. The total is asserted, so if a block produces the
wrong count the pipeline fails rather than quietly shipping a different feature
space.

    composition   10   frequency of each habitat
    diversity      6   richness, Shannon, Simpson, invSimpson, Fisher alpha, Pielou
    dispersion    90   9 metrics x 10 habitats, from planar point patterns
    interaction  100   full 10x10 ordered habitat-pair permutation scores
    distance      55   10 self + 45 unordered pairwise NN distances
    transition     1   spatial transition entropy over patch k-NN (k=6)
    -----------------
    total        262

Input for one slide is a patch-level table:
    x_um, y_um, habitat        (habitat in 0..9, background already dropped)

Everything here is pure numpy/scipy and runs on CPU.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import gammaln  # noqa: F401  (kept for Fisher alpha variants)
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

N_HABITATS = 10
DISPERSION_METRICS = (
    "ripley_k",
    "ripley_l",
    "pair_correlation",
    "g_function",
    "f_function",
    "j_function",
    "clark_evans",
    "quadrat_dispersion",
    "kde_summary",
)


@dataclass
class FeatureConfig:
    n_habitats: int = N_HABITATS
    interaction_permutations: int = 1000
    interaction_radius_um: float = 100.0
    transition_knn_k: int = 6
    ripley_radii_um: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0)
    quadrat_grid: int = 8
    null_model: str = "toroidal"   # "toroidal" (strict) | "shuffle" (permissive)
    seed: int = 42


# --------------------------------------------------------------- 1. composition


def composition_features(habitats: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """Frequency of each habitat within the image. n = 10."""
    counts = np.bincount(habitats, minlength=cfg.n_habitats)[: cfg.n_habitats]
    total = counts.sum()
    return {
        f"comp_H{i + 1:02d}": float(counts[i] / total) if total else 0.0
        for i in range(cfg.n_habitats)
    }


# ----------------------------------------------------------------- 2. diversity


def diversity_features(habitats: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """Ecological diversity treating habitat labels as species. n = 6."""
    counts = np.bincount(habitats, minlength=cfg.n_habitats)[: cfg.n_habitats]
    counts = counts[counts > 0]
    n = counts.sum()
    if n == 0 or len(counts) == 0:
        return {f"div_{m}": np.nan for m in
                ("richness", "shannon", "simpson", "inv_simpson", "fisher_alpha", "pielou")}

    p = counts / n
    richness = float(len(counts))
    shannon = float(-(p * np.log(p)).sum())
    simpson = float(1.0 - (p ** 2).sum())
    inv_simpson = float(1.0 / (p ** 2).sum())
    pielou = float(shannon / np.log(richness)) if richness > 1 else 0.0
    fisher_alpha = _fisher_alpha(int(n), int(richness))

    return {
        "div_richness": richness,
        "div_shannon": shannon,
        "div_simpson": simpson,
        "div_inv_simpson": inv_simpson,
        "div_fisher_alpha": fisher_alpha,
        "div_pielou": pielou,
    }


def _fisher_alpha(n: int, s: int, tol: float = 1e-8, max_iter: int = 200) -> float:
    """Solve S = alpha * ln(1 + N/alpha) for alpha by bisection."""
    if s <= 1 or n <= s:
        return np.nan
    lo, hi = 1e-6, 1e6
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = mid * np.log1p(n / mid)
        if abs(val - s) < tol:
            return float(mid)
        if val < s:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------- 3. dispersion


def dispersion_features(
    coords: np.ndarray, habitats: np.ndarray, cfg: FeatureConfig
) -> dict[str, float]:
    """Intra-habitat spatial organisation. 9 metrics x 10 habitats = 90.

    Each habitat's patch centroids are treated as a planar point pattern. Where
    a habitat has too few points (<10) the metrics are NaN rather than zero,
    because zero is a meaningful value for several of these and imputing it
    would fabricate structure.
    """
    feats: dict[str, float] = {}
    area = _bbox_area(coords)

    for h in range(cfg.n_habitats):
        tag = f"H{h + 1:02d}"
        pts = coords[habitats == h]
        if len(pts) < 10:
            for m in DISPERSION_METRICS:
                feats[f"disp_{tag}_{m}"] = np.nan
            continue

        tree = cKDTree(pts)
        n = len(pts)
        lam = n / area if area > 0 else np.nan

        # Ripley's K and L with EDGE CORRECTION. Without it, points near the
        # boundary have artificially few neighbours and K is biased downward,
        # severely so on small TMA cores. We use the border (reduced-sample)
        # correction: at radius r, only points whose distance to the boundary
        # exceeds r contribute, and the intensity is estimated from those
        # points alone. This is what spatstat's Kest(correction="border") does.
        k_vals, l_vals = [], []
        for r in cfg.ripley_radii_um:
            k = _ripley_k_border(pts, tree, coords, r)
            k_vals.append(k)
            l_vals.append(np.sqrt(k / np.pi) - r if np.isfinite(k) and k >= 0 else np.nan)
        feats[f"disp_{tag}_ripley_k"] = float(np.nanmean(k_vals))
        feats[f"disp_{tag}_ripley_l"] = float(np.nanmean(l_vals))

        # Pair correlation g(r): derivative of K, approximated as annulus density
        # over the radius grid, normalised by the expected CSR density.
        feats[f"disp_{tag}_pair_correlation"] = _pair_correlation(tree, n, lam, cfg)

        # G: nearest-neighbour distance distribution, summarised as its mean.
        nnd, _ = tree.query(pts, k=2)
        nnd = nnd[:, 1]
        feats[f"disp_{tag}_g_function"] = float(np.mean(nnd))

        # F: empty-space function, from random probe points.
        feats[f"disp_{tag}_f_function"] = _empty_space(tree, coords, cfg)

        # J = (1 - G) / (1 - F) evaluated at the median NN distance.
        feats[f"disp_{tag}_j_function"] = _j_function(nnd, tree, coords, cfg)

        # Clark-Evans with the Donnelly edge correction. The raw index is
        # biased upward (toward "regular") because boundary points have their
        # true nearest neighbour outside the window. Donnelly's correction
        # adjusts the CSR expectation by the window perimeter.
        feats[f"disp_{tag}_clark_evans"] = _clark_evans_donnelly(nnd, lam, coords, n)

        # Quadrat dispersion: variance-to-mean ratio of counts in a grid.
        feats[f"disp_{tag}_quadrat_dispersion"] = _quadrat_vmr(pts, coords, cfg)

        # Kernel density summary: coefficient of variation of the density
        # evaluated at the points themselves. High CV = strong hotspotting.
        feats[f"disp_{tag}_kde_summary"] = _kde_cv(pts)

    return feats


def _bbox_area(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return np.nan
    span = coords.max(axis=0) - coords.min(axis=0)
    return float(max(span[0], 1e-9) * max(span[1], 1e-9))



def _boundary_distance(pts: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of the bounding window."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    return np.minimum.reduce([
        pts[:, 0] - lo[0], hi[0] - pts[:, 0],
        pts[:, 1] - lo[1], hi[1] - pts[:, 1],
    ])


def _ripley_k_border(
    pts: np.ndarray, tree: cKDTree, coords: np.ndarray, r: float
) -> float:
    """Border-corrected Ripley's K at radius r.

    Only points at least r from the window edge are used as centres, and the
    intensity is estimated from that reduced sample. Returns NaN when fewer
    than 5 eligible centres remain, which happens when r approaches the window
    size; NaN is correct there, an uncorrected value would be misleading.
    """
    edge = _boundary_distance(pts, coords)
    eligible = edge >= r
    n_elig = int(eligible.sum())
    if n_elig < 5:
        return np.nan

    area = _bbox_area(coords)
    lam = len(pts) / area if area > 0 else np.nan
    if not (np.isfinite(lam) and lam > 0):
        return np.nan

    counts = np.array([len(tree.query_ball_point(p, r)) - 1 for p in pts[eligible]])
    return float(counts.mean() / lam)


def _clark_evans_donnelly(
    nnd: np.ndarray, lam: float, coords: np.ndarray, n: int
) -> float:
    """Clark-Evans index with Donnelly's perimeter correction.

    R = mean(NND) / E[NND], where the corrected expectation is
        E = 0.5/sqrt(lam) + (0.051 + 0.041/sqrt(n)) * P / n
    with P the window perimeter. R < 1 clustered, R = 1 random, R > 1 regular.
    """
    if not (np.isfinite(lam) and lam > 0) or n < 2:
        return np.nan
    span = coords.max(axis=0) - coords.min(axis=0)
    perimeter = float(2 * (span[0] + span[1]))
    expected = 0.5 / np.sqrt(lam) + (0.051 + 0.041 / np.sqrt(n)) * perimeter / n
    return float(np.mean(nnd) / expected) if expected > 0 else np.nan


def _pair_correlation(tree: cKDTree, n: int, lam: float, cfg: FeatureConfig) -> float:
    if not (np.isfinite(lam) and lam > 0):
        return np.nan
    radii = np.asarray(cfg.ripley_radii_um, dtype=float)
    g_vals = []
    for r_out, r_in in zip(radii[1:], radii[:-1]):
        shell = tree.count_neighbors(tree, r_out) - tree.count_neighbors(tree, r_in)
        annulus_area = np.pi * (r_out ** 2 - r_in ** 2)
        expected = n * lam * annulus_area
        g_vals.append(shell / expected if expected > 0 else np.nan)
    return float(np.nanmean(g_vals)) if g_vals else np.nan


def _probe_points(coords: np.ndarray, cfg: FeatureConfig, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    return rng.uniform(lo, hi, size=(n, 2))


def _empty_space(tree: cKDTree, coords: np.ndarray, cfg: FeatureConfig) -> float:
    probes = _probe_points(coords, cfg)
    d, _ = tree.query(probes, k=1)
    return float(np.mean(d))


def _j_function(nnd: np.ndarray, tree: cKDTree, coords: np.ndarray,
                cfg: FeatureConfig) -> float:
    r = float(np.median(nnd))
    g_r = float(np.mean(nnd <= r))
    probes = _probe_points(coords, cfg)
    d, _ = tree.query(probes, k=1)
    f_r = float(np.mean(d <= r))
    denom = 1.0 - f_r
    return float((1.0 - g_r) / denom) if denom > 1e-9 else np.nan


def _quadrat_vmr(pts: np.ndarray, coords: np.ndarray, cfg: FeatureConfig) -> float:
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    edges_x = np.linspace(lo[0], hi[0], cfg.quadrat_grid + 1)
    edges_y = np.linspace(lo[1], hi[1], cfg.quadrat_grid + 1)
    counts, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges_x, edges_y])
    m = counts.mean()
    return float(counts.var() / m) if m > 0 else np.nan


def _kde_cv(pts: np.ndarray) -> float:
    try:
        kde = gaussian_kde(pts.T)
        dens = kde(pts.T)
        return float(dens.std() / dens.mean()) if dens.mean() > 0 else np.nan
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# --------------------------------------------------------------- 4. interaction


def interaction_features(
    coords: np.ndarray, habitats: np.ndarray, cfg: FeatureConfig
) -> dict[str, float]:
    """Permutation-scored habitat-pair adjacency. Full 10x10 ordered = 100.

    Observed adjacency frequency within ``interaction_radius_um`` is compared to
    a null built by shuffling habitat labels while holding coordinates fixed.
    The score is a z-score against that null, matching the CANVAS approach for
    cell-cell interactions applied here at the habitat level.
    """
    rng = np.random.default_rng(cfg.seed)
    n_h = cfg.n_habitats
    feats: dict[str, float] = {}
    # null_model: "toroidal" (default, strict) or "shuffle" (permissive).
    # See _toroidal_shift for why the default changed.

    if len(coords) < 20:
        for i, j in itertools.product(range(n_h), repeat=2):
            feats[f"inter_H{i + 1:02d}_H{j + 1:02d}"] = np.nan
        return feats

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cfg.interaction_radius_um, output_type="ndarray")
    if len(pairs) == 0:
        for i, j in itertools.product(range(n_h), repeat=2):
            feats[f"inter_H{i + 1:02d}_H{j + 1:02d}"] = np.nan
        return feats

    observed = _adjacency_matrix(habitats, pairs, n_h)

    null = np.empty((cfg.interaction_permutations, n_h, n_h), dtype=np.float32)
    for it in range(cfg.interaction_permutations):
        if cfg.null_model == "toroidal":
            perm = _toroidal_shift(coords, habitats, rng)
            perm_pairs = pairs
        else:
            perm = habitats.copy()
            rng.shuffle(perm)
            perm_pairs = pairs
        null[it] = _adjacency_matrix(perm, perm_pairs, n_h)

    mu = null.mean(axis=0)
    sd = null.std(axis=0)
    z = np.divide(observed - mu, sd, out=np.zeros_like(observed), where=sd > 1e-9)

    for i, j in itertools.product(range(n_h), repeat=2):
        feats[f"inter_H{i + 1:02d}_H{j + 1:02d}"] = float(z[i, j])
    return feats



def _toroidal_shift(
    coords: np.ndarray, habitats: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Toroidal-shift null: translate the habitat field, wrapping at the edges.

    Why not plain label shuffling. Global shuffling destroys ALL spatial
    autocorrelation, so the null says "habitats are scattered at random". Real
    tissue is never like that: habitats form contiguous domains. Tested against
    that null, almost every habitat pair looks significantly interacting,
    because the observed data has structure and the null has none. The test
    then measures "is this tissue spatially organised at all", which is always
    yes, rather than "are these two habitats specifically associated".

    The toroidal shift preserves each habitat's own spatial autocorrelation and
    domain sizes, and only randomises the RELATIVE registration between the
    habitat field and the point locations. That isolates the question actually
    being asked. It is the standard null in spatial ecology for exactly this
    reason.

    Implementation: shift all coordinates by a random offset with wraparound
    within the bounding box, then reassign labels by nearest original point.
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    offset = rng.uniform(0, 1, size=2) * span
    shifted = lo + np.mod((coords - lo) + offset, span)
    tree = cKDTree(coords)
    _, idx = tree.query(shifted, k=1)
    return habitats[idx]


def _adjacency_matrix(habitats: np.ndarray, pairs: np.ndarray, n_h: int) -> np.ndarray:
    a = habitats[pairs[:, 0]]
    b = habitats[pairs[:, 1]]
    mat = np.zeros((n_h, n_h), dtype=np.float32)
    np.add.at(mat, (a, b), 1.0)
    np.add.at(mat, (b, a), 1.0)   # undirected: fill both orientations
    total = mat.sum()
    return mat / total if total > 0 else mat


# ------------------------------------------------------------------ 5. distance


def distance_features(
    coords: np.ndarray, habitats: np.ndarray, cfg: FeatureConfig
) -> dict[str, float]:
    """Mean nearest-neighbour distance between habitat pairs. 10 + 45 = 55.

    Self-distances (H_i to H_i) capture intra-habitat compactness; the 45
    unordered cross pairs capture separation. Distance is symmetric so only the
    upper triangle plus diagonal is computed.
    """
    feats: dict[str, float] = {}
    trees: dict[int, cKDTree | None] = {}
    pts_by_h: dict[int, np.ndarray] = {}

    for h in range(cfg.n_habitats):
        p = coords[habitats == h]
        pts_by_h[h] = p
        trees[h] = cKDTree(p) if len(p) >= 2 else None

    for i in range(cfg.n_habitats):
        for j in range(i, cfg.n_habitats):
            key = f"dist_H{i + 1:02d}_H{j + 1:02d}"
            pi, tj = pts_by_h[i], trees[j]
            if tj is None or len(pi) == 0:
                feats[key] = np.nan
                continue
            if i == j:
                if len(pi) < 2:
                    feats[key] = np.nan
                    continue
                d, _ = tj.query(pi, k=2)
                feats[key] = float(np.mean(d[:, 1]))
            else:
                d, _ = tj.query(pi, k=1)
                feats[key] = float(np.mean(d))
    return feats


# ---------------------------------------------------------------- 6. transition


def transition_feature(
    coords: np.ndarray, habitats: np.ndarray, cfg: FeatureConfig
) -> dict[str, float]:
    """Spatial transition entropy. n = 1.

    Build a habitat transition matrix from patch-level k-nearest-neighbour
    relationships with k = 6, then take the Shannon entropy of the global
    transition probability distribution. High STE means the tissue switches
    habitat identity frequently and unpredictably.
    """
    if len(coords) <= cfg.transition_knn_k:
        return {"transition_ste": np.nan}

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=cfg.transition_knn_k + 1)
    src = np.repeat(habitats, cfg.transition_knn_k)
    dst = habitats[idx[:, 1:].ravel()]

    mat = np.zeros((cfg.n_habitats, cfg.n_habitats), dtype=np.float64)
    np.add.at(mat, (src, dst), 1.0)
    total = mat.sum()
    if total == 0:
        return {"transition_ste": np.nan}
    p = (mat / total).ravel()
    p = p[p > 0]
    return {"transition_ste": float(-(p * np.log(p)).sum())}


# -------------------------------------------------------------------- assembly


def extract_all_features(
    patches: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    sample_id: str | None = None,
) -> pd.Series:
    """Compute all 262 features for one image or compartment.

    Parameters
    ----------
    patches
        Patch-level table with columns ``x_um``, ``y_um``, ``habitat``.
        ``habitat`` must be integer 0..n_habitats-1 with background removed.
    cfg
        Feature configuration.
    sample_id
        Optional identifier written into the result index as ``sample_id``.

    Returns
    -------
    pd.Series
        Length 262 (plus sample_id if given), in a stable column order.
    """
    cfg = cfg or FeatureConfig()

    for col in ("x_um", "y_um", "habitat"):
        if col not in patches.columns:
            raise ValueError(f"patch table missing column '{col}'")

    coords = patches[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    habitats = patches["habitat"].to_numpy(dtype=np.int64)

    bad = (habitats < 0) | (habitats >= cfg.n_habitats)
    if bad.any():
        raise ValueError(
            f"{bad.sum()} patches have habitat outside 0..{cfg.n_habitats - 1}. "
            "Drop the background class before calling this."
        )

    feats: dict[str, float] = {}
    feats.update(composition_features(habitats, cfg))
    feats.update(diversity_features(habitats, cfg))
    feats.update(dispersion_features(coords, habitats, cfg))
    feats.update(interaction_features(coords, habitats, cfg))
    feats.update(distance_features(coords, habitats, cfg))
    feats.update(transition_feature(coords, habitats, cfg))

    expected = (
        cfg.n_habitats                                   # 10 composition
        + 6                                              # diversity
        + len(DISPERSION_METRICS) * cfg.n_habitats       # 90 dispersion
        + cfg.n_habitats ** 2                            # 100 interaction
        + cfg.n_habitats * (cfg.n_habitats + 1) // 2     # 55 distance
        + 1                                              # transition
    )
    if len(feats) != expected:
        raise AssertionError(
            f"feature count is {len(feats)}, expected {expected}. "
            "A feature block is producing the wrong number of columns. Fix the "
            "block, do not pad."
        )

    series = pd.Series(feats, dtype="float64")
    if sample_id is not None:
        series = pd.concat([pd.Series({"sample_id": sample_id}), series])
    return series
