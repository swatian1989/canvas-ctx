"""Feature extraction contract tests. Synthetic data only, no downloads."""

import numpy as np
import pandas as pd
import pytest

from canvas_brca.stage5_features.spatial_features import (
    FeatureConfig,
    diversity_features,
    extract_all_features,
    transition_feature,
)

CFG = FeatureConfig(interaction_permutations=25)


def _random_patches(n=800, seed=0, n_habitats=10):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_um": rng.uniform(0, 3000, n),
        "y_um": rng.uniform(0, 3000, n),
        "habitat": rng.integers(0, n_habitats, n),
    })


def test_exactly_262_features():
    s = extract_all_features(_random_patches(), CFG)
    assert len(s) == 262, f"expected 262 features, got {len(s)}"


def test_feature_block_counts():
    s = extract_all_features(_random_patches(), CFG)
    assert sum(k.startswith("comp_") for k in s.index) == 10
    assert sum(k.startswith("div_") for k in s.index) == 6
    assert sum(k.startswith("disp_") for k in s.index) == 90
    assert sum(k.startswith("inter_") for k in s.index) == 100
    assert sum(k.startswith("dist_") for k in s.index) == 55
    assert sum(k.startswith("transition_") for k in s.index) == 1


def test_column_order_is_stable():
    a = extract_all_features(_random_patches(seed=1), CFG)
    b = extract_all_features(_random_patches(seed=2), CFG)
    assert list(a.index) == list(b.index)


def test_composition_sums_to_one():
    s = extract_all_features(_random_patches(), CFG)
    comp = s[[k for k in s.index if k.startswith("comp_")]].astype(float)
    assert np.isclose(comp.sum(), 1.0)


def test_diversity_single_habitat():
    """One habitat: richness 1, Shannon 0, Pielou 0."""
    h = np.zeros(500, dtype=int)
    d = diversity_features(h, CFG)
    assert d["div_richness"] == 1.0
    assert np.isclose(d["div_shannon"], 0.0)
    assert np.isclose(d["div_pielou"], 0.0)


def test_transition_entropy_bounds():
    """STE cannot exceed log(n_habitats^2)."""
    s = transition_feature(
        _random_patches()[["x_um", "y_um"]].to_numpy(),
        _random_patches()["habitat"].to_numpy(),
        CFG,
    )
    assert 0 <= s["transition_ste"] <= np.log(CFG.n_habitats ** 2) + 1e-9


def test_rejects_background_class():
    """Background must be dropped before feature extraction."""
    df = _random_patches()
    df.loc[0, "habitat"] = 10          # the background label
    with pytest.raises(ValueError, match="outside"):
        extract_all_features(df, CFG)


# ---------------------------------------------- edge correction & null model


def test_ripley_edge_correction_reduces_bias():
    """On a CSR pattern, border-corrected K should sit closer to the theoretical
    pi*r^2 than the uncorrected estimate, which is biased downward."""
    from canvas_brca.stage5_features.spatial_features import _ripley_k_border
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, (3000, 2))
    tree = cKDTree(pts)
    r = 60.0
    theory = np.pi * r ** 2

    corrected = _ripley_k_border(pts, tree, pts, r)
    lam = len(pts) / (1000.0 * 1000.0)
    naive = (tree.count_neighbors(tree, r) - len(pts)) / (len(pts) * lam)

    assert abs(corrected - theory) < abs(naive - theory)


def test_ripley_returns_nan_when_radius_exceeds_window():
    from canvas_brca.stage5_features.spatial_features import _ripley_k_border
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (200, 2))
    assert np.isnan(_ripley_k_border(pts, cKDTree(pts), pts, r=500.0))


def test_clark_evans_near_one_for_csr():
    """Donnelly-corrected Clark-Evans should be close to 1 on random points."""
    from canvas_brca.stage5_features.spatial_features import _clark_evans_donnelly
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, (2000, 2))
    d, _ = cKDTree(pts).query(pts, k=2)
    lam = len(pts) / (1000.0 * 1000.0)
    r = _clark_evans_donnelly(d[:, 1], lam, pts, len(pts))
    assert 0.93 < r < 1.07, f"CSR should give ~1.0, got {r:.3f}"


def test_toroidal_null_preserves_label_counts():
    """The shift must relabel, not resample: composition is unchanged."""
    from canvas_brca.stage5_features.spatial_features import _toroidal_shift

    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1000, (500, 2))
    hab = rng.integers(0, 10, 500)
    shifted = _toroidal_shift(coords, hab, rng)
    assert len(shifted) == len(hab)
    assert set(np.unique(shifted)).issubset(set(np.unique(hab)))


def test_toroidal_null_is_stricter_than_shuffle():
    """On clustered habitats, plain shuffling destroys autocorrelation and
    declares nearly everything significant. The toroidal null must flag fewer
    pairs. This is the bug that would have produced false interaction findings."""
    rng = np.random.default_rng(0)
    n = 900
    centres = rng.uniform(0, 3000, (10, 2))
    a = rng.integers(0, 10, n)
    pts = centres[a] + rng.normal(0, 250, (n, 2))
    df = pd.DataFrame({"x_um": pts[:, 0], "y_um": pts[:, 1], "habitat": a})

    def n_sig(null_model):
        s = extract_all_features(
            df, FeatureConfig(interaction_permutations=80, null_model=null_model))
        z = s[[k for k in s.index if k.startswith("inter_")]].astype(float)
        return int((z.abs() > 1.96).sum())

    assert n_sig("toroidal") < n_sig("shuffle")


def test_null_model_default_is_toroidal():
    assert FeatureConfig().null_model == "toroidal"
