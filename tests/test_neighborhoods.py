"""CN discovery contract tests."""

import numpy as np
import pandas as pd
import pytest

from canvas_brca.stage1_cn.neighborhoods import (
    NeighbourhoodConfig,
    build_composition_matrix,
    validate_cells,
)

TYPES = ["Tumour", "CD8_T", "B_cell", "Macrophage", "CAF", "Endothelial"]


def _cells(n_img=3, n_cell=800, span=800.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        for c in range(n_cell):
            x, y = rng.uniform(0, span, 2)
            q = int(x > span / 2)
            p = np.full(len(TYPES), 0.05)
            p[q] = 0.75
            p /= p.sum()
            rows.append((f"img{i}", f"{i}_{c}", x, y, rng.choice(TYPES, p=p)))
    return pd.DataFrame(rows, columns=["image_id", "cell_id", "x_um", "y_um", "cell_type"])


def test_rejects_pixel_coordinates():
    """A 50,000-unit span is pixels, not microns. This must fail loudly."""
    cells = _cells(span=50_000.0)
    with pytest.raises(ValueError, match="pixels"):
        validate_cells(cells)


def test_rejects_missing_columns():
    cells = _cells().drop(columns=["cell_type"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_cells(cells)


def test_composition_matrix_shape():
    cells = _cells()
    cfg = NeighbourhoodConfig(cell_type_order=TYPES)
    counts, index, types = build_composition_matrix(cells, cfg)
    assert counts.shape == (len(cells), len(TYPES))
    assert len(index) == len(cells)
    assert types == TYPES


def test_index_cell_included_in_own_neighbourhood():
    """Every row must have at least one count: the index cell itself."""
    cells = _cells()
    cfg = NeighbourhoodConfig(cell_type_order=TYPES)
    counts, _, _ = build_composition_matrix(cells, cfg)
    assert (np.asarray(counts.sum(axis=1)).ravel() >= 1).all()


def test_small_images_skipped():
    cells = _cells(n_img=2, n_cell=50)
    cfg = NeighbourhoodConfig(cell_type_order=TYPES, min_cells_per_image=200)
    with pytest.raises(ValueError, match="no images passed"):
        build_composition_matrix(cells, cfg)
