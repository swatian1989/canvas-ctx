#!/usr/bin/env python
"""Extract real H&E patches from the Orion slide and build the F7 montage.

    python scripts/extract_orion_patches.py

Patches are read from the registered H&E at level 0 through a zarr view, so
the 13.5 GB full-resolution image is never loaded into memory: only the
345x345 px windows actually requested are decoded.

Patches are grouped by the DOMINANT GATED LINEAGE of the cells inside them,
not by transferred CN label. That is deliberate. Lineage identity (tumour
epithelium versus stroma versus immune infiltrate) is visually checkable in
H&E by eye, so this montage doubles as a visual audit of the marker gating,
which the label-transfer step depends on. Grouping by transferred CN would
show a label the reader cannot independently verify from the image.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GATE_MARKERS = ["Pan-CK", "E-cadherin", "CD3e", "CD20", "CD68", "CD163",
                "CD31", "SMA", "CD45"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default="data/raw/orion_crc/CRC01/CRC01-HE-registered.ome.tif")
    ap.add_argument("--cells", default="data/raw/orion_crc/CRC01/P37_S29-CRC01.csv")
    ap.add_argument("--patch-px", type=int, default=345, help="native px (224 @ 20x)")
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--min-cells", type=int, default=25)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--outdir", default="data/interim/orion_patches")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("orion_patches")

    import tifffile
    import zarr

    from canvas_brca.reporting.style import (
        apply_style, save_figure, source_caption,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_orion_label_transfer import gate_lineages

    log.info("gating cells")
    orion = pd.read_csv(args.cells, usecols=["X_centroid", "Y_centroid"] + GATE_MARKERS)
    lineage, _ = gate_lineages(orion, log)
    orion["lineage"] = lineage

    p = args.patch_px
    orion["px"] = (orion["X_centroid"] // p).astype(int)
    orion["py"] = (orion["Y_centroid"] // p).astype(int)

    grp = orion.groupby(["px", "py"])
    summary = grp["lineage"].agg(["count", lambda s: s.value_counts().idxmax(),
                                  lambda s: s.value_counts(normalize=True).max()])
    summary.columns = ["n_cells", "dominant", "purity"]
    summary = summary[summary["n_cells"] >= args.min_cells]
    log.info("%d patches with >=%d cells", len(summary), args.min_cells)
    log.info("dominant-lineage distribution:\n%s",
             summary["dominant"].value_counts().to_string())

    store = tifffile.imread(args.he, aszarr=True, level=0)
    z = zarr.open(store, mode="r")
    H, W = z.shape[:2]
    log.info("H&E level0 %dx%d, reading %dpx windows", W, H, p)

    groups = ["tumour", "stroma_sma", "t_cell", "macrophage", "vascular"]
    groups = [g for g in groups if (summary["dominant"] == g).any()]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(len(groups), args.per_group,
                             figsize=(2.5 * args.per_group, 2.6 * len(groups)))
    axes = np.atleast_2d(axes)

    rng = np.random.default_rng(0)
    for r, g in enumerate(groups):
        cand = summary[summary["dominant"] == g].sort_values("purity", ascending=False)
        cand = cand.head(400)
        pick = cand.iloc[rng.choice(len(cand), size=min(args.per_group, len(cand)),
                                    replace=False)]
        for c, (idx, row) in enumerate(pick.iterrows()):
            px, py = idx
            x0, y0 = px * p, py * p
            if y0 + p > H or x0 + p > W:
                axes[r, c].axis("off"); continue
            tile = np.asarray(z[y0:y0 + p, x0:x0 + p])
            axes[r, c].imshow(tile)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
            axes[r, c].set_title(f"n={int(row.n_cells)}, {row.purity:.0%}", fontsize=7.5)
            np.save(out / f"{g}_{px}_{py}.npy", tile)
        axes[r, 0].set_ylabel(g.replace("_", " "), fontsize=10, fontweight="bold")

    fig.suptitle("Real H&E patches (224 px at 20x equivalent), grouped by dominant "
                 "gated cell lineage", fontsize=11, y=1.0)
    source_caption(fig, "REAL DATA (Orion CRC01 registered H&E, doi:10.1038/s43018-023-00576-1; "
                        f"{len(orion):,} gated cells; {p} native px = 224 px at 0.5 um/px). "
                        "Titles give cell count and dominant-lineage purity per patch.",
                   y=-0.02)
    paths = save_figure(fig, "F7_patch_labels", args.figdir)
    log.info("wrote %s", paths["png"])


if __name__ == "__main__":
    main()
