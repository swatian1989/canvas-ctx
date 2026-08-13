#!/usr/bin/env python
"""Transfer the CRC CN taxonomy onto Orion H&E patches.

    python scripts/run_orion_label_transfer.py

Pipeline, using the EXISTING tested functions wherever they exist
(`build_composition_matrix`, `label_patches` are not reimplemented here):

  1. Gate Orion cells to lineages by per-marker Otsu thresholds on log
     intensity. Thresholds are derived from the data, not hand-picked.
  2. Collapse the Schurch 28-type taxonomy onto the SAME lineage vocabulary,
     so the two platforms become comparable.
  3. Compute each Schurch CN's mean neighbourhood composition in that shared
     vocabulary (the CN "centroid").
  4. Build 40 um neighbourhood compositions for Orion cells and assign each
     to its nearest Schurch CN centroid by cosine distance. This is the
     cross-platform label transfer.
  5. Apply the CANVAS patch purity rules via the tested `label_patches`.

THE CENTRAL LIMITATION, stated rather than buried: Schurch resolves 28 cell
types from 56 markers, Orion 16 biological markers. The taxonomies cannot be
matched cell-for-cell, so both are collapsed to the 8 lineages Orion can
actually express. CNs distinguished only by fine phenotype in the 56-plex
data (for example the several T-cell-enriched neighbourhoods) are therefore
not separable here, and the transfer is coarser than a same-platform one
would be.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage1_cn.neighborhoods import (  # noqa: E402
    NeighbourhoodConfig, build_composition_matrix,
)
from canvas_brca.stage2_pair.label_transfer import label_patches  # noqa: E402
from canvas_brca.utils.config import load_config  # noqa: E402

# Shared lineage vocabulary: the coarsest set BOTH platforms can express.
LINEAGES = ["tumour", "t_cell", "b_cell", "macrophage", "vascular",
            "stroma_sma", "other_immune", "other"]

# Schurch's 28 phenotypes collapsed onto that vocabulary.
SCHURCH_TO_LINEAGE = {
    "tumor cells": "tumour",
    "tumor cells / immune cells": "tumour",
    "CD4+ T cells": "t_cell", "CD4+ T cells CD45RO+": "t_cell",
    "CD4+ T cells GATA3+": "t_cell", "CD8+ T cells": "t_cell",
    "CD3+ T cells": "t_cell", "Tregs": "t_cell",
    "B cells": "b_cell",
    "CD68+ macrophages": "macrophage", "CD163+ macrophages": "macrophage",
    "CD68+CD163+ macrophages": "macrophage",
    "CD11b+CD68+ macrophages": "macrophage",
    "CD68+ macrophages GzmB+": "macrophage",
    "vasculature": "vascular", "lymphatics": "vascular",
    "immune cells / vasculature": "vascular",
    "smooth muscle": "stroma_sma", "stroma": "stroma_sma",
    "granulocytes": "other_immune", "plasma cells": "other_immune",
    "immune cells": "other_immune", "NK cells": "other_immune",
    "CD11c+ DCs": "other_immune", "CD11b+ monocytes": "other_immune",
    "adipocytes": "other", "nerves": "other", "undefined": "other",
}

GATE_MARKERS = ["Pan-CK", "E-cadherin", "CD3e", "CD20", "CD68", "CD163",
                "CD31", "SMA", "CD45"]


def otsu_threshold(x: np.ndarray) -> float:
    """Otsu on log1p intensity. Data-driven, no hand-picked cutoff."""
    import cv2
    v = np.log1p(np.clip(x, 0, None))
    v8 = np.clip((v - v.min()) / (v.max() - v.min() + 1e-9) * 255, 0, 255).astype(np.uint8)
    t8, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.expm1(t8 / 255.0 * (v.max() - v.min()) + v.min()))


def gate_lineages(df: pd.DataFrame, log) -> tuple[pd.Series, dict]:
    """Hierarchical gate to the shared lineage vocabulary."""
    pos, thr = {}, {}
    for m in GATE_MARKERS:
        t = otsu_threshold(df[m].to_numpy())
        thr[m] = round(t, 1)
        pos[m] = df[m].to_numpy() >= t
        log.info("  gate %-12s threshold %8.1f  ->  %5.2f%% positive",
                 m, t, 100 * pos[m].mean())

    epithelial = pos["Pan-CK"] | pos["E-cadherin"]
    lineage = np.full(len(df), "other", dtype=object)
    # order matters: later assignments do not overwrite earlier ones
    lineage[pos["CD45"]] = "other_immune"
    lineage[pos["SMA"] & ~pos["CD45"]] = "stroma_sma"
    lineage[pos["CD31"] & ~pos["CD45"]] = "vascular"
    lineage[(pos["CD68"] | pos["CD163"]) & pos["CD45"]] = "macrophage"
    lineage[pos["CD20"] & pos["CD45"]] = "b_cell"
    lineage[pos["CD3e"] & pos["CD45"]] = "t_cell"
    lineage[epithelial] = "tumour"          # epithelial identity dominates
    return pd.Series(lineage, index=df.index, name="lineage"), thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/crc_train_brca_apply.yaml")
    ap.add_argument("--orion-cells", default="data/raw/orion_crc/CRC01/P37_S29-CRC01.csv")
    ap.add_argument("--schurch-cells", default="data/interim/schurch_crc_cells.parquet")
    ap.add_argument("--schurch-cns", default="data/processed/cn_assignments.parquet")
    ap.add_argument("--sample-id", default="CRC01")
    ap.add_argument("--mpp", type=float, default=0.325)
    ap.add_argument("--outdir", default="data/interim/stage2_labels")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("orion_transfer")

    cfg_raw = load_config(args.config)
    radius_um = cfg_raw["cn_discovery"]["radius_um"]
    patch_px_20x = cfg_raw["patching"]["patch_px"]
    target_mpp = cfg_raw["patching"]["laptop_mpp"]
    n_habitats = cfg_raw["project"]["n_habitats"]
    # a 224 px patch at the 20x target resolution, expressed in native pixels
    patch_px_native = int(round(patch_px_20x * target_mpp / args.mpp))
    log.info("patch: %d px @ %.2f um/px (20x) = %d native px @ %.3f um/px",
             patch_px_20x, target_mpp, patch_px_native, args.mpp)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{args.sample_id}_patch_labels.parquet"
    if final.exists() and not args.force:
        log.info("%s exists, use --force to rerun", final)
        return

    # ---------------------------------------------------------------- Schurch
    log.info("building Schurch CN centroids in the shared lineage vocabulary")
    sch = pd.read_parquet(args.schurch_cells)
    cns = pd.read_parquet(args.schurch_cns)[["image_id", "cell_id", "cn_label"]]
    sch = sch.merge(cns, on=["image_id", "cell_id"], how="inner")
    sch["lineage"] = sch["cell_type"].map(SCHURCH_TO_LINEAGE).fillna("other")
    unmapped = set(sch.loc[sch["lineage"] == "other", "cell_type"]) - set(
        k for k, v in SCHURCH_TO_LINEAGE.items() if v == "other")
    if unmapped:
        log.warning("cell types falling through to 'other': %s", sorted(unmapped))

    # max_span_um raised from the 20 mm default: Orion sections are whole-slide
    # resections spanning ~25 mm, so the TMA-scale guard would reject them. The
    # units were confirmed independently first (X_centroid max 78,371 against an
    # H&E width of 78,417 px at 0.325 um/px, and a median cell area of 406 px^2
    # giving a 7.4 um diameter), so this raises the bound on verified data
    # rather than bypassing the check.
    ncfg = NeighbourhoodConfig(radius_um=radius_um, min_cells_per_image=200,
                               cell_type_order=LINEAGES, max_span_um=40_000.0)
    sch_tidy = sch.rename(columns={"lineage": "cell_type_orig"}).assign(
        cell_type=sch["lineage"])
    counts, index, types = build_composition_matrix(sch_tidy, ncfg)
    comp = np.asarray(counts.todense() if hasattr(counts, "todense") else counts, dtype=float)
    comp = comp / np.maximum(comp.sum(axis=1, keepdims=True), 1.0)
    index = index.reset_index(drop=True)
    index["cn_label"] = sch_tidy["cn_label"].to_numpy()[: len(index)] \
        if len(index) == len(sch_tidy) else None
    if index["cn_label"].isna().any():
        merged = index.merge(sch_tidy[["image_id", "cell_id", "cn_label"]],
                             on=["image_id", "cell_id"], how="left", suffixes=("", "_y"))
        index["cn_label"] = merged["cn_label_y"].to_numpy()

    centroids = {}
    for cn, idx in index.groupby("cn_label").groups.items():
        centroids[cn] = comp[np.asarray(list(idx))].mean(axis=0)
    cn_names = sorted(centroids)
    C = np.vstack([centroids[c] for c in cn_names])
    log.info("built %d CN centroids over %d lineages (%s)", len(cn_names), C.shape[1],
             ", ".join(types))

    # ------------------------------------------------------------------ Orion
    log.info("gating Orion cells")
    usecols = ["CellID", "X_centroid", "Y_centroid"] + GATE_MARKERS
    orion = pd.read_csv(args.orion_cells, usecols=usecols)
    lineage, thresholds = gate_lineages(orion, log)
    orion["cell_type"] = lineage
    log.info("Orion lineage composition:\n%s",
             orion["cell_type"].value_counts(normalize=True).mul(100).round(2).to_string())

    orion_tidy = pd.DataFrame({
        "image_id": args.sample_id,
        "cell_id": np.arange(len(orion)),
        "x_um": orion["X_centroid"] * args.mpp,
        "y_um": orion["Y_centroid"] * args.mpp,
        "cell_type": orion["cell_type"],
    })
    log.info("Orion field: %.0f x %.0f um, %d cells",
             orion_tidy["x_um"].max(), orion_tidy["y_um"].max(), len(orion_tidy))

    ocounts, oindex, otypes = build_composition_matrix(orion_tidy, ncfg)
    ocomp = np.asarray(ocounts.todense() if hasattr(ocounts, "todense") else ocounts,
                       dtype=float)
    nbrs = ocomp.sum(axis=1)
    log.info("Orion neighbourhood size at %.0f um: median %.0f cells",
             radius_um, np.median(nbrs))
    ocomp = ocomp / np.maximum(nbrs[:, None], 1.0)

    # nearest Schurch CN centroid by cosine distance
    def _unit(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    sim = _unit(ocomp) @ _unit(C).T
    assigned = np.asarray(cn_names)[sim.argmax(axis=1)]
    conf = sim.max(axis=1)
    oindex = oindex.reset_index(drop=True)
    oindex["cn_label"] = assigned
    oindex["confidence"] = conf
    log.info("transferred CN distribution:\n%s",
             pd.Series(assigned).value_counts(normalize=True).mul(100).round(2).to_string())
    log.info("cosine similarity to assigned centroid: median %.3f", float(np.median(conf)))

    # ------------------------------------------------------- patch labelling
    cn_to_int = {c: i for i, c in enumerate(cn_names)}
    nuclei = pd.DataFrame({
        "x_px": oindex["x_um"] / args.mpp,
        "y_px": oindex["y_um"] / args.mpp,
        "cn": [cn_to_int[c] for c in oindex["cn_label"]],
    })
    patches = label_patches(nuclei, patch_size_px=patch_px_native, n_habitats=n_habitats)
    patches["sample_id"] = args.sample_id
    patches["patch_px_native"] = patch_px_native
    patches["x_px"] = patches["patch_x"] * patch_px_native
    patches["y_px"] = patches["patch_y"] * patch_px_native
    patches.to_parquet(final, index=False)

    dist = patches["label"].value_counts().sort_index()
    named = {int(k): (cn_names[int(k)] if int(k) < len(cn_names) else "background")
             for k in dist.index}
    log.info("patch label distribution:\n%s",
             "\n".join(f"  {named[int(k)]:>8}  {v:6d}" for k, v in dist.items()))
    log.info("wrote %s (%d patches)", final, len(patches))

    meta = {"sample_id": args.sample_id, "n_cells": int(len(orion)),
            "gate_thresholds": thresholds,
            "lineage_pct": orion["cell_type"].value_counts(normalize=True)
                                 .mul(100).round(2).to_dict(),
            "median_neighbours_40um": float(np.median(nbrs)),
            "median_cosine_similarity": float(np.median(conf)),
            "patch_px_native": patch_px_native,
            "cn_names": cn_names,
            "patch_label_counts": {named[int(k)]: int(v) for k, v in dist.items()}}
    (Path("results") / "orion_label_transfer.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote results/orion_label_transfer.json")


if __name__ == "__main__":
    main()
