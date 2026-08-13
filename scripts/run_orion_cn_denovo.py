#!/usr/bin/env python
"""Derive cellular neighbourhoods de novo on the Orion cohort.

    python scripts/run_orion_cn_denovo.py

WHY THIS IS THE RIGHT METHOD, not a fallback. CANVAS discovers CNs from the
spatial-omics data that is PAIRED WITH THE H&E, then transfers those labels
onto the same tissue. Orion is that paired modality here. Deriving CNs on
Orion is therefore closer to the published design than importing a taxonomy
from a different cohort, and it dissolves the failure documented in
run_orion_label_transfer.py: there is no 56-plex-to-16-plex vocabulary
collapse, because discovery and transfer share one panel.

The cost is that these CNs are defined over 8 gated lineages rather than
Schurch's 28 phenotypes, so they are necessarily coarser. Their relationship
to the Schurch CNs is measured at the end rather than assumed.

Every clustering step reuses the tested stage-1 functions unchanged.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from canvas_brca.stage1_cn.neighborhoods import (  # noqa: E402
    NeighbourhoodConfig, assign_cns, build_composition_matrix,
    cn_lineage_enrichment, fit_spatial_lda, sweep_k,
)
from canvas_brca.stage2_pair.label_transfer import label_patches  # noqa: E402
from canvas_brca.utils.config import load_config  # noqa: E402
from run_orion_label_transfer import (  # noqa: E402
    GATE_MARKERS, LINEAGES, SCHURCH_TO_LINEAGE, gate_lineages,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/crc_train_brca_apply.yaml")
    ap.add_argument("--cells", default="data/raw/orion_crc/CRC01/P37_S29-CRC01.csv")
    ap.add_argument("--schurch-cells", default="data/interim/schurch_crc_cells.parquet")
    ap.add_argument("--schurch-cns", default="data/processed/cn_assignments.parquet")
    ap.add_argument("--sample-id", default="CRC01")
    ap.add_argument("--mpp", type=float, default=0.325)
    ap.add_argument("--n-topics", type=int, default=8,
                    help="LDA topics; capped at the 8-word lineage vocabulary")
    ap.add_argument("--sweep-sample", type=int, default=200_000,
                    help="cells subsampled for the k sweep (assignment uses all)")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("orion_cn")

    raw = load_config(args.config)
    radius_um = raw["cn_discovery"]["radius_um"]
    k_min, k_max = raw["cn_discovery"]["kmeans"]["k_min"], raw["cn_discovery"]["kmeans"]["k_max"]
    final_k = raw["cn_discovery"]["kmeans"]["final_k"]
    seed = raw["project"]["seed"]
    patch_px_native = int(round(raw["patching"]["patch_px"]
                                * raw["patching"]["laptop_mpp"] / args.mpp))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"orion_{args.sample_id}_cn_assignments.parquet"
    if final.exists() and not args.force:
        log.info("%s exists, use --force to rerun", final)
        return

    # ------------------------------------------------------------------ gate
    log.info("gating %s", args.cells)
    orion = pd.read_csv(args.cells, usecols=["X_centroid", "Y_centroid"] + GATE_MARKERS)
    lineage, thresholds = gate_lineages(orion, log)
    tidy = pd.DataFrame({
        "image_id": args.sample_id,
        "cell_id": np.arange(len(orion)),
        "x_um": orion["X_centroid"] * args.mpp,
        "y_um": orion["Y_centroid"] * args.mpp,
        "cell_type": lineage.to_numpy(),
    })
    log.info("%d cells, %d lineages, field %.0f x %.0f um",
             len(tidy), tidy["cell_type"].nunique(),
             tidy["x_um"].max(), tidy["y_um"].max())

    # max_span_um raised for whole-slide scale; units verified independently
    # (see run_orion_label_transfer.py).
    cfg = NeighbourhoodConfig(
        radius_um=radius_um, min_cells_per_image=200, cell_type_order=LINEAGES,
        n_topics=min(args.n_topics, len(LINEAGES)), k_min=k_min, k_max=k_max,
        final_k=final_k, n_init=10, seed=seed, max_span_um=40_000.0,
    )
    if args.n_topics > len(LINEAGES):
        log.warning("n_topics capped at the %d-word vocabulary", len(LINEAGES))

    # ------------------------------------------------- neighbourhoods + LDA
    counts, index, types = build_composition_matrix(tidy, cfg)
    topics = fit_spatial_lda(counts, cfg)
    log.info("topic matrix %s", topics.shape)

    # ------------------------------------------------------------- k sweep
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(topics), size=min(args.sweep_sample, len(topics)), replace=False)
    log.info("k sweep on a %d-cell subsample (assignment then uses all %d)",
             len(sub), len(topics))
    sweep = sweep_k(topics[sub], cfg)
    sweep.to_csv(out / "orion_k_sweep.csv", index=False)
    log.info("\n%s", sweep.to_string(index=False))

    # ---------------------------------------------------------- assignment
    assignments = assign_cns(topics, index, cfg)
    assignments.to_parquet(final, index=False)
    log.info("assigned %d cells to %d CNs", len(assignments), cfg.final_k)

    enrich = cn_lineage_enrichment(assignments, counts, types)
    enrich.to_csv(out / "orion_cn_lineage_enrichment.csv")
    log.info("\n%s", enrich.round(2).to_string())

    # ------------------------------------- relate to the Schurch CN taxonomy
    log.info("comparing Orion CNs to the Schurch CNs by lineage composition")
    sch = pd.read_parquet(args.schurch_cells)
    scns = pd.read_parquet(args.schurch_cns)[["image_id", "cell_id", "cn_label"]]
    sch = sch.merge(scns, on=["image_id", "cell_id"], how="inner")
    sch["lin"] = sch["cell_type"].map(SCHURCH_TO_LINEAGE).fillna("other")
    sch_prof = pd.crosstab(sch["cn_label"], sch["lin"], normalize="index")
    sch_prof = sch_prof.reindex(columns=LINEAGES, fill_value=0.0)

    ori_prof = pd.crosstab(assignments["cn_label"], assignments["cell_type"],
                           normalize="index").reindex(columns=LINEAGES, fill_value=0.0)

    def _unit(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    sim = _unit(ori_prof.to_numpy()) @ _unit(sch_prof.to_numpy()).T
    simdf = pd.DataFrame(sim, index=ori_prof.index, columns=sch_prof.index)
    simdf.to_csv(out / "orion_vs_schurch_cn_similarity.csv")
    rows = []
    for cn in simdf.index:
        best = simdf.loc[cn].idxmax()
        rows.append({"orion_cn": cn, "best_schurch_cn": best,
                     "cosine": round(float(simdf.loc[cn, best]), 3),
                     "n_cells": int((assignments["cn_label"] == cn).sum())})
        log.info("  %s -> %s (cosine %.3f)", cn, best, simdf.loc[cn, best])
    pd.DataFrame(rows).to_csv(out / "orion_vs_schurch_cn_match.csv", index=False)

    # ------------------------------------------------------ patch labelling
    nuclei = pd.DataFrame({
        "x_px": assignments["x_um"] / args.mpp,
        "y_px": assignments["y_um"] / args.mpp,
        "cn": assignments["cn"].to_numpy(),
    })
    patches = label_patches(nuclei, patch_size_px=patch_px_native,
                            n_habitats=cfg.final_k)
    patches["sample_id"] = args.sample_id
    patches["x_px"] = patches["patch_x"] * patch_px_native
    patches["y_px"] = patches["patch_y"] * patch_px_native
    pdir = Path("data/interim/stage2_labels")
    pdir.mkdir(parents=True, exist_ok=True)
    patches.to_parquet(pdir / f"{args.sample_id}_denovo_patch_labels.parquet", index=False)

    dist = patches["label"].value_counts().sort_index()
    log.info("patch labels (%d total):\n%s", len(patches),
             "\n".join(f"  {'background' if int(k)==cfg.final_k else f'CN{int(k)+1:02d}'}"
                       f"  {v:6d}" for k, v in dist.items()))

    meta = {"sample_id": args.sample_id, "n_cells": int(len(tidy)),
            "gate_thresholds": thresholds,
            "n_topics": cfg.n_topics, "final_k": cfg.final_k,
            "patch_px_native": patch_px_native,
            "cn_frequency_pct": (assignments["cn_label"].value_counts(normalize=True)
                                 * 100).round(2).to_dict(),
            "patch_label_counts": {("background" if int(k) == cfg.final_k
                                    else f"CN{int(k)+1:02d}"): int(v)
                                   for k, v in dist.items()},
            "orion_vs_schurch": rows}
    (Path("results") / "orion_cn_denovo.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s and results/orion_cn_denovo.json", final)


if __name__ == "__main__":
    main()
