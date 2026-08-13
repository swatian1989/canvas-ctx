#!/usr/bin/env python
"""Derive one shared habitat taxonomy across the whole Orion cohort.

    python scripts/run_orion_cohort.py

DESIGN. CANVAS derives cellular neighbourhoods ONCE over the discovery
cohort and then applies that single taxonomy to every sample. Deriving CNs
per specimen would give each slide its own private label set, and "CN03"
would mean something different on every slide, which makes the downstream
classifier meaningless. So the model is fitted on cells pooled across all
specimens and then applied to each.

Two concessions to cohort scale (~19M cells), both stated rather than
hidden:
  - The k-means model is FITTED on a pooled random subsample and then used
    to PREDICT every cell. Fitting on 19M points with multiple restarts is
    not tractable on CPU, and a subsample of this size estimates the
    centroids well.
  - The k sweep is skipped. k was already selected on this tissue type in
    stage 1 (silhouette 9, Davies-Bouldin 11, protocol 10); rerunning a
    16-value sweep over the cohort would cost hours and change nothing.

Patch labels are capped per specimen so downstream encoding stays feasible
on CPU, matching the pilot profile's max_patches_per_slide.
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
    NeighbourhoodConfig, build_composition_matrix, cn_lineage_enrichment,
)
from canvas_brca.stage2_pair.label_transfer import label_patches  # noqa: E402
from canvas_brca.utils.config import load_config  # noqa: E402
from run_orion_label_transfer import GATE_MARKERS, LINEAGES, gate_lineages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/crc_train_brca_apply.yaml")
    ap.add_argument("--orion-dir", default="data/raw/orion_crc")
    ap.add_argument("--mpp", type=float, default=0.325)
    ap.add_argument("--fit-sample", type=int, default=400_000)
    ap.add_argument("--max-patches", type=int, default=2000,
                    help="labelled patches kept per specimen (CPU encoding budget)")
    ap.add_argument("--outdir", default="data/interim/orion_cohort")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("orion_cohort")

    raw = load_config(args.config)
    radius_um = raw["cn_discovery"]["radius_um"]
    final_k = raw["cn_discovery"]["kmeans"]["final_k"]
    seed = raw["project"]["seed"]
    patch_px = int(round(raw["patching"]["patch_px"]
                         * raw["patching"]["laptop_mpp"] / args.mpp))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "cohort_patch_labels.parquet"
    if final.exists() and not args.force:
        log.info("%s exists, use --force to rerun", final)
        return

    specimens = sorted(d.name for d in Path(args.orion_dir).iterdir()
                       if d.is_dir() and (d / f"{d.name}-cells.csv").exists()
                       or (d / "P37_S29-CRC01.csv").exists())
    log.info("specimens found: %s", specimens)

    cfg = NeighbourhoodConfig(
        radius_um=radius_um, min_cells_per_image=200, cell_type_order=LINEAGES,
        n_topics=len(LINEAGES), final_k=final_k, n_init=10, seed=seed,
        max_span_um=60_000.0,   # whole-slide scale; units verified upstream
    )

    # ------------------------------------------------- pass 1: gate + compose
    comps, indices = [], []
    for name in specimens:
        d = Path(args.orion_dir) / name
        csv = d / f"{name}-cells.csv"
        if not csv.exists():
            csv = d / "P37_S29-CRC01.csv"
        log.info("--- %s ---", name)
        cells = pd.read_csv(csv, usecols=["X_centroid", "Y_centroid"] + GATE_MARKERS)
        lineage, _ = gate_lineages(cells, logging.getLogger("gate"))
        tidy = pd.DataFrame({
            "image_id": name,
            "cell_id": np.arange(len(cells)),
            "x_um": cells["X_centroid"] * args.mpp,
            "y_um": cells["Y_centroid"] * args.mpp,
            "cell_type": lineage.to_numpy(),
        })
        counts, index, types = build_composition_matrix(tidy, cfg)
        arr = np.asarray(counts.todense() if hasattr(counts, "todense") else counts,
                         dtype=np.float32)
        comps.append(arr)
        indices.append(index.assign(image_id=name))
        log.info("  %s: %d cells, median %.0f neighbours at %.0f um",
                 name, len(tidy), np.median(arr.sum(axis=1)), radius_um)

    comp = np.vstack(comps)
    index = pd.concat(indices, ignore_index=True)
    del comps, indices
    log.info("pooled: %s composition rows over %d specimens", comp.shape, len(specimens))

    # ------------------------------------------- pass 2: one shared taxonomy
    # BOTH models are fitted on a pooled subsample and then applied to every
    # cell. fit_spatial_lda() fit-transforms the whole matrix in one go, which
    # is fine for a 250k-cell TMA cohort but not for ~19M whole-slide cells:
    # an initial attempt stalled here for well over an hour. The estimator
    # below is the same sklearn LatentDirichletAllocation that
    # fit_spatial_lda() falls back to whenever the optional spatial_lda
    # package is absent, which is the path taken throughout this project.
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(comp), size=min(args.fit_sample, len(comp)),
                         replace=False)
    from sklearn.decomposition import LatentDirichletAllocation

    log.info("fitting LDA (%d topics) on %d pooled neighbourhoods", cfg.n_topics,
             len(fit_idx))
    lda = LatentDirichletAllocation(n_components=cfg.n_topics, random_state=seed,
                                    learning_method="online", batch_size=8192)
    lda.fit(comp[fit_idx])
    log.info("transforming all %d neighbourhoods in chunks", len(comp))
    chunks = []
    for s in range(0, len(comp), 1_000_000):
        chunks.append(lda.transform(comp[s:s + 1_000_000]).astype(np.float32))
        log.info("  transformed %d / %d", min(s + 1_000_000, len(comp)), len(comp))
    topics = np.vstack(chunks)
    del chunks

    from sklearn.cluster import KMeans
    log.info("fitting k-means (k=%d) on %d pooled cells", final_k, len(fit_idx))
    km = KMeans(n_clusters=final_k, n_init=cfg.n_init, random_state=seed)
    km.fit(topics[fit_idx])
    labels = km.predict(topics)
    index["cn"] = labels
    index["cn_label"] = [f"CN{i + 1:02d}" for i in labels]
    log.info("fitted k=%d on %d pooled cells, applied to all %d",
             final_k, len(fit_idx), len(index))
    log.info("cohort CN frequency (%%):\n%s",
             (index["cn_label"].value_counts(normalize=True) * 100).round(2).to_string())

    from scipy import sparse
    enrich = cn_lineage_enrichment(index, sparse.csr_matrix(comp), LINEAGES)
    enrich.to_csv(out / "cohort_cn_lineage_enrichment.csv")
    log.info("\n%s", enrich.round(2).to_string())
    index.to_parquet(out / "cohort_cn_assignments.parquet", index=False)

    # ---------------------------------------------------- pass 3: patch labels
    all_patches = []
    for name in specimens:
        g = index[index["image_id"] == name]
        nuclei = pd.DataFrame({"x_px": g["x_um"] / args.mpp,
                               "y_px": g["y_um"] / args.mpp,
                               "cn": g["cn"].to_numpy()})
        p = label_patches(nuclei, patch_size_px=patch_px, n_habitats=final_k)
        if p.empty:
            log.warning("  %s produced no patches", name)
            continue
        if len(p) > args.max_patches:
            # Keep a CONTIGUOUS lattice region, not a random subsample.
            # Method 2's grid and graph encoders read each patch's spatial
            # neighbours off the lattice; randomly thinning patches would
            # punch holes in it and degrade exactly the signal the benchmark
            # is meant to measure. Taking the patches nearest the lattice
            # centroid keeps a dense, roughly circular block.
            cx, cy = p["patch_x"].median(), p["patch_y"].median()
            d2 = (p["patch_x"] - cx) ** 2 + (p["patch_y"] - cy) ** 2
            p = p.loc[d2.nsmallest(args.max_patches).index].reset_index(drop=True)
            occ = len(p) / max(1, (p["patch_x"].max() - p["patch_x"].min() + 1)
                               * (p["patch_y"].max() - p["patch_y"].min() + 1))
            log.info("  %s: capped to a contiguous block, lattice occupancy %.0f%%",
                     name, 100 * occ)
        p["sample_id"] = name
        p["x_px"] = p["patch_x"] * patch_px
        p["y_px"] = p["patch_y"] * patch_px
        all_patches.append(p)
        log.info("  %s: %d patches kept", name, len(p))

    patches = pd.concat(all_patches, ignore_index=True)
    patches["patch_px_native"] = patch_px
    patches.to_parquet(final, index=False)
    log.info("wrote %s: %d patches, %d specimens", final, len(patches),
             patches["sample_id"].nunique())
    log.info("label distribution:\n%s",
             patches["label"].value_counts().sort_index().to_string())

    meta = {"specimens": specimens, "final_k": final_k, "patch_px_native": patch_px,
            "n_patches": int(len(patches)),
            "patches_per_specimen": patches["sample_id"].value_counts().to_dict(),
            "label_counts": {int(k): int(v) for k, v
                             in patches["label"].value_counts().items()}}
    (Path("results") / "orion_cohort.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
