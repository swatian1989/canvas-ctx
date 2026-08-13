#!/usr/bin/env python
"""Stage 1: discover breast cellular neighbourhoods from IMC single-cell data.

Runs on CPU. Start here.

    python scripts/run_stage1_cn.py --cells data/raw/danenberg_singlecell.csv

Expects a tidy table with image_id, cell_id, x_um, y_um, cell_type. If the
Danenberg download uses different column names, map them with --colmap.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage1_cn.neighborhoods import (  # noqa: E402
    NeighbourhoodConfig, assign_cns, build_composition_matrix,
    cn_lineage_enrichment, fit_spatial_lda, sweep_k,
)
from canvas_brca.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="single-cell table (csv/parquet)")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--colmap", default=None,
                    help='JSON, e.g. \'{"ImageNumber":"image_id","Location_X":"x_um"}\'')
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / "cn_assignments.parquet"
    if final.exists() and not args.force:
        log.info("%s exists, use --force to rerun", final)
        return

    # load_config resolves the `inherit:` chain (e.g. crc_train_brca_apply ->
    # pilot -> default). A plain yaml.safe_load sees only the keys defined
    # directly in the file it was pointed at, so any config that overrides a
    # subset would fail here on the keys it inherits rather than restates.
    raw = load_config(args.config)
    c = raw["cn_discovery"]
    cfg = NeighbourhoodConfig(
        radius_um=c["radius_um"],
        expected_neighbours=c["expected_neighbours"],
        min_cells_per_image=c["min_cells_per_image"],
        n_topics=c["spatial_lda"]["n_topics"],
        k_min=c["kmeans"]["k_min"], k_max=c["kmeans"]["k_max"],
        final_k=c["kmeans"]["final_k"], n_init=c["kmeans"]["n_init"],
        seed=raw["project"]["seed"],
    )

    path = Path(args.cells)
    cells = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if args.colmap:
        cells = cells.rename(columns=json.loads(args.colmap))

    counts, index, types = build_composition_matrix(cells, cfg)
    topics = fit_spatial_lda(counts, cfg)

    if not args.skip_sweep:
        sweep = sweep_k(topics, cfg)
        sweep.to_csv(outdir / "k_sweep.csv", index=False)
        log.info("k sweep written. INSPECT IT before trusting final_k=%d.", cfg.final_k)
        log.info("\n%s", sweep.to_string(index=False))

    assignments = assign_cns(topics, index, cfg)
    assignments.to_parquet(final, index=False)

    enrichment = cn_lineage_enrichment(assignments, counts, types)
    enrichment.to_csv(outdir / "cn_lineage_enrichment.csv")

    log.info("\n%s", enrichment.round(2).to_string())
    log.info(
        "\nNAME YOUR CNs FROM THE TABLE ABOVE. Do not reuse the NSCLC names "
        "(tumour core, macrophage niche, B cell niche, stromal-fibrotic, plasma "
        "cell, neutrophil-rich, tumour-immune interface, T cell compartment, "
        "pan-immune, vasculature). Breast will differ, and the myoepithelial "
        "and adipose compartments have no lung counterpart."
    )
    log.info("wrote %s (%d cells, %d CNs)", final, len(assignments), cfg.final_k)


if __name__ == "__main__":
    main()
