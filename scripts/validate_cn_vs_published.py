#!/usr/bin/env python
"""Compare our rediscovered CNs against the CN labels published with the data.

    python scripts/validate_cn_vs_published.py

This is the strongest validation available at stage 1 and it needs no extra
data: the Schurch table ships the authors' own neighbourhood assignments
(`published_cn`), so our independently rediscovered CNs can be scored
against them directly.

Agreement will NOT be perfect, and low agreement is not automatically a
failure. The two procedures genuinely differ:
  - Schurch et al. built each window from the k = 10 nearest neighbours.
  - CANVAS specifies a fixed 40 um radius, which on this tissue captures a
    median of ~33 neighbours, and adds a spatial-LDA topic decomposition
    before k-means.
So this measures "does a CANVAS-style procedure recover comparable tissue
structure on the same cells", not "did we reproduce their code".

Adjusted Rand index and normalised mutual information are reported, plus a
contingency table naming which published CN each of our CNs maps onto.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", default="data/processed/cn_assignments.parquet")
    ap.add_argument("--cells", default="data/interim/schurch_crc_cells.parquet")
    ap.add_argument("--outdir", default="results/tables")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("validate_cn")

    ours = pd.read_parquet(args.assignments)
    cells = pd.read_parquet(args.cells)

    # assign_cns returns the index-cell frame with cn/cn_label attached; join
    # back to the source table on the columns both carry
    key = [c for c in ("image_id", "cell_id") if c in ours.columns and c in cells.columns]
    if not key:
        raise SystemExit(f"cannot join: ours has {list(ours.columns)[:8]}")
    merged = ours.merge(cells[key + ["published_cn", "cell_type"]], on=key, how="inner")
    log.info("joined %d cells on %s", len(merged), key)

    merged = merged[merged["published_cn"].notna()]
    ours_lab = merged["cn"].to_numpy()
    pub_lab = pd.factorize(merged["published_cn"])[0]

    ari = adjusted_rand_score(pub_lab, ours_lab)
    nmi = normalized_mutual_info_score(pub_lab, ours_lab)
    log.info("adjusted Rand index vs published CNs : %.3f", ari)
    log.info("normalised mutual information        : %.3f", nmi)
    log.info("(ARI 0 = chance, 1 = identical partitions; the two procedures "
             "differ by construction, see this script's docstring)")

    ct = pd.crosstab(merged["cn_label"], merged["published_cn"])
    ct_frac = ct.div(ct.sum(axis=1), axis=0)

    log.info("\nbest-matching published CN for each of ours:")
    rows = []
    for cn in ct_frac.index:
        best = ct_frac.loc[cn].idxmax()
        frac = ct_frac.loc[cn].max()
        n = int(ct.loc[cn].sum())
        rows.append({"our_cn": cn, "n_cells": n, "best_published_match": best,
                     "overlap_fraction": round(float(frac), 3)})
        log.info("  %-6s n=%6d  ->  %-28s %.1f%%", cn, n, best, 100 * frac)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "T11_cn_vs_published.csv", index=False)
    ct.to_csv(out / "T11_cn_contingency.csv")
    summary = pd.DataFrame([{"metric": "adjusted_rand_index", "value": ari},
                            {"metric": "normalized_mutual_info", "value": nmi},
                            {"metric": "n_cells", "value": len(merged)},
                            {"metric": "n_our_cns", "value": merged["cn"].nunique()},
                            {"metric": "n_published_cns", "value": merged["published_cn"].nunique()}])
    summary.to_csv(out / "T11_cn_validation_summary.csv", index=False)
    log.info("wrote %s", out / "T11_cn_vs_published.csv")


if __name__ == "__main__":
    main()
