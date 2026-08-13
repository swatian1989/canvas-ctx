#!/usr/bin/env python
"""Stage 6: survival, ecotypes, and the spatial signature.

    python scripts/run_stage6_clinical.py \
        --features data/processed/spatial_features.parquet \
        --clinical data/raw/tcga_brca_cdr.csv \
        --endpoint OS --stratify PAM50

Clinical table needs sample_id plus {endpoint}_time and {endpoint}_event
columns, per TCGA-CDR. Times in days.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage6_clinical.signature import (  # noqa: E402
    ClinicalConfig, consensus_cluster, reduce_collinearity, univariate_cox,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--endpoint", default="OS", choices=["OS", "DSS", "PFI", "DFI"])
    ap.add_argument("--stratify", default=None,
                    help="clinical column to analyse within, e.g. PAM50")
    ap.add_argument("--ecotype-k", type=int, default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage6")

    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ClinicalConfig(seed=raw["project"]["seed"])
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    clin = pd.read_csv(args.clinical, index_col=0)
    log.info("features %s, clinical %s, overlap %d",
             feats.shape, clin.shape, len(feats.index.intersection(clin.index)))

    # --- univariate Cox, optionally within strata
    strata = [(None, feats.index)]
    if args.stratify:
        if args.stratify not in clin.columns:
            raise SystemExit(f"'{args.stratify}' not in the clinical table")
        strata = [(g, sub.index) for g, sub in clin.groupby(args.stratify)]
        log.info("stratifying by %s: %s", args.stratify, [s[0] for s in strata])

    for name, idx in strata:
        sub = feats.loc[feats.index.intersection(idx)]
        if len(sub) < 40:
            log.warning("stratum %s has only %d samples, skipping", name, len(sub))
            continue
        res = univariate_cox(sub, clin, args.endpoint, cfg=cfg)
        tag = f"_{name}" if name else ""
        res.to_csv(out / f"cox_{args.endpoint}{tag}.csv", index=False)
        sig = res[res["q"] < cfg.alpha] if not res.empty else res
        log.info("stratum %s: %d/%d features significant at FDR %.2f",
                 name or "all", len(sig), len(res), cfg.alpha)

    # --- ecotypes from composition features only, as in the paper
    comp = feats[[c for c in feats.columns if c.startswith("comp_")]]
    labels, consensus = consensus_cluster(comp, cfg, k=args.ecotype_k or 4)
    labels.to_csv(out / "ecotypes.csv")
    log.info("ecotype sizes: %s", labels.value_counts().to_dict())
    if args.ecotype_k is None:
        log.warning(
            "k defaulted to 4 because CANVAS found 4 in lung. Sweep k=2..8 and "
            "pick from the consensus CDF before reporting anything.")

    # --- collinearity reduction, the first step of the signature model
    keep = reduce_collinearity(feats, cfg)
    pd.Series(keep, name="feature").to_csv(out / "noncollinear_features.csv", index=False)
    log.info("kept %d of %d features after collinearity reduction",
             len(keep), feats.shape[1])
    log.info("next: lasso_cox_selection -> rsf_importance -> fit_signature")


if __name__ == "__main__":
    main()
