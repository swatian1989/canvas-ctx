#!/usr/bin/env python
"""Validate the stage6 statistical chain against a NULL (random) outcome.

Habitat features are real (from simulated spatial patterns via stage5), but
the survival outcome is independently random noise -- exponential event and
censoring times drawn with no dependence on any feature. Every stage6
statistic should therefore come out null: BH-FDR-corrected univariate Cox
should flag ~0 features, and the final signature's C-index should sit near
0.5. This is a machinery check, run BEFORE any real habitat predictions or
clinical data exist -- a "significant" result here means a bug in the
pipeline, not a finding.

    python scripts/validate_stage6_null.py --features data/interim/sim_features.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage6_clinical.signature import (  # noqa: E402
    ClinicalConfig, consensus_cluster, fit_signature, lasso_cox_selection,
    reduce_collinearity, rsf_importance, univariate_cox,
)


def simulate_null_clinical(index: pd.Index, endpoints: list[str], seed: int) -> pd.DataFrame:
    """Random exponential survival times/events, independent of any feature."""
    rng = np.random.default_rng(seed)
    cols = {}
    for ep in endpoints:
        event_time = rng.exponential(scale=1000.0, size=len(index))
        censor_time = rng.exponential(scale=1200.0, size=len(index))
        cols[f"{ep}_time"] = np.minimum(event_time, censor_time)
        cols[f"{ep}_event"] = (event_time <= censor_time).astype(int)
    return pd.DataFrame(cols, index=index)


def _top_frac(s: pd.Series, frac: float) -> list[str]:
    threshold = s.quantile(1.0 - frac)
    return s[s >= threshold].index.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/interim/sim_features.parquet")
    ap.add_argument("--endpoint", default="OS")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lasso-repeats", type=int, default=50)
    ap.add_argument("--rsf-bootstrap", type=int, default=200)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("validate_null")

    features = pd.read_parquet(args.features)
    log.info("features %s", features.shape)

    clinical = simulate_null_clinical(features.index, [args.endpoint], seed=args.seed)
    cfg = ClinicalConfig(lasso_repeats=args.lasso_repeats, rsf_bootstrap=args.rsf_bootstrap,
                          seed=args.seed)

    # 1. univariate Cox, BH-FDR corrected -- should be null
    cox = univariate_cox(features, clinical, args.endpoint, cfg=cfg)
    n_sig = int((cox["q"] < cfg.alpha).sum()) if not cox.empty else 0
    log.info("univariate Cox: %d/%d features at q<%.2f (expect ~0 on random data)",
              n_sig, len(cox), cfg.alpha)

    # 2. ecotypes from composition features -- runs regardless of outcome
    comp = features[[c for c in features.columns if c.startswith("comp_")]]
    labels, _ = consensus_cluster(comp, cfg, k=4)
    log.info("ecotypes: %s", labels.value_counts().to_dict())

    # 3. collinearity reduction
    keep = reduce_collinearity(features, cfg)
    reduced = features[keep]
    log.info("collinearity reduction: %d -> %d features", features.shape[1], len(keep))

    # 4. LASSO-Cox selection frequency
    lasso_freq = lasso_cox_selection(reduced, clinical, args.endpoint, cfg)
    lasso_top = _top_frac(lasso_freq, cfg.lasso_top_frac)
    log.info("LASSO top %.0f%%: %d features, max selection freq %.2f",
              cfg.lasso_top_frac * 100, len(lasso_top), lasso_freq.max())

    # 5. RSF permutation importance
    rsf_imp = rsf_importance(reduced, clinical, args.endpoint, cfg)
    rsf_top = _top_frac(rsf_imp, cfg.lasso_top_frac)
    log.info("RSF top %.0f%%: %d features, max importance drop %.4f",
              cfg.lasso_top_frac * 100, len(rsf_top), rsf_imp.max())

    # 6. intersection -> multivariable signature, exactly as PROTOCOL.md specifies
    selected = sorted(set(lasso_top) & set(rsf_top))
    log.info("LASSO ^ RSF intersection: %d features -> %s", len(selected), selected)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cox.to_csv(out / f"null_cox_{args.endpoint}.csv", index=False)

    if len(selected) < 2:
        log.warning(
            "fewer than 2 features survived selection (%d) -- multivariable Cox needs "
            "at least 2 to be meaningful. This IS a plausible null outcome (both "
            "selectors agreeing on almost nothing is what 'no real signal' looks "
            "like): skipping fit_signature rather than fabricating a fit on 0-1 "
            "features.", len(selected))
        return

    model, metrics = fit_signature(reduced, clinical, args.endpoint, selected, cfg)
    metrics.to_csv(out / f"null_signature_metrics_{args.endpoint}.csv", index=False)
    log.info("\n%s", metrics.to_string())
    log.info(
        "C-index should sit near 0.5 (chance) on random data. Anything durably above "
        "~0.55-0.6 on a few hundred samples suggests the selection+fit pipeline is "
        "picking up its own noise (a known risk of selecting and fitting on the same "
        "data with no held-out split), not a real signal -- worth a second look "
        "before trusting a real-data result from this same script."
    )


if __name__ == "__main__":
    main()
