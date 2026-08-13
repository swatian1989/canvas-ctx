#!/usr/bin/env python
"""Stage 5: extract the 262 spatial features.

Two modes.

  --simulate   generate random habitat maps and run the full extraction. Use
               this BEFORE you have a trained model, to prove the statistical
               machinery works while nothing is at stake.

  --habitats   real patch-level habitat predictions from stage 4.

    python scripts/run_stage5_features.py --simulate --n-samples 300
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage5_features.spatial_features import (  # noqa: E402
    FeatureConfig, extract_all_features,
)


def simulate(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Random habitat maps with plausible per-slide patch counts and clustering."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_samples):
        n = int(rng.integers(400, 4000))
        # spatially clustered habitats, not uniform noise, so dispersion and
        # interaction features have something to measure
        centres = rng.uniform(0, 8000, size=(10, 2))
        assign = rng.integers(0, 10, n)
        pts = centres[assign] + rng.normal(0, 900, size=(n, 2))
        frames.append(pd.DataFrame({
            "sample_id": f"SIM-{i:04d}",
            "x_um": pts[:, 0], "y_um": pts[:, 1], "habitat": assign,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--habitats", default=None,
                    help="parquet with sample_id, x_um, y_um, habitat[, compartment]")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--by-compartment", action="store_true",
                    help="extract separately for tumour_bulk and leading_edge")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage5")

    raw = yaml.safe_load(Path(args.config).read_text())
    f = raw["features"]
    cfg = FeatureConfig(
        n_habitats=raw["project"]["n_habitats"],
        interaction_permutations=f["interaction_permutations"],
        transition_knn_k=f["transition_knn_k"],
        ripley_radii_um=tuple(f["ripley_radii_um"]),
        seed=raw["project"]["seed"],
    )

    if args.simulate:
        patches = simulate(args.n_samples, cfg.seed)
        out = Path(args.out or "data/interim/sim_features.parquet")
    elif args.habitats:
        patches = pd.read_parquet(args.habitats)
        out = Path(args.out or "data/processed/spatial_features.parquet")
    else:
        ap.error("pass --simulate or --habitats")

    groups = ["sample_id"] + (["compartment"] if args.by_compartment else [])
    rows = []
    for key, grp in tqdm(patches.groupby(groups), desc="samples"):
        sid = key if isinstance(key, str) else "|".join(map(str, key))
        try:
            rows.append(extract_all_features(grp, cfg, sample_id=sid))
        except Exception as exc:
            log.warning("skipped %s: %s", sid, exc)

    df = pd.DataFrame(rows).set_index("sample_id")
    assert df.shape[1] == f["total_expected"], (
        f"expected {f['total_expected']} features, got {df.shape[1]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    log.info("wrote %s  shape=%s", out, df.shape)
    log.info("all-NaN columns: %s", df.columns[df.isna().all()].tolist() or "none")


if __name__ == "__main__":
    main()
