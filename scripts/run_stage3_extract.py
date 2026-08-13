#!/usr/bin/env python
"""Stage 3: extract and cache patch embeddings from WSIs.

    python scripts/run_stage3_extract.py \
        --config config/crc_train_brca_apply.yaml \
        --slides "data/raw/orion_crc/*.svs" \
        --outdir data/interim/patch_embeddings

Output: one parquet shard per slide under --outdir, columns slide_id, x_um,
y_um, patch_x, patch_y, emb_0..emb_{D-1}. Read the whole cohort at once with
`pd.read_parquet(outdir)`. This is the embeddings half of the schema
scripts/run_final_benchmark.py expects; a `label` column is joined on
separately during stage2 label transfer, not produced here.

Idempotent and resumable: slides whose shard already exists are skipped
unless --force.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage3_model.encoders import load_encoder, resolve_encoder  # noqa: E402
from canvas_brca.stage3_model.extract_embeddings import (  # noqa: E402
    ExtractConfig, extract_embeddings,
)
from canvas_brca.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True,
                    help="WSI paths or globs, e.g. 'data/raw/orion_crc/*.svs'")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="data/interim/patch_embeddings")
    ap.add_argument("--encoder", default=None, help="override config's model.encoder")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage3_extract")

    raw = load_config(args.config)
    profile = raw["project"]["profile"]
    patching = raw["patching"]
    model_cfg = raw["model"]

    target_mpp = patching["laptop_mpp"] if profile == "laptop" else patching["target_mpp"]
    cfg = ExtractConfig(
        patch_px=patching["patch_px"],
        stride_px=patching["stride_px"],
        target_mpp=target_mpp,
        min_tissue_frac=patching["min_tissue_frac"],
        max_patches_per_slide=patching.get("max_patches_per_slide"),
        batch_size=model_cfg.get("batch_size", 16),
        stain_norm=patching.get("stain_norm", "macenko"),
        seed=raw["project"]["seed"],
    )
    log.info("profile=%s target_mpp=%.2f patch_px=%d max_patches_per_slide=%s",
              profile, target_mpp, cfg.patch_px, cfg.max_patches_per_slide)

    requested = args.encoder or model_cfg["encoder"]
    spec = resolve_encoder(requested)
    log.info("encoder: %s (input %dpx, %dd)%s", spec.name, spec.input_px, spec.embed_dim,
              "" if spec.name == requested else f"  [fallback from '{requested}']")
    model, preprocess = load_encoder(spec)

    paths: list[str] = []
    for pattern in args.slides:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    if not paths:
        raise SystemExit(f"no slides matched: {args.slides}")
    log.info("found %d slide(s)", len(paths))

    extract_embeddings(paths, model, preprocess, cfg, args.outdir, force=args.force)
    log.info("done. read the cohort with pd.read_parquet(%r)", args.outdir)


if __name__ == "__main__":
    main()
