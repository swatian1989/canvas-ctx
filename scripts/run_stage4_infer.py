#!/usr/bin/env python
"""Stage 4: WSI intake, DX filtering, patch embedding, optional habitat
inference and tumour-bulk/leading-edge compartments.

Wires the EXISTING, tested functions in stage4_infer/wsi_pipeline.py
(`read_slide_info`, `is_diagnostic_slide`, `assign_compartments`) and stage3's
embedding extraction (extract_embeddings.py). None of those are rewritten
here.

Two model-dependent steps have NO implementation anywhere in this repo yet
(see STATUS.md): a tumour-probability detector and a trained habitat
classifier head. Both are optional here -- omit `--habitat-model` and this
script does slide validation + embedding extraction only, and says so
plainly, rather than fabricating compartments or habitat maps without a
real model.

    python scripts/run_stage4_infer.py \\
        --config config/crc_train_brca_apply.yaml \\
        --slides "data/raw/tcga_coad/*.svs" \\
        --outdir data/interim/stage4

Add `--habitat-model PATH --tumour-habitat-idx N` once a trained head exists
(the four-way benchmark's `mode=none` checkpoint, or any CanvasHead-shaped
state_dict) to also get per-patch habitat predictions and
tumour_bulk/leading_edge/other compartments.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage3_model.encoders import load_encoder, resolve_encoder  # noqa: E402
from canvas_brca.stage3_model.extract_embeddings import (  # noqa: E402
    ExtractConfig, extract_embeddings,
)
from canvas_brca.stage4_infer.wsi_pipeline import (  # noqa: E402
    assign_compartments, is_diagnostic_slide, read_slide_info,
)
from canvas_brca.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--outdir", default="data/interim/stage4")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--habitat-model", default=None,
                    help="optional: state_dict checkpoint for a trained CanvasHead; "
                         "without it, this script only validates slides and caches "
                         "embeddings")
    ap.add_argument("--tumour-habitat-idx", type=int, default=None,
                    help="which habitat class index is 'tumour', for compartments; "
                         "required together with --habitat-model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage4_infer")

    if bool(args.habitat_model) != (args.tumour_habitat_idx is not None):
        raise SystemExit("--habitat-model and --tumour-habitat-idx must be given together")

    raw = load_config(args.config)
    profile = raw["project"]["profile"]
    patching = raw["patching"]
    inference = raw["inference"]

    paths: list[str] = []
    for pattern in args.slides:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    if not paths:
        raise SystemExit(f"no slides matched: {args.slides}")

    # 1. DX filter + native mpp/mag validation -- never assume 40x
    dx_paths = []
    for p in paths:
        stem = Path(p).stem
        if inference.get("slide_type", "DX") == "DX" and not is_diagnostic_slide(stem):
            log.info("%s: not a DX barcode, skipping (frozen sections have "
                     "freezing artefact)", stem)
            continue
        info = read_slide_info(p)
        log.info("%s: %.3f mpp, %.0fx native, %s", info.slide_id, info.native_mpp,
                 info.native_mag, info.dimensions)
        dx_paths.append(p)
    log.info("%d/%d slides are DX and readable", len(dx_paths), len(paths))
    if not dx_paths:
        raise SystemExit("no DX slides survived filtering")

    # 2. patch embeddings -- the Phase 1 pipeline, unchanged
    target_mpp = patching["laptop_mpp"] if profile == "laptop" else patching["target_mpp"]
    cfg = ExtractConfig(
        patch_px=patching["patch_px"], stride_px=patching["stride_px"],
        target_mpp=target_mpp, min_tissue_frac=patching["min_tissue_frac"],
        max_patches_per_slide=patching.get("max_patches_per_slide"),
        batch_size=raw["model"].get("batch_size", 16),
        stain_norm=patching.get("stain_norm", "macenko"), seed=raw["project"]["seed"],
    )
    requested = args.encoder or raw["model"]["encoder"]
    spec = resolve_encoder(requested)
    model, preprocess = load_encoder(spec)
    emb_dir = Path(args.outdir) / "embeddings"
    extract_embeddings(dx_paths, model, preprocess, cfg, str(emb_dir), force=args.force)

    if not args.habitat_model:
        log.info("no --habitat-model given: stopping after slide validation + "
                 "embedding caching. Habitat prediction and compartments need a "
                 "trained head and a tumour detector (still MISSING per STATUS.md) "
                 "respectively.")
        return

    # 3. habitat prediction from the cached embeddings, IF a trained head is given
    from canvas_brca.stage3_model.head import CanvasHead

    embeddings = pd.read_parquet(emb_dir)
    emb_cols = sorted([c for c in embeddings.columns if c.startswith("emb_")],
                      key=lambda s: int(s.split("_")[1]))
    n_classes = raw["project"]["n_habitats"] + 1
    head = CanvasHead(embed_dim=len(emb_cols), n_classes=n_classes)
    head.load_state_dict(torch.load(args.habitat_model, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        logits = head(torch.as_tensor(embeddings[emb_cols].to_numpy(), dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    embeddings["habitat"] = probs.argmax(axis=1)
    embeddings["tumour_prob"] = probs[:, args.tumour_habitat_idx]

    # 4. compartments per slide -- assign_compartments is unchanged, tested
    all_compartments = []
    for slide_id, grp in embeddings.groupby("slide_id"):
        if len(grp) < inference.get("min_patches_per_compartment", 25):
            log.warning("%s: only %d patches, skipping compartments", slide_id, len(grp))
            continue
        comp = assign_compartments(
            grp, n_clusters=inference["compartments"]["n_clusters"],
            knn_k=inference["compartments"]["neighbourhood_k"], seed=raw["project"]["seed"],
        )
        all_compartments.append(comp)

    if all_compartments:
        out_df = pd.concat(all_compartments, ignore_index=True)
        out_path = Path(args.outdir) / "habitat_compartments.parquet"
        out_df.to_parquet(out_path, index=False)
        log.info("wrote %s (%d patches, %d slides)", out_path, len(out_df),
                 out_df["slide_id"].nunique())


if __name__ == "__main__":
    main()
