#!/usr/bin/env python
"""Encode labelled Orion H&E patches to cached embeddings.

    python scripts/encode_orion_patches.py

Writes the exact schema scripts/run_final_benchmark.py reads:
    slide_id, x_um, y_um, patch_x, patch_y, label, emb_0..emb_{D-1}

Orion H&E are pyramidal OME-TIFFs, not the formats openslide handles here,
so patches are read through a zarr view of level 0 rather than via
stage3_model/extract_embeddings.py's openslide path. Everything after the
read is the same pipeline: Macenko normalisation, then the encoder with its
own preprocessing.

Resumable per specimen: a finished shard is skipped.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage3_model.encoders import load_encoder, resolve_encoder  # noqa: E402
from canvas_brca.stage3_model.extract_embeddings import macenko_normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="data/interim/orion_cohort/cohort_patch_labels.parquet")
    ap.add_argument("--orion-dir", default="data/raw/orion_crc")
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-out-px", type=int, default=224)
    ap.add_argument("--outdir", default="data/interim/orion_embeddings")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("encode_orion")

    import tifffile
    import zarr

    patches = pd.read_parquet(args.patches)
    log.info("%d patches over %d specimens", len(patches), patches["sample_id"].nunique())

    spec = resolve_encoder(args.encoder)
    log.info("encoder %s (%dd)", spec.name, spec.embed_dim)
    model, preprocess = load_encoder(spec)
    model.eval()
    torch.set_grad_enabled(False)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for name, grp in patches.groupby("sample_id"):
        shard = out / f"{name}.parquet"
        if shard.exists() and not args.force:
            log.info("%s: shard exists, skipping", name)
            continue

        d = Path(args.orion_dir) / name
        he = next(iter(d.glob("*-HE-registered.ome.tif")), None)
        if he is None:
            log.warning("%s: no registered H&E, skipping", name)
            continue

        store = tifffile.imread(str(he), aszarr=True, level=0)
        z = zarr.open(store, mode="r")
        H, W = z.shape[:2]
        p = int(grp["patch_px_native"].iloc[0])
        log.info("--- %s: %d patches, H&E %dx%d, %dpx windows ---",
                 name, len(grp), W, H, p)

        rows, batch_img, batch_meta = [], [], []
        t0 = time.time()

        def flush():
            if not batch_img:
                return
            tens = torch.stack([preprocess(Image.fromarray(im)) for im in batch_img])
            emb = model(tens).detach().cpu().numpy()
            for meta, vec in zip(batch_meta, emb):
                r = dict(meta)
                r.update({f"emb_{i}": float(v) for i, v in enumerate(vec)})
                rows.append(r)
            batch_img.clear()
            batch_meta.clear()

        for i, (_, row) in enumerate(grp.iterrows()):
            x0, y0 = int(row["x_px"]), int(row["y_px"])
            if y0 + p > H or x0 + p > W:
                continue
            tile = np.asarray(z[y0:y0 + p, x0:x0 + p])
            if tile.ndim != 3 or tile.shape[2] != 3:
                continue
            if p != args.patch_out_px:
                tile = np.array(Image.fromarray(tile).resize(
                    (args.patch_out_px, args.patch_out_px), Image.BILINEAR))
            tile = macenko_normalize(tile)
            batch_img.append(tile)
            batch_meta.append({
                "slide_id": name,
                "x_um": float(row["x_px"]) * 0.325,
                "y_um": float(row["y_px"]) * 0.325,
                "patch_x": int(row["patch_x"]), "patch_y": int(row["patch_y"]),
                "label": int(row["label"]),
            })
            if len(batch_img) >= args.batch_size:
                flush()
                if (i + 1) % (args.batch_size * 10) == 0:
                    rate = (i + 1) / (time.time() - t0)
                    log.info("  %d/%d  %.1f patches/s", i + 1, len(grp), rate)
        flush()

        if not rows:
            log.warning("%s: nothing encoded", name)
            continue
        df = pd.DataFrame(rows)
        tmp = shard.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(shard)
        log.info("%s: wrote %d embeddings in %.1f min",
                 name, len(df), (time.time() - t0) / 60)

    shards = sorted(out.glob("*.parquet"))
    log.info("done: %d shards. Read the cohort with pd.read_parquet(%r)",
             len(shards), str(out))


if __name__ == "__main__":
    main()
