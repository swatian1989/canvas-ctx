#!/usr/bin/env python
"""Stage 2: register, transfer CN labels, and patient-split the paired cohort.

Wires the EXISTING, tested functions in stage2_pair/label_transfer.py --
`assign_cn_to_he_nuclei` and `label_patches` are not touched here. This
script is glue: per-sample registration + label transfer, then a
patient-level train/val/test split across the whole cohort.

One `--sample` per slide (repeatable), a simple `key=value,...` spec:

    python scripts/run_stage2_pair.py \\
        --config config/crc_train_brca_apply.yaml \\
        --cn-assignments data/processed/cn_assignments.parquet \\
        --sample sample_id=ORION01,image_id=ORION01,nuclei=data/raw/orion_crc/ORION01_nuclei.parquet,affine=data/raw/orion_crc/ORION01_affine.npy \\
        --sample sample_id=ORION02,image_id=ORION02,nuclei=data/raw/orion_crc/ORION02_nuclei.parquet,spatial_ref=data/raw/orion_crc/ORION02_if.ome.tif,he_ref=data/raw/orion_crc/ORION02_he.ome.tif \\
        --outdir data/interim/stage2_labels

Per-sample keys:
    sample_id   patient/slide identifier (also the split unit)
    image_id    matches `image_id` in --cn-assignments for this sample
    nuclei      parquet/csv of H&E-segmented nuclei, columns x_px, y_px --
                NOT produced by any code in this repo yet (StarDist/CellViT
                wrapper is still MISSING per STATUS.md); supply it externally.
    affine      OPTIONAL: path to a saved 2x3 numpy array (.npy) to use
                directly, e.g. a converted 10x alignment file.
    spatial_ref, he_ref
                OPTIONAL alternative to `affine`: image paths passed to
                `estimate_affine_palom`, which is still a stub
                (NotImplementedError) -- expected to fail loudly until that
                is implemented against real data.
Exactly one of {affine} or {spatial_ref,he_ref} must be given per sample.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage2_pair.label_transfer import (  # noqa: E402
    AffineTransform, assign_cn_to_he_nuclei, estimate_affine_palom, label_patches,
)
from canvas_brca.utils.config import load_config  # noqa: E402


def _parse_sample_spec(spec: str) -> dict:
    out = {}
    for part in spec.split(","):
        key, _, value = part.partition("=")
        if not key or not value:
            raise ValueError(f"bad --sample entry {spec!r}: expected key=value pairs")
        out[key.strip()] = value.strip()
    for required in ("sample_id", "image_id", "nuclei"):
        if required not in out:
            raise ValueError(f"--sample {spec!r} missing required key '{required}'")
    return out


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _resolve_affine(spec: dict) -> AffineTransform:
    has_affine = "affine" in spec
    has_palom = "spatial_ref" in spec and "he_ref" in spec
    if has_affine == has_palom:
        raise ValueError(
            f"sample {spec['sample_id']}: give exactly one of `affine=` or "
            f"`spatial_ref=...,he_ref=...`, not both/neither"
        )
    if has_affine:
        matrix = np.load(spec["affine"])
        if matrix.shape != (2, 3):
            raise ValueError(f"{spec['affine']}: expected a (2,3) affine, got {matrix.shape}")
        return AffineTransform(matrix=matrix)
    return estimate_affine_palom(spec["spatial_ref"], spec["he_ref"])


def patient_level_split(
    sample_ids: list[str], n_train: int, n_val: int, n_test: int, seed: int,
) -> dict[str, str]:
    """Shuffle sample_ids and assign train/val/test. [PAPER-adjacent] never
    split at the patch level -- this project's whole cohort is small enough
    (8/2/2) that the split unit is the patient/slide itself.
    """
    n_needed = n_train + n_val + n_test
    if len(sample_ids) < n_needed:
        raise ValueError(
            f"need {n_needed} samples for an {n_train}/{n_val}/{n_test} split, "
            f"got {len(sample_ids)}: {sample_ids}"
        )
    rng = np.random.default_rng(seed)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)
    out = {}
    for sid in shuffled[:n_train]:
        out[sid] = "train"
    for sid in shuffled[n_train:n_train + n_val]:
        out[sid] = "val"
    for sid in shuffled[n_train + n_val:n_train + n_val + n_test]:
        out[sid] = "test"
    for sid in shuffled[n_needed:]:
        out[sid] = "unused"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--config", default="config/crc_train_brca_apply.yaml")
    ap.add_argument("--cn-assignments", required=True,
                    help="stage1 output: image_id, cell_id, x_um, y_um, cn[, cn_label]")
    ap.add_argument("--sample", action="append", required=True, dest="samples",
                    help="key=value,... spec, repeatable (one per slide)")
    ap.add_argument("--outdir", default="data/interim/stage2_labels")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage2_pair")

    raw = load_config(args.config)
    paired = raw["paired"]
    mpp = paired["native_mpp"]
    max_dist = paired["max_centroid_distance_um"]
    patch_px = raw["patching"]["patch_px"]
    n_habitats = raw["project"]["n_habitats"]
    seed = raw["project"]["seed"]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "patch_labels.parquet"
    if final.exists() and not args.force:
        log.info("%s exists, use --force to rerun", final)
        return

    cn_assignments = pd.read_parquet(args.cn_assignments)
    log.info("cn_assignments: %d cells, %d images", len(cn_assignments),
              cn_assignments["image_id"].nunique())

    all_patches = []
    failed = []
    for raw_spec in args.samples:
        spec = _parse_sample_spec(raw_spec)
        sid, image_id = spec["sample_id"], spec["image_id"]
        log.info("--- sample %s (image_id=%s) ---", sid, image_id)

        try:
            spatial_cells = cn_assignments[cn_assignments["image_id"] == image_id]
            if spatial_cells.empty:
                log.warning("%s: no cells for image_id=%s in --cn-assignments, skipping",
                            sid, image_id)
                failed.append(sid)
                continue

            he_nuclei = _read_table(spec["nuclei"])
            transform = _resolve_affine(spec)

            labelled = assign_cn_to_he_nuclei(he_nuclei, spatial_cells, transform, mpp,
                                              max_distance_um=max_dist)
            patches = label_patches(labelled, patch_size_px=patch_px, n_habitats=n_habitats)
            if patches.empty:
                log.warning("%s: no patches survived labelling", sid)
                failed.append(sid)
                continue
            patches["sample_id"] = sid
            all_patches.append(patches)
        except Exception as exc:
            # One sample's missing file / unimplemented registration path (e.g.
            # estimate_affine_palom's NotImplementedError) must not blow away
            # every other sample already processed in this batch.
            log.error("%s: failed (%s: %s), skipping this sample", sid, type(exc).__name__, exc)
            failed.append(sid)

    if failed:
        log.warning("%d/%d samples failed or produced nothing: %s",
                    len(failed), len(args.samples), failed)
    if not all_patches:
        raise SystemExit("no sample produced any labelled patches -- nothing to split")

    combined = pd.concat(all_patches, ignore_index=True)
    try:
        split_map = patient_level_split(
            sorted(combined["sample_id"].unique()),
            paired["n_train_slides"], paired["n_val_slides"], paired["n_test_slides"], seed,
        )
    except ValueError as exc:
        # Don't let a split-count shortfall (e.g. one sample failed above)
        # throw away the labelling work already done for every OTHER sample.
        # Save it unsplit so a rerun with --force after fixing the missing
        # sample doesn't have to re-do the expensive part.
        partial = out / "patch_labels_partial_unsplit.parquet"
        combined.to_parquet(partial, index=False)
        log.error("wrote %s (%d patches, %d samples) but could NOT split: %s",
                  partial, len(combined), combined["sample_id"].nunique(), exc)
        raise SystemExit(
            f"patient-level split needs {paired['n_train_slides']}+"
            f"{paired['n_val_slides']}+{paired['n_test_slides']} samples, only "
            f"{combined['sample_id'].nunique()} succeeded. Fix/add the missing "
            f"sample(s) and rerun --force; labelled patches for the samples that "
            f"DID succeed are saved at {partial}."
        ) from exc
    combined["split"] = combined["sample_id"].map(split_map)

    combined.to_parquet(final, index=False)
    log.info("wrote %s (%d patches, %d samples)", final, len(combined),
              combined["sample_id"].nunique())
    log.info("samples per split: %s",
              {k: v for k, v in pd.Series(split_map).value_counts().items()})
    log.info("class distribution overall:\n%s",
              combined["label"].value_counts().sort_index().to_string())
    log.info("class distribution by split:\n%s",
              combined.groupby("split")["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
