#!/usr/bin/env python
"""Verify the Orion IF-to-H&E registration instead of trusting it.

    python scripts/run_orion_registration_qc.py

Orion images both modalities from the SAME section and ships the H&E already
registered to the multiplex frame, so the expected affine is the identity.
The project brief is explicit that this must be verified, not assumed: a
silent half-cell offset would corrupt every downstream patch label while
still producing plausible-looking output.

There are no published landmark pairs for this specimen, so a landmark
residual cannot be computed directly. Instead this measures the residual the
data itself can support:

  1. Read the H&E at a coarse pyramid level and build a tissue mask.
  2. Bin the segmented cell centroids onto the same grid to get a cell
     density map.
  3. Cross-correlate the two via FFT and locate the peak.

If the two modalities are registered, the correlation peak sits at zero
offset. The peak displacement, converted to microns, is the registration
residual. This is a global rigid check: it will not detect local warping,
which is stated as a limitation rather than glossed over.
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

THRESHOLD_UM = 5.0  # [PAPER] centroid-to-centroid acceptance threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--he", default="data/raw/orion_crc/CRC01/CRC01-HE-registered.ome.tif")
    ap.add_argument("--cells", default="data/raw/orion_crc/CRC01/P37_S29-CRC01.csv")
    ap.add_argument("--level", type=int, default=4, help="H&E pyramid level to work at")
    ap.add_argument("--mpp", type=float, default=0.325, help="H&E level-0 microns/pixel")
    ap.add_argument("--sample", type=int, default=400_000, help="cells to sample")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("orion_qc")

    import tifffile

    tif = tifffile.TiffFile(args.he)
    series = tif.series[0]
    full_h, full_w = series.levels[0].shape[:2]
    lvl = series.levels[args.level]
    h, w = lvl.shape[:2]
    downsample = full_w / w
    um_per_px_level = args.mpp * downsample
    log.info("H&E level %d: %dx%d px (level0 %dx%d), downsample %.1fx, %.2f um/px",
             args.level, w, h, full_w, full_h, downsample, um_per_px_level)

    rgb = lvl.asarray()
    log.info("read H&E level into memory: %.0f MB", rgb.nbytes / 1e6)

    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 5)
    _, tissue = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = (tissue > 0).astype(np.float32)
    log.info("tissue mask: %.1f%% of the frame", 100 * tissue.mean())

    cells = pd.read_csv(args.cells, usecols=["X_centroid", "Y_centroid"])
    log.info("read %d cell centroids", len(cells))
    if len(cells) > args.sample:
        cells = cells.sample(args.sample, random_state=42)

    # centroids are level-0 pixels in the same frame as the H&E
    gx = np.clip((cells["X_centroid"].to_numpy() / downsample).astype(int), 0, w - 1)
    gy = np.clip((cells["Y_centroid"].to_numpy() / downsample).astype(int), 0, h - 1)
    density = np.zeros((h, w), dtype=np.float32)
    np.add.at(density, (gy, gx), 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=2.0)

    # direct agreement: what fraction of cells land on tissue
    on_tissue = float(tissue[gy, gx].mean())
    log.info("cells landing on tissue mask: %.2f%%", 100 * on_tissue)

    # FFT phase-style cross-correlation of the two maps
    a = tissue - tissue.mean()
    b = density - density.mean()
    corr = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    resid_px_level = float(np.hypot(dx, dy))
    resid_um = resid_px_level * um_per_px_level
    log.info("cross-correlation peak offset: dx=%d dy=%d px (level %d)", dx, dy, args.level)

    # A zero-pixel peak does not mean a zero residual: this check cannot
    # resolve below one pixel at the level it was run. Report the BOUND, not
    # a false-precision zero.
    resid_str = (f"< {um_per_px_level:.1f} um (below this check's resolution)"
                 if resid_px_level == 0 else f"{resid_um:.1f} um")
    log.info("global registration residual: %s", resid_str)

    resid_for_test = resid_um if resid_px_level > 0 else um_per_px_level
    verdict = "PASS" if resid_for_test <= THRESHOLD_UM else "FAIL"
    log.info("residual %s vs [PAPER] %.1f um threshold -> %s",
             resid_str, THRESHOLD_UM, verdict)
    if verdict == "FAIL":
        log.warning("residual exceeds threshold: PALOM refinement would be required "
                    "before label transfer")

    # ---- QC figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from canvas_brca.reporting.style import (
        NAVY, STEEL_BLUE, apply_style, letter_panels, save_figure, source_caption,
    )

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("H&E (registered), pyramid level %d" % args.level, fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].grid(False)

    axes[1].imshow(tissue, cmap="gray")
    sub = np.random.default_rng(0).choice(len(gx), size=min(40000, len(gx)), replace=False)
    axes[1].scatter(gx[sub], gy[sub], s=0.12, c=STEEL_BLUE, alpha=0.35, linewidths=0)
    axes[1].set_title("tissue mask + IF cell centroids", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].grid(False)

    win = 60
    cy, cx = h // 2, w // 2
    corr_shift = np.fft.fftshift(corr)
    axes[2].imshow(corr_shift[cy - win:cy + win, cx - win:cx + win],
                   cmap="viridis", extent=[-win, win, win, -win])
    axes[2].axvline(0, color="white", lw=0.7, ls="--")
    axes[2].axhline(0, color="white", lw=0.7, ls="--")
    axes[2].plot(dx, dy, "o", mfc="none", mec="red", ms=12, mew=1.8)
    axes[2].set_title(f"cross-correlation peak\nresidual {resid_str} ({verdict})",
                      fontsize=10)
    axes[2].set_xlabel(f"dx (level-{args.level} px)")
    axes[2].set_ylabel(f"dy (level-{args.level} px)")
    axes[2].grid(False)

    letter_panels(axes)
    source_caption(fig, f"REAL DATA (Orion CRC01, PMID n/a doi:10.1038/s43018-023-00576-1; "
                        f"{len(cells):,} cell centroids sampled, H&E {full_w}x{full_h} px "
                        f"at {args.mpp} um/px).", y=-0.04)
    paths = save_figure(fig, "F6_registration_qc", args.figdir)
    log.info("wrote %s", paths["png"])

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "specimen": "CRC01", "he_width_px": int(full_w), "he_height_px": int(full_h),
        "mpp_level0_um": args.mpp, "pyramid_level_used": args.level,
        "um_per_px_at_level": round(um_per_px_level, 3),
        "n_cells_total": int(len(pd.read_csv(args.cells, usecols=["CellID"]))),
        "n_cells_sampled": int(len(cells)),
        "frac_cells_on_tissue": round(on_tissue, 4),
        "offset_dx_px_level": int(dx), "offset_dy_px_level": int(dy),
        "residual_um": round(resid_um, 2), "residual_bound_um": round(um_per_px_level, 2), "residual_statement": resid_str, "threshold_um": THRESHOLD_UM,
        "verdict": verdict,
    }
    (out / "orion_registration_qc.json").write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out / "orion_registration_qc.json")


if __name__ == "__main__":
    main()
