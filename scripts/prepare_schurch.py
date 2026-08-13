#!/usr/bin/env python
"""Convert the Schurch 2020 CRC CODEX table into the tidy schema stage 1 expects.

    python scripts/prepare_schurch.py

Source: Mendeley Data mpjzbtfgfr v1, CRC_clusters_neighborhoods_markers.csv
        (223 MB, sha256 416cc392...c677), Schurch et al. Cell 2020,
        PMID 32763154.

WHY THIS SCRIPT EXISTS: stage 1 needs `x_um`/`y_um` in MICRONS, and this
table stores `X:X`/`Y:Y` in PIXELS. `run_stage1_cn.py --colmap` can rename
columns but cannot rescale them, so the conversion has to happen here.

COORDINATE UNITS, established before any analysis:
  - Raw X spans 0-1919 and Y spans 0-1439 across every one of the 140
    images: a 1920x1440 sensor grid, i.e. pixels, not microns.
  - Pixel size 377.44 nm = 0.37744 um/px, documented for THIS dataset by
    the Cancer Imaging Archive collection page (CRC_FFPE-CODEX_CellNeighs):
    "lateral resolution of 377.44 nm/pixel", CFI Plan Apo lambda 20x/0.75
    objective on a Keyence BZ-X710.
  - Independent empirical check: at that scale the mean cell density gives
    ~24 cells inside a 40 um radius, against the ~25 the CANVAS STAR Methods
    states. The conversion is confirmed by the data itself, not assumed.
  - NOTE the same TCIA page documents half-resolution "montage" images at
    188.72 nm/px. These single-cell coordinates are full resolution: each
    image spans the full 1920x1440 frame.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.stage1_cn.neighborhoods import validate_cells  # noqa: E402

MICRONS_PER_PIXEL = 0.37744        # TCIA, this dataset. See module docstring.
ARTEFACT_CELL_TYPES = ("dirt",)    # imaging artefact, not a biological class

SRC_COLS = ["CellID", "File Name", "patients", "groups", "spots", "Region",
            "ClusterName", "neighborhood name", "neighborhood10", "X:X", "Y:Y"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/CRC_clusters_neighborhoods_markers.csv")
    ap.add_argument("--out", default="data/interim/schurch_crc_cells.parquet")
    ap.add_argument("--mpp", type=float, default=MICRONS_PER_PIXEL)
    ap.add_argument("--keep-artefacts", action="store_true",
                    help="retain 'dirt' cells (excluded by default)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("prepare_schurch")

    out = Path(args.out)
    if out.exists() and not args.force:
        log.info("%s exists, use --force to rerun", out)
        return

    log.info("reading %s", args.src)
    df = pd.read_csv(args.src, usecols=SRC_COLS, low_memory=False)
    log.info("read %d cells, %d images, %d patients",
             len(df), df["File Name"].nunique(), df["patients"].nunique())

    x_px_span = df["X:X"].max() - df["X:X"].min()
    y_px_span = df["Y:Y"].max() - df["Y:Y"].min()
    log.info("raw coordinate span: X %.0f px, Y %.0f px -> PIXELS (1920x1440 grid)",
             x_px_span, y_px_span)
    log.info("converting at %.5f um/px -> %.1f x %.1f um field of view",
             args.mpp, x_px_span * args.mpp, y_px_span * args.mpp)

    tidy = pd.DataFrame({
        "image_id": df["File Name"].astype(str),
        "cell_id": df["CellID"].astype(np.int64),
        "x_um": df["X:X"].astype(float) * args.mpp,
        "y_um": df["Y:Y"].astype(float) * args.mpp,
        "cell_type": df["ClusterName"].astype(str),
        # carried through for downstream stratification and validation, not
        # used by CN discovery itself
        "patient_id": df["patients"].astype(str),
        "group": df["groups"].astype(str),
        "published_cn": df["neighborhood name"].astype(str),
        "published_cn_id": df["neighborhood10"],
    })

    if not args.keep_artefacts:
        before = len(tidy)
        tidy = tidy[~tidy["cell_type"].isin(ARTEFACT_CELL_TYPES)].reset_index(drop=True)
        log.info("dropped %d '%s' cells (%.1f%%) as imaging artefact",
                 before - len(tidy), "/".join(ARTEFACT_CELL_TYPES),
                 100 * (before - len(tidy)) / before)

    # cell_id is only unique within image in this table; make it globally
    # unique so validate_cells' duplicate check is meaningful
    tidy["cell_id"] = np.arange(len(tidy), dtype=np.int64)

    validate_cells(tidy)

    out.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(out, index=False)
    log.info("wrote %s: %d cells, %d images, %d cell types, %d published CNs",
             out, len(tidy), tidy["image_id"].nunique(),
             tidy["cell_type"].nunique(), tidy["published_cn"].nunique())


if __name__ == "__main__":
    main()
