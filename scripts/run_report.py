#!/usr/bin/env python
"""Regenerate the complete analysis report from cached artefacts.

    python scripts/run_report.py

Idempotent: rerunning overwrites the same figure/table/report files in place
and produces the same content, because every figure and table function reads
already-computed artefacts (results/*.csv, data/interim/sim_features.parquet)
rather than re-running training or feature extraction. Nothing here trains a
model or encodes a slide.

Outputs:
    figures/F*.png, figures/F*.pdf     22 figures, 300 dpi raster + vector
    results/tables/T*.csv              10 tables
    reports/analysis_report.md         markdown, relative image links
    reports/analysis_report.html       self-contained, base64-embedded images
    reports/analysis_report.docx       navy/steel blue, Calibri, justified

The few figures that run new computation (F15-F18, F20-F22) call only the
existing tested functions in spatial_features.py / signature.py, both of
which are protected files and are NOT modified.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.reporting.report import build_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--config", default="config/crc_train_brca_apply.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("report")

    out = build_report(figures_dir=args.figures_dir, tables_dir=args.tables_dir,
                       reports_dir=args.reports_dir, config_path=args.config)

    log.info("figures: %d (%d MISSING-DATA placeholders)",
             len(out["figures"]), out["n_missing_figures"])
    log.info("tables:  %d (%d MISSING-DATA placeholders)",
             len(out["tables"]), out["n_missing_tables"])
    for kind in ("md", "html", "docx"):
        p = Path(args.reports_dir) / f"analysis_report.{kind}"
        log.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024)

    real = [f["id"] for f in out["figures"].values() if f["source"].startswith("REAL")]
    log.info("figures built from REAL cohort data: %s",
             ", ".join(real) if real else "NONE (data/raw/ is empty)")


if __name__ == "__main__":
    main()
