#!/usr/bin/env python
"""Generate the manuscript draft: reports/manuscript.{md,docx}.

    python scripts/run_manuscript.py

Writes no results. Sections that need data this project does not yet have
are emitted as explicit [RESULTS PENDING - requires <file>] markers, in red
in the .docx, so an incomplete claim cannot be mistaken for a finished one.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.reporting.manuscript import REFERENCES, build_manuscript  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("manuscript")

    out = build_manuscript(args.reports_dir)
    for kind in ("md", "docx"):
        p = Path(args.reports_dir) / f"manuscript.{kind}"
        log.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024)

    log.info("%d references, %d could not be independently verified",
             out["n_refs"], out["n_unverified_refs"])
    for r in REFERENCES:
        if not r.verified:
            log.warning("UNVERIFIED reference [%s]: %s", r.key, r.note)
    log.info("%d [RESULTS PENDING] slots remain -- these are the sections that "
             "need data before the manuscript can be completed", out["n_pending"])


if __name__ == "__main__":
    main()
