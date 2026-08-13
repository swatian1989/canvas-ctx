#!/usr/bin/env python
"""Download the Schurch colorectal CODEX single-cell table.

    python scripts/download_schurch.py

Source: Mendeley Data, doi:10.17632/mpjzbtfgfr.1, released with
Schurch et al., Cell 2020 (PMID 32763154). 223 MB, CC BY 4.0.

The download is verified by SHA-256 against the published checksum, and an
already-complete file is left alone, so this is safe to rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

FILE_URL = ("https://data.mendeley.com/public-files/datasets/mpjzbtfgfr/files/"
            "c24351b3-76d7-444f-9edf-0246356b0c78/file_downloaded")
EXPECTED_SHA256 = "416cc3926a7a900ce3b22a33be699535ca35ae85fca4585a8ba5dad6d7f3c677"
EXPECTED_BYTES = 222_999_273


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/CRC_clusters_neighborhoods_markers.csv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-checksum", action="store_true",
                    help="skip verification (hashing 223 MB takes a few seconds)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("download_schurch")

    dest = Path(args.out)
    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES and not args.force:
        log.info("%s already complete (%.0f MB), skipping", dest,
                 dest.stat().st_size / 1e6)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %.0f MB from Mendeley Data ...", EXPECTED_BYTES / 1e6)

    def progress(blocks: int, block_size: int, total: int) -> None:
        got = blocks * block_size
        if total > 0 and blocks % 500 == 0:
            log.info("  %.0f%% (%.0f / %.0f MB)", 100 * got / total,
                     got / 1e6, total / 1e6)

    urllib.request.urlretrieve(FILE_URL, tmp, reporthook=progress)

    size = tmp.stat().st_size
    if size != EXPECTED_BYTES:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"size mismatch: got {size}, expected {EXPECTED_BYTES}. "
                         "The download was truncated; rerun.")

    if not args.skip_checksum:
        log.info("verifying checksum ...")
        got = sha256(tmp)
        if got != EXPECTED_SHA256:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch:\n  got      {got}\n  "
                             f"expected {EXPECTED_SHA256}\nFile not kept.")
        log.info("checksum verified")

    tmp.replace(dest)
    log.info("wrote %s (%.0f MB)", dest, size / 1e6)


if __name__ == "__main__":
    main()
