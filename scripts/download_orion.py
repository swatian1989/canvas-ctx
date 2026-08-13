#!/usr/bin/env python
"""Download Orion CRC specimens (registered H&E + single-cell table only).

    python scripts/download_orion.py --specimens CRC02 CRC03 ...

Deliberately does NOT fetch the 19-channel multiplex OME-TIFFs: they are
44-147 GB each and nothing downstream reads them. The registered H&E and the
single-cell table are what label transfer and patch extraction need.

Resumable and verifying: every file is size-checked against the S3 listing,
and an already-complete file is skipped rather than re-fetched.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://lin-2023-orion-crc.s3.amazonaws.com"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_bucket() -> dict[str, dict]:
    token, items = None, []
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(urllib.request.urlopen(url, timeout=120).read())
        for c in root.findall("s:Contents", NS):
            items.append((c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)))
        nxt = root.find("s:NextContinuationToken", NS)
        if nxt is None:
            break
        token = nxt.text

    spec: dict[str, dict] = {}
    for key, size in items:
        m = re.match(r"data/(CRC\d+)/", key)
        if not m:
            continue
        d = spec.setdefault(m.group(1), {})
        if key.endswith("-registered.ome.tif"):
            d["he"] = (key, size)
        elif key.endswith(".csv") and "markers" not in key:
            d["csv"] = (key, size)
        elif key.endswith("markers.csv"):
            d["markers"] = (key, size)
    return spec


def fetch(key: str, size: int, dest: Path, log) -> bool:
    if dest.exists() and dest.stat().st_size == size:
        log.info("  skip %s (already complete, %.2f GB)", dest.name, size / 1e9)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    log.info("  get  %s (%.2f GB)", dest.name, size / 1e9)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = tmp.stat().st_size
    if got != size:
        log.error("  SIZE MISMATCH %s: got %d expected %d", dest.name, got, size)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimens", nargs="+",
                    default=[f"CRC{i:02d}" for i in range(2, 13)])
    ap.add_argument("--outdir", default="data/raw/orion_crc")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("download_orion")

    spec = list_bucket()
    missing = [s for s in args.specimens if s not in spec]
    if missing:
        raise SystemExit(f"not in bucket: {missing}")

    total = sum(spec[s]["he"][1] + spec[s]["csv"][1] for s in args.specimens)
    log.info("%d specimens, %.1f GB (H&E + single-cell only)",
             len(args.specimens), total / 1e9)

    ok, failed = [], []
    for name in args.specimens:
        log.info("--- %s ---", name)
        d = Path(args.outdir) / name
        good = True
        for kind, fname in (("markers", "markers.csv"),
                            ("csv", f"{name}-cells.csv"),
                            ("he", f"{name}-HE-registered.ome.tif")):
            if kind not in spec[name]:
                log.warning("  %s has no %s", name, kind)
                good = False
                continue
            key, size = spec[name][kind]
            good &= fetch(key, size, d / fname, log)
        (ok if good else failed).append(name)

    log.info("complete: %d ok, %d failed", len(ok), len(failed))
    if failed:
        log.warning("failed: %s -- rerun to resume", failed)


if __name__ == "__main__":
    main()
