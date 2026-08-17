#!/usr/bin/env python
"""Recover completed benchmark runs from the log when the process has not finished.

    python scripts/salvage_benchmark_log.py results/real_benchmark.log

run_final_benchmark.py writes its CSVs only after every seed and mode has
finished, so a run that is killed, stalls, or hits a power cut loses everything
even though most of the work is done. Each completed run does however log its
own result line, and the training curves are logged every five epochs. This
reconstructs both from the log so partial work survives.

This is a salvage tool, not a substitute for the real output. Runs that never
finished are simply absent, and the summary states how many of the expected
seeds x modes are present so a partial table cannot be mistaken for a full one.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULT = re.compile(
    r"seed (?P<seed>\d+) (?P<mode>\w+)\s+macro_f1 (?P<f1>[\d.]+)\s+"
    r"min_recall (?P<recall>[\d.]+)(?:\s+COLLAPSED classes: (?P<collapsed>\[[^\]]*\]))?")
EPOCH = re.compile(r"epoch\s+(?P<epoch>\d+)\s+loss (?P<loss>[\d.]+)\s+"
                   r"acc (?P<acc>[\d.]+)\s+f1 (?P<f1>[\d.]+)")
START = re.compile(r"HabitatNet\[(?P<mode>\w+)\]")
SEEDLINE = re.compile(r"seed (?P<seed>\d+) \| .* slides")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="results/real_benchmark.log")
    ap.add_argument("--outdir", default="results/real_benchmark")
    ap.add_argument("--modes", nargs="+", default=["none", "graph", "grid2d", "grid3d"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], []
    seed, mode = None, None
    for line in text:
        m = SEEDLINE.search(line)
        if m:
            seed = int(m.group("seed"))
        m = START.search(line)
        if m:
            mode = m.group("mode")
        m = EPOCH.search(line)
        if m and seed is not None and mode is not None:
            curves.append({"seed": seed, "mode": mode, "epoch": int(m.group("epoch")),
                           "loss": float(m.group("loss")), "acc": float(m.group("acc")),
                           "f1": float(m.group("f1"))})
        m = RESULT.search(line)
        if m:
            col = m.group("collapsed")
            rows.append({"seed": int(m.group("seed")), "mode": m.group("mode"),
                         "macro_f1": float(m.group("f1")),
                         "min_recall": float(m.group("recall")),
                         "collapsed_classes": col or "[]",
                         "n_collapsed": 0 if not col or col == "[]"
                         else len([x for x in col.strip("[]").split(",") if x.strip()])})

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no completed runs found in the log")
    res.to_csv(out / "final_benchmark_SALVAGED.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "training_curves_SALVAGED.csv", index=False)

    expected = len(args.seeds) * len(args.modes)
    print(f"recovered {len(res)} of {expected} expected runs "
          f"({len(res)/expected:.0%}) from {args.log}")
    missing = [(s, m) for s in args.seeds for m in args.modes
               if not ((res["seed"] == s) & (res["mode"] == m)).any()]
    if missing:
        print(f"MISSING ({len(missing)}): " +
              ", ".join(f"seed{s}-{m}" for s, m in missing))

    print("\nmacro-F1 by mode (mean over the seeds that finished):")
    g = res.groupby("mode")["macro_f1"].agg(["count", "mean", "std", "min", "max"])
    g = g.reindex([m for m in args.modes if m in g.index])
    print(g.round(4).to_string())

    base = res[res["mode"] == "none"].set_index("seed")["macro_f1"]
    print("\npaired difference vs mode 'none', on seeds where BOTH finished:")
    for m in args.modes:
        if m == "none":
            continue
        sub = res[res["mode"] == m].set_index("seed")["macro_f1"]
        common = base.index.intersection(sub.index)
        if len(common) < 2:
            continue
        d = (sub[common] - base[common])
        print(f"  {m:7} n={len(common)}  mean {d.mean():+.4f}  "
              f"median {d.median():+.4f}  wins {int((d>0).sum())}/{len(common)}")

    print(f"\nclass collapse: {int((res['n_collapsed']>0).sum())} of {len(res)} runs "
          f"collapsed at least one class; median {res['n_collapsed'].median():.0f} "
          f"classes per run")
    print(f"wrote {out/'final_benchmark_SALVAGED.csv'}")


if __name__ == "__main__":
    main()
