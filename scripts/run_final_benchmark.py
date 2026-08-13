#!/usr/bin/env python
"""FINAL BENCHMARK — Method 1 vs Method 2 (graph / 2D grid / 3D grid).

One controlled ablation. Same cached embeddings, same labels, same sample-level
splits, same focal loss, same weighted sampler, same CANVAS head. The only
variable is how neighbourhood evidence reaches the head.

    none    Method 1. CANVAS exactly, per-patch.
    graph   k-NN deep-set + distance-biased attention (permutation invariant)
    grid2d  local WxW feature grid, 2D conv + learned positional encoding
    grid3d  local SxWxW multi-scale cube, 3D conv across magnification

    python scripts/run_final_benchmark.py --simulate
    python scripts/run_final_benchmark.py \
        --embeddings data/interim/patch_embeddings.parquet \
        --modes none graph grid2d grid3d --epochs 30

Embeddings parquet: slide_id, x_um, y_um, label, emb_0..emb_{D-1}.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.method2_context.habitat_net import (  # noqa: E402
    HabitatNet, HabitatNetConfig,
)
from canvas_brca.method2_context.unified_dataset import (  # noqa: E402
    UnifiedPatchDataset, make_loaders, train_unified,
)
from canvas_brca.stage3_model.evaluate import evaluate  # noqa: E402


def simulate(n_slides=14, side=26, dim=64, n_classes=11, seed=0):
    """Fixture where habitat depends on regional context AND on direction.

    Each slide is partitioned into oriented bands, so habitat identity depends
    on where a patch sits relative to a boundary, not only on which region it
    is in. A single patch embedding carries a weak noisy signal. Permutation
    invariant pooling (graph) can recover the region; only the grid can recover
    the orientation.

    This fixture is CONSTRUCTED to reward the grid. It is a machinery check.
    It says nothing about breast tissue and must not appear in a manuscript.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, (n_classes, dim))
    frames = []
    for s in range(n_slides):
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        gx, gy = gx.ravel(), gy.ravel()
        band = ((gx + 2 * gy) // 4) % n_classes          # oriented, directional
        emb = centroids[band] + rng.normal(0, 3.2, (len(band), dim))
        frames.append(pd.DataFrame({
            "slide_id": f"S{s:02d}", "x_um": gx * 112.0, "y_um": gy * 112.0,
            "label": band, **{f"emb_{i}": emb[:, i] for i in range(dim)},
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["none", "graph", "grid2d", "grid3d"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--n-scales", type=int, default=3)
    ap.add_argument("--k-neighbours", type=int, default=8)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--grid-channels", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="multiple seeds -> mean +/- SD and a paired test vs mode[0]")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("final")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = simulate(seed=args.seeds[0]) if args.simulate else pd.read_parquet(args.embeddings)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")],
                      key=lambda s: int(s.split("_")[1]))
    X = df[emb_cols].to_numpy(np.float32)
    coords = df[["x_um", "y_um"]].to_numpy(np.float64)
    y = df["label"].to_numpy(np.int64)
    slides = df["slide_id"].to_numpy()

    # SAMPLE-LEVEL split. [PAPER] Patch-level splitting plus spatial context is
    # catastrophic leakage: neighbours land in both sets.
    rows, histories, per_class_rows, confusion_rows = [], {}, [], []
    for seed in args.seeds:
      # Each seed reshuffles the SLIDE split as well as model init, so the
      # spread across seeds reflects both sources of variance. A single seed
      # cannot distinguish a real 0.02 macro-F1 gap from split luck.
      rng = np.random.default_rng(seed)
      uniq = np.unique(slides)
      rng.shuffle(uniq)
      n_val = max(1, int(args.val_frac * len(uniq)))
      val = set(uniq[:n_val])
      is_val = np.array([s in val for s in slides])
      log.info("seed %d | %d train / %d val slides | %d / %d patches",
               seed, len(uniq) - n_val, n_val, (~is_val).sum(), is_val.sum())

      for mode in args.modes:
        t0 = time.time()
        cfg = HabitatNetConfig(
            context_mode=mode, n_classes=int(y.max()) + 1, proj_dim=args.proj_dim,
            k_neighbours=args.k_neighbours, window=args.window,
            n_scales=args.n_scales, grid_channels=args.grid_channels, seed=seed,
        )
        ds_tr = UnifiedPatchDataset(X[~is_val], coords[~is_val], y[~is_val],
                                    slides[~is_val], cfg)
        ds_va = UnifiedPatchDataset(X[is_val], coords[is_val], y[is_val],
                                    slides[is_val], cfg)
        tr, va = make_loaders(ds_tr, ds_va, batch_size=args.batch_size)

        model = HabitatNet(X.shape[1], cfg)
        hist, best, _ = train_unified(model, tr, va, epochs=args.epochs)
        model.load_state_dict(best)
        histories[f"{mode}_s{seed}"] = hist

        model.eval()
        p, t = [], []
        with torch.no_grad():
            for c, ctx, yy in va:
                p.append(model(c, ctx).argmax(1).numpy())
                t.append(yy.numpy())
        pred, true = np.concatenate(p), np.concatenate(t)
        m = evaluate(true, pred, n_bootstrap=300)
        m["mode"] = mode
        m["seed"] = seed
        m["method"] = "Method 1 (CANVAS)" if mode == "none" else f"Method 2 ({mode})"
        m["n_params"] = sum(pp.numel() for pp in model.parameters())
        m["train_seconds"] = round(time.time() - t0, 1)
        rows.append(m)

        # per-class precision/recall/F1: accuracy hides a collapsed minority
        # habitat. Also persisted (not just logged) so a report table can be
        # built without re-running training.
        from sklearn.metrics import precision_recall_fscore_support
        labels = np.arange(int(y.max()) + 1)
        prec, rec, f1c, support = precision_recall_fscore_support(
            true, pred, labels=labels, average=None, zero_division=0)
        for cls, p_, r_, f_, s_ in zip(labels, prec, rec, f1c, support):
            per_class_rows.append({"mode": mode, "seed": seed, "class": int(cls),
                                   "precision": float(p_), "recall": float(r_),
                                   "f1": float(f_), "support": int(s_)})
        dead = np.flatnonzero(rec < 0.10)
        log.info("seed %d %-7s macro_f1 %.3f  min_recall %.3f%s", seed, mode,
                 float(m.loc[m.metric == "macro_f1", "value"].iloc[0]), rec.min(),
                 f"  COLLAPSED classes: {dead.tolist()}" if len(dead) else "")

        # row-normalised confusion matrix, long form (mode, seed, true, pred, count)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true, pred, labels=labels)
        for i in labels:
            for j in labels:
                if cm[i, j] > 0:
                    confusion_rows.append({"mode": mode, "seed": int(seed),
                                           "true_class": int(i), "pred_class": int(j),
                                           "count": int(cm[i, j])})

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "final_benchmark.csv", index=False)
    pd.concat({k: v for k, v in histories.items()}).to_csv(out / "training_curves.csv")
    pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "confusion_matrices.csv", index=False)

    # mean +/- SD across seeds, and a paired comparison against modes[0]
    agg = (res.groupby(["mode", "metric"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    piv = agg.pivot(index="mode", columns="metric", values=["mean", "std"])
    meta = res.groupby("mode")[["n_params", "train_seconds"]].first()
    log.info("\n%s\n\n%s", piv.round(3).to_string(), meta.to_string())

    if len(args.seeds) >= 3:
        from scipy.stats import wilcoxon
        base = args.modes[0]
        f1 = res[res.metric == "macro_f1"].pivot(index="seed", columns="mode",
                                                 values="value")
        log.info("\nPaired Wilcoxon vs '%s' across %d seeds:", base, len(args.seeds))
        for mode in args.modes[1:]:
            d = f1[mode] - f1[base]
            try:
                _, pv = wilcoxon(f1[mode], f1[base])
            except ValueError:
                pv = np.nan
            log.info("  %-7s delta %+.4f +/- %.4f   p=%.4f", mode,
                     d.mean(), d.std(), pv)
    else:
        log.warning("\nONLY %d SEED(S). A macro-F1 gap under ~0.02 is not "
                    "distinguishable from split luck. Rerun with --seeds 1 2 3 4 5 "
                    "before reporting any comparison.", len(args.seeds))

    (out / "benchmark_config.json").write_text(json.dumps(vars(args), indent=2))
    log.info(
        "\nREADING THIS TABLE\n"
        "  mode=none IS CANVAS. Every other row differs only in context.\n"
        "  Report macro-F1 and kappa with bootstrap CIs, not accuracy alone:\n"
        "  background dominates and accuracy hides per-class collapse.\n"
        "  If grid3d beats grid2d, check it is not just parameter count. Rerun\n"
        "  grid2d with grid_channels raised to match, then compare again."
    )


if __name__ == "__main__":
    main()
