#!/usr/bin/env python
"""Method 1 vs Method 2 benchmark: the k-sweep ablation.

k = 0 is CANVAS exactly. k > 0 adds spatial context and nothing else. Same
encoder, labels, splits, loss and head throughout, so any difference is
attributable to context.

    python scripts/run_method2_benchmark.py \
        --embeddings data/interim/patch_embeddings.parquet \
        --k-values 0 4 8 16

Embeddings parquet needs: slide_id, x_um, y_um, label, and emb_0..emb_{D-1}.
Use --simulate to test the machinery with synthetic data first.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canvas_brca.method2_context.context_model import (  # noqa: E402
    ContextConfig, ContextHabitatNet,
)
from canvas_brca.method2_context.dataset import (  # noqa: E402
    ContextPatchDataset, make_loaders, train_context_model,
)
from canvas_brca.stage3_model.evaluate import evaluate  # noqa: E402


def simulate(n_slides=12, per_slide=900, dim=64, n_classes=11, seed=0):
    """Synthetic patches where habitat depends on a LOCAL REGION, not the patch.

    Each slide is tiled into regions with a habitat identity. A patch embedding
    carries a weak, noisy signal of its own habitat. The point is that context
    should help, because the region label is only weakly recoverable from one
    patch alone. If the k-sweep shows no gain on this fixture, the context
    machinery is broken, not the biology.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_classes, dim))
    rows = []
    for s in range(n_slides):
        side = int(np.sqrt(per_slide))
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        x = gx.ravel() * 224 * 0.5
        y = gy.ravel() * 224 * 0.5
        region = ((gx // 5) * 7 + (gy // 5) * 11).ravel() % n_classes
        emb = centroids[region] + rng.normal(0, 3.0, size=(len(region), dim))
        rows.append(pd.DataFrame({
            "slide_id": f"S{s:02d}", "x_um": x, "y_um": y, "label": region,
            **{f"emb_{i}": emb[:, i] for i in range(dim)},
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--k-values", type=int, nargs="+", default=[0, 4, 8, 16])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--radius-um", type=float, default=300.0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/method_benchmark.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("bench")

    df = simulate(seed=args.seed) if args.simulate else pd.read_parquet(args.embeddings)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")],
                      key=lambda s: int(s.split("_")[1]))
    X = df[emb_cols].to_numpy(np.float32)
    coords = df[["x_um", "y_um"]].to_numpy(np.float64)
    y = df["label"].to_numpy(np.int64)
    slides = df["slide_id"].to_numpy()

    # SAMPLE-LEVEL split. [PAPER] Never split patches.
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(slides)
    rng.shuffle(uniq)
    n_val = max(1, int(args.val_frac * len(uniq)))
    val_slides = set(uniq[:n_val])
    is_val = np.array([s in val_slides for s in slides])
    log.info("split: %d train / %d val slides, %d / %d patches",
             len(uniq) - n_val, n_val, (~is_val).sum(), is_val.sum())

    results = []
    for k in args.k_values:
        cfg = ContextConfig(k_neighbours=k, radius_um=args.radius_um,
                            n_classes=int(y.max()) + 1, seed=args.seed)
        ds_tr = ContextPatchDataset(X[~is_val], coords[~is_val], y[~is_val],
                                    slides[~is_val], cfg)
        ds_va = ContextPatchDataset(X[is_val], coords[is_val], y[is_val],
                                    slides[is_val], cfg)
        tr, va = make_loaders(ds_tr, ds_va, batch_size=64)

        model = ContextHabitatNet(X.shape[1], cfg)
        res = train_context_model(model, tr, va, epochs=args.epochs)
        model.load_state_dict(res.best_state)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for centre, nbr, mask, dist, yy in va:
                preds.append(model(centre, nbr, mask, dist).argmax(1).numpy())
                trues.append(yy.numpy())
        m = evaluate(np.concatenate(trues), np.concatenate(preds), n_bootstrap=500)
        m["k"] = k
        m["method"] = "CANVAS (Method 1)" if k == 0 else f"CANVAS-CTX k={k}"
        m["n_params"] = sum(p.numel() for p in model.parameters())
        results.append(m)
        log.info("k=%d  macro_f1=%.3f", k,
                 float(m.loc[m.metric == "macro_f1", "value"].iloc[0]))

    out = pd.concat(results, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    piv = out.pivot_table(index=["k", "method"], columns="metric", values="value")
    log.info("\n%s", piv.round(3).to_string())
    log.info("\nk=0 IS CANVAS. Any gain at k>0 is spatial context alone.")


if __name__ == "__main__":
    main()
