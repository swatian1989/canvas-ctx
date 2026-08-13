"""Evaluation with bootstrap confidence intervals. [PAPER] n_bootstrap = 1000."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def evaluate(y_true, y_pred, n_bootstrap: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Accuracy, macro-F1 and Cohen's kappa with bootstrap 95% CIs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = {
        "accuracy": accuracy_score,
        "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score,
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in metrics.items():
        point = fn(y_true, y_pred)
        boots = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            boots.append(fn(y_true[idx], y_pred[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        rows.append({"metric": name, "value": float(point),
                     "ci_lower": float(lo), "ci_upper": float(hi)})
    return pd.DataFrame(rows)
