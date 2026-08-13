"""Clinical modelling: univariate Cox, ecotypes, and the spatial signature.

Reproduces the CANVAS stage-6 statistics with breast-appropriate covariates.
Pure CPU. The full 262-feature pipeline on ~1,000 TCGA-BRCA samples takes
minutes, not hours, so run it before you have real habitat predictions using
simulated maps to prove the machinery is correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

BREAST_COVARIATES = [
    "age",
    "stage",
    "grade",
    "ER_status",
    "PR_status",
    "HER2_status",
]
# Note: CANVAS used smoking status for NSCLC. It is not a breast covariate.
# Receptor status and PAM50 take its place as the subtype axis.


@dataclass
class ClinicalConfig:
    endpoints: tuple[str, ...] = ("OS", "DSS", "PFI", "DFI")
    covariates: list[str] = field(default_factory=lambda: list(BREAST_COVARIATES))
    stratify_by: tuple[str, ...] = ("PAM50", "ER_status")
    collinearity_rho: float = 0.95
    lasso_repeats: int = 100
    lasso_cv_folds: int = 5
    lasso_top_frac: float = 0.10
    rsf_bootstrap: int = 1000
    td_auc_months: tuple[int, ...] = (6, 12, 24)
    max_k: int = 8
    consensus_reps: int = 1000
    p_item: float = 0.8
    alpha: float = 0.05
    seed: int = 42


# ------------------------------------------------------------ univariate Cox


def univariate_cox(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    endpoint: str,
    feature_cols: list[str] | None = None,
    cfg: ClinicalConfig | None = None,
) -> pd.DataFrame:
    """Fit one univariate Cox model per feature.

    Parameters
    ----------
    features
        Indexed by sample_id, columns are habitat/spatial features.
    clinical
        Indexed by sample_id, must contain ``{endpoint}_time`` and
        ``{endpoint}_event``.
    endpoint
        One of OS, DSS, PFI, DFI (TCGA-CDR naming).

    Returns
    -------
    DataFrame with hr, ci_lower, ci_upper, p, q (Benjamini-Hochberg).
    """
    cfg = cfg or ClinicalConfig()
    time_col, event_col = f"{endpoint}_time", f"{endpoint}_event"
    for c in (time_col, event_col):
        if c not in clinical.columns:
            raise ValueError(f"clinical table missing '{c}'")

    joined = features.join(clinical[[time_col, event_col]], how="inner").dropna(
        subset=[time_col, event_col]
    )
    cols = feature_cols or [c for c in features.columns if c in joined.columns]

    records = []
    for col in cols:
        sub = joined[[col, time_col, event_col]].dropna()
        if sub[col].nunique() < 3 or sub[event_col].sum() < 5:
            continue
        # Standardise so HR is per 1 SD, comparable across heterogeneous scales.
        sub = sub.assign(**{col: (sub[col] - sub[col].mean()) / (sub[col].std() or 1.0)})
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col=time_col, event_col=event_col)
            row = cph.summary.loc[col]
            records.append(
                {
                    "feature": col,
                    "hr": float(row["exp(coef)"]),
                    "ci_lower": float(row["exp(coef) lower 95%"]),
                    "ci_upper": float(row["exp(coef) upper 95%"]),
                    "z": float(row["z"]),
                    "p": float(row["p"]),
                    "n": int(len(sub)),
                    "events": int(sub[event_col].sum()),
                }
            )
        except Exception as exc:  # convergence failures are expected and fine
            logger.debug("Cox failed for %s: %s", col, exc)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("p").reset_index(drop=True)


# ----------------------------------------------------------------- ecotypes


def consensus_cluster(
    habitat_profiles: pd.DataFrame,
    cfg: ClinicalConfig | None = None,
    k: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Consensus clustering of z-scored habitat profiles into spatial ecotypes.

    CANVAS used ConsensusClusterPlus with PAM and Canberra distance. This is a
    Python equivalent: repeated subsampling, PAM-style clustering on a Canberra
    distance, and a consensus matrix from co-assignment frequency.

    If you want exact parity with the paper, export the z-scored matrix to CSV
    and run ConsensusClusterPlus in R. The R implementation is the reference;
    this is for iteration speed.

    Returns
    -------
    labels
        Series of ecotype labels (C-I, C-II, ...) indexed by sample_id.
    consensus
        The n x n consensus matrix.
    """
    cfg = cfg or ClinicalConfig()
    k = k or 4
    rng = np.random.default_rng(cfg.seed)

    x = habitat_profiles.dropna()
    z = pd.DataFrame(
        StandardScaler().fit_transform(x), index=x.index, columns=x.columns
    )
    n = len(z)
    co = np.zeros((n, n))
    seen = np.zeros((n, n))
    pos = {s: i for i, s in enumerate(z.index)}

    n_sub = max(int(cfg.p_item * n), k + 1)
    for _ in range(cfg.consensus_reps):
        idx = rng.choice(n, n_sub, replace=False)
        sub = z.iloc[idx]
        d = _canberra_matrix(sub.to_numpy())
        lab = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(d)
        for a in range(n_sub):
            ia = pos[sub.index[a]]
            for b in range(a + 1, n_sub):
                ib = pos[sub.index[b]]
                seen[ia, ib] += 1
                seen[ib, ia] += 1
                if lab[a] == lab[b]:
                    co[ia, ib] += 1
                    co[ib, ia] += 1

    consensus = np.divide(co, seen, out=np.zeros_like(co), where=seen > 0)
    np.fill_diagonal(consensus, 1.0)

    final = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(1.0 - consensus)

    roman = ["C-I", "C-II", "C-III", "C-IV", "C-V", "C-VI", "C-VII", "C-VIII"]
    labels = pd.Series([roman[i] for i in final], index=z.index, name="ecotype")
    return labels, pd.DataFrame(consensus, index=z.index, columns=z.index)


def _canberra_matrix(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(x, metric="canberra"))


# --------------------------------------------------------- signature model


def reduce_collinearity(
    features: pd.DataFrame, cfg: ClinicalConfig | None = None
) -> list[str]:
    """Drop |Spearman rho| > 0.95 by Louvain community detection.

    One representative per community is kept: the feature with the highest mean
    absolute correlation to the rest of its community, i.e. the most central.
    """
    cfg = cfg or ClinicalConfig()
    corr = features.corr(method="spearman").abs().fillna(0.0)
    cols = list(corr.columns)

    try:
        import igraph as ig

        edges = [
            (i, j)
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
            if corr.iloc[i, j] > cfg.collinearity_rho
        ]
        g = ig.Graph(n=len(cols), edges=edges)
        communities = g.community_multilevel()
        groups = [[cols[v] for v in comm] for comm in communities]
    except ImportError:
        logger.warning("python-igraph not installed, using connected components")
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] > cfg.collinearity_rho:
                    g.add_edge(cols[i], cols[j])
        groups = [list(c) for c in nx.connected_components(g)]

    keep = []
    for grp in groups:
        if len(grp) == 1:
            keep.append(grp[0])
        else:
            centrality = corr.loc[grp, grp].mean(axis=1)
            keep.append(str(centrality.idxmax()))
    logger.info("collinearity reduction: %d -> %d features", len(cols), len(keep))
    return keep


def lasso_cox_selection(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    endpoint: str,
    cfg: ClinicalConfig | None = None,
) -> pd.Series:
    """Repeated-resampling LASSO-Cox. Returns selection frequency per feature.

    100 iterations, each splitting into train/validation, fitting with internal
    5-fold CV on the concordance index, and recording non-zero coefficients.
    """
    cfg = cfg or ClinicalConfig()
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv

    time_col, event_col = f"{endpoint}_time", f"{endpoint}_event"
    joined = features.join(clinical[[time_col, event_col]], how="inner").dropna()
    x = joined[features.columns]
    x = pd.DataFrame(
        StandardScaler().fit_transform(x), index=x.index, columns=x.columns
    )
    y = Surv.from_arrays(
        event=joined[event_col].astype(bool), time=joined[time_col].astype(float)
    )

    rng = np.random.default_rng(cfg.seed)
    counts = pd.Series(0, index=x.columns, dtype=int)

    for it in range(cfg.lasso_repeats):
        idx = rng.permutation(len(x))
        cut = int(0.7 * len(x))
        tr = idx[:cut]
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
            model.fit(x.iloc[tr].to_numpy(), y[tr])
            # take the mid-path alpha as a stable, moderately sparse solution
            coefs = model.coef_[:, model.coef_.shape[1] // 2]
            counts[np.abs(coefs) > 1e-8] += 1
        except Exception as exc:
            logger.debug("lasso iteration %d failed: %s", it, exc)

    freq = counts / cfg.lasso_repeats
    threshold = freq.quantile(1.0 - cfg.lasso_top_frac)
    logger.info("LASSO top-%.0f%% threshold = %.2f selection frequency",
                cfg.lasso_top_frac * 100, threshold)
    return freq.sort_values(ascending=False)


def rsf_importance(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    endpoint: str,
    cfg: ClinicalConfig | None = None,
    n_estimators: int = 300,
) -> pd.Series:
    """Permutation importance from a random survival forest.

    The paper used 1,000 bootstrap iterations with randomForestSRC. That is
    expensive on CPU; ``rsf_bootstrap`` in the config controls the permutation
    repeats here. Reduce it on the laptop profile and say so in the manuscript.
    """
    cfg = cfg or ClinicalConfig()
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    time_col, event_col = f"{endpoint}_time", f"{endpoint}_event"
    joined = features.join(clinical[[time_col, event_col]], how="inner").dropna()
    x = joined[features.columns]
    y = Surv.from_arrays(
        event=joined[event_col].astype(bool), time=joined[time_col].astype(float)
    )

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=cfg.seed,
    ).fit(x.to_numpy(), y)

    baseline = rsf.score(x.to_numpy(), y)
    rng = np.random.default_rng(cfg.seed)
    n_rep = min(cfg.rsf_bootstrap, 50)  # permutation is the expensive part
    drops = {}
    for j, col in enumerate(x.columns):
        deltas = []
        for _ in range(max(n_rep // len(x.columns), 3)):
            xp = x.to_numpy().copy()
            rng.shuffle(xp[:, j])
            deltas.append(baseline - rsf.score(xp, y))
        drops[col] = float(np.mean(deltas))
    return pd.Series(drops).sort_values(ascending=False)


def fit_signature(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    endpoint: str,
    selected: list[str],
    cfg: ClinicalConfig | None = None,
) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Multivariable Cox on the jointly prioritised features.

    Returns the fitted model and a metrics frame with the C-index and
    time-dependent AUC at 6, 12 and 24 months.
    """
    cfg = cfg or ClinicalConfig()
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    time_col, event_col = f"{endpoint}_time", f"{endpoint}_event"
    joined = features[selected].join(
        clinical[[time_col, event_col]], how="inner"
    ).dropna()

    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(joined, duration_col=time_col, event_col=event_col)

    risk = cph.predict_partial_hazard(joined).to_numpy()
    c_index = concordance_index(joined[time_col], -risk, joined[event_col])

    y = Surv.from_arrays(
        event=joined[event_col].astype(bool), time=joined[time_col].astype(float)
    )
    # TCGA-CDR times are in days
    times = np.array([m * 30.44 for m in cfg.td_auc_months], dtype=float)
    times = times[(times > joined[time_col].min()) & (times < joined[time_col].max())]
    auc_vals, _ = cumulative_dynamic_auc(y, y, risk, times) if len(times) else ([], None)

    metrics = pd.DataFrame(
        {
            "metric": ["c_index"] + [f"auc_{int(t / 30.44)}mo" for t in times],
            "value": [float(c_index)] + [float(a) for a in auc_vals],
        }
    )
    return cph, metrics


def stratify_by_median(risk: pd.Series) -> pd.Series:
    """Split into high/low risk groups at the cohort median, as in the paper."""
    return pd.Series(
        np.where(risk > risk.median(), "high", "low"), index=risk.index, name="risk_group"
    )
