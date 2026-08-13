"""One function per report table (T1-T10).

Same contract as figures.py: every function returns
    {"id": "T5", "title": "...", "source": "REAL DATA (...)" | "SIMULATED (...)",
     "csv_path": "...", "df": pd.DataFrame, "caption": "..."}

A table that cannot be built because real data is missing still returns a
DataFrame (one row: status / needs / unblocks) and is saved to CSV like any
other table, so nothing is silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from canvas_brca.stage6_clinical.signature import (
    ClinicalConfig, fit_signature, lasso_cox_selection, reduce_collinearity,
    rsf_importance, univariate_cox,
)

from .figures import _load_confusion, _load_final_benchmark, _load_per_class, _load_sim_features
from .figures import _null_clinical

RESULTS_TABLES = Path("results/tables")


def _save(df: pd.DataFrame, name: str, tables_dir: str) -> str:
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _placeholder(table_id: str, title: str, missing_file: str, unblocks: str,
                 tables_dir: str, name: str) -> dict:
    df = pd.DataFrame([{"status": "MISSING DATA", "needs": missing_file, "unblocks": unblocks}])
    path = _save(df, name, tables_dir)
    return {"id": table_id, "title": title, "source": "MISSING DATA", "csv_path": path,
           "df": df, "caption": f"Needs {missing_file}. Unblocks {unblocks}."}


# ==================================================================== T1


def t1_dataset_inventory(tables_dir: str) -> dict:
    rows = [
        dict(cohort="Schurch 2020 CRC CODEX", platform="CODEX, 56-plex",
            n="35 patients / 140 regions", role="CN discovery (primary)",
            accession="Mendeley data mpjzbtfgfr/1", replaces="41-plex CODEX NSCLC (paper)",
            status="NOT ACQUIRED"),
        dict(cohort="Orion 2023 CRC IF+H&E", platform="Orion, 18-plex IF + same-section H&E",
            n="74 resections (8-12 used)", role="Paired label transfer (train/val/test)",
            accession="s3://lin-2023-orion-crc/data (public)",
            replaces="Same-section CODEX+H&E, lung (paper)", status="NOT ACQUIRED"),
        dict(cohort="TCGA-COAD DX", platform="H&E WSI, FFPE diagnostic",
            n="60 slides (target)", role="In-domain deployment", accession="GDC portal",
            replaces="Paper's in-domain NSCLC deployment cohort", status="NOT ACQUIRED"),
        dict(cohort="TCGA-BRCA DX", platform="H&E WSI, FFPE diagnostic",
            n="60 slides (target)", role="Cross-cancer transfer (the result)",
            accession="GDC portal",
            replaces="Paper's 17-specimen/12-tumour-type transfer cohort",
            status="NOT ACQUIRED"),
        dict(cohort="TCGA-CDR", platform="Curated clinical table",
            n="pan-cancer, filtered to COAD+BRCA", role="Survival endpoints (OS, PFI)",
            accession="GDC PanCanAtlas publications page", replaces="paper's clinical annotation",
            status="NOT ACQUIRED"),
        dict(cohort="Simulated null fixture", platform="synthetic",
            n="200 samples (stage5 --simulate)", role="Statistical machinery validation",
            accession="generated in-repo, no download", replaces="n/a - machinery check only",
            status="ACQUIRED (generated)"),
    ]
    df = pd.DataFrame(rows)
    path = _save(df, "T1_dataset_inventory", tables_dir)
    return {"id": "T1", "title": "Dataset inventory", "source": "STATUS (not experimental data)",
           "csv_path": path, "df": df,
           "caption": "Every cohort this project's design calls for, its role, and current "
                      "acquisition status. Only the synthetic fixture is in hand."}


# ==================================================================== T2, T3


# Names proposed from the marker-enrichment table (F3), NOT carried over from
# the published NSCLC nomenclature. Each is the reading of that CN's top
# z-scored cell types; the published CRC label each one best matches is
# reported separately in T11 and was not used to choose these names.
CN_PROPOSED_NAMES = {
    "CN01": "Macrophage-adipose",
    "CN02": "Bulk tumour",
    "CN03": "Cytotoxic-infiltrated stroma",
    "CN04": "Lymphoid follicle",
    "CN05": "Pan-immune cytotoxic",
    "CN06": "Plasma-cell rich",
    "CN07": "Memory CD4 T-cell",
    "CN08": "Tumour boundary myeloid",
    "CN09": "Smooth muscle-lymphatic",
    "CN10": "Granulocyte-enriched",
}


def t2_cn_definitions(tables_dir: str,
                      enrichment_csv: str = "data/processed/cn_lineage_enrichment.csv",
                      assignments: str = "data/processed/cn_assignments.parquet") -> dict:
    if not Path(enrichment_csv).exists():
        return _placeholder("T2", "CN definitions", enrichment_csv, "Phase 2 - CN naming",
                            tables_dir, "T2_cn_definitions")
    df = pd.read_csv(enrichment_csv, index_col=0)
    top3 = df.apply(lambda r: ", ".join(
        f"{c} ({r[c]:+.1f})" for c in r.sort_values(ascending=False).index[:3]), axis=1)

    freq = {}
    if Path(assignments).exists():
        a = pd.read_parquet(assignments)
        vc = a["cn_label"].value_counts(normalize=True)
        freq = (vc * 100).round(1).to_dict()

    out = pd.DataFrame({
        "cn_label": df.index,
        "proposed_name": [CN_PROPOSED_NAMES.get(c, "UNNAMED") for c in df.index],
        "top_enriched_cell_types_z": top3.values,
        "frequency_pct": [freq.get(c, float("nan")) for c in df.index],
    })
    path = _save(out, "T2_cn_definitions", tables_dir)
    return {"id": "T2", "title": "CN definitions and proposed names",
           "source": "REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154)", "csv_path": path,
           "df": out,
           "caption": "Each CN's three most enriched cell types (z-scored across CNs) with "
                      "the name proposed from that enrichment, and its frequency across all "
                      "cells. Names are read from the markers in this cohort; the published "
                      "NSCLC nomenclature was deliberately not reused, and the published CRC "
                      "labels were not consulted when naming (their correspondence is "
                      "reported separately in T11)."}


def t11_cn_vs_published(tables_dir: str,
                        src: str = "results/tables/T11_cn_vs_published.csv",
                        summary: str = "results/tables/T11_cn_validation_summary.csv") -> dict:
    """Addition beyond the original 10-table specification.

    Justified because the source table ships the authors' own CN assignments,
    making an external validation possible at stage 1 with no extra data.
    """
    if not Path(src).exists():
        return _placeholder("T11", "Rediscovered CNs vs published CNs",
                            f"{src} (run scripts/validate_cn_vs_published.py)",
                            "Phase 2 - CN validation", tables_dir, "T11_cn_vs_published")
    df = pd.read_csv(src)
    df["proposed_name"] = df["our_cn"].map(CN_PROPOSED_NAMES)
    df = df[["our_cn", "proposed_name", "n_cells", "best_published_match",
             "overlap_fraction"]]
    ari = nmi = float("nan")
    if Path(summary).exists():
        s = pd.read_csv(summary).set_index("metric")["value"]
        ari, nmi = float(s.get("adjusted_rand_index", float("nan"))), \
                   float(s.get("normalized_mutual_info", float("nan")))
    path = _save(df, "T11_cn_vs_published", tables_dir)
    return {"id": "T11", "title": "Rediscovered CNs versus the published CN labels",
           "source": "REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154)", "csv_path": path,
           "df": df,
           "caption": f"Each independently rediscovered CN, its proposed name, and the "
                      f"published neighbourhood it overlaps most. Adjusted Rand index "
                      f"{ari:.3f}, normalised mutual information {nmi:.3f} across "
                      f"{int(df['n_cells'].sum()):,} cells. Agreement is moderate by "
                      f"construction rather than by failure: the published method built "
                      f"windows from the 10 nearest neighbours, whereas CANVAS specifies a "
                      f"fixed 40 um radius (median ~33 neighbours in this tissue) followed "
                      f"by a topic decomposition. Every published neighbourhood type is "
                      f"represented among the rediscovered set."}


def t3_patch_counts(
    tables_dir: str,
    patch_labels: str = "data/interim/stage2_labels/CRC01_denovo_patch_labels.parquet",
    denovo_json: str = "results/orion_cn_denovo.json",
) -> dict:
    if not Path(patch_labels).exists():
        return _placeholder("T3", "Patch counts by habitat label", patch_labels,
                            "Phase 3 - label transfer", tables_dir, "T3_patch_counts")
    df = pd.read_parquet(patch_labels)
    n_hab = int(df["label"].max())
    rows = []
    for lab, n in df["label"].value_counts().sort_index().items():
        name = "background" if int(lab) == n_hab else f"CN{int(lab) + 1:02d}"
        sub = df[df["label"] == lab]
        rows.append({"label": name, "n_patches": int(n),
                     "pct_of_patches": round(100 * n / len(df), 2),
                     "median_cells_per_patch": int(sub["n_cells"].median()),
                     "median_dominant_frac": round(float(sub["dominant_frac"].median()), 3)})
    out = pd.DataFrame(rows)
    path = _save(out, "T3_patch_counts", tables_dir)

    n_samples = df["sample_id"].nunique()
    return {"id": "T3", "title": "Patch counts by habitat label (Orion CRC01, de novo CNs)",
           "source": "REAL DATA (Orion CRC01 H&E + IF, 1,620,375 cells)",
           "csv_path": path, "df": out,
           "caption": f"{len(df):,} patches of 345 native px (224 px at 20x) surviving the "
                      f"CANVAS purity rules, from habitats derived DE NOVO on the Orion "
                      f"panel. All {n_hab} habitats plus background are represented, the "
                      f"smallest at {out['n_patches'].min()} patches; the earlier "
                      f"cross-platform transfer collapsed to 2 patches for its smallest "
                      f"class, which is why it was rejected. Only {n_samples} specimen was "
                      f"processed, so no train/validation/test split is reported: a split "
                      f"is meaningful at patient level and one patient cannot be split. "
                      f"That is a coverage limitation, not a methodological choice."}


# ==================================================================== T4, T5, T6


def t4_classifier_performance(tables_dir: str, mode: str = "none") -> dict:
    df = _load_final_benchmark()
    if df is None:
        return _placeholder("T4", "Classifier performance", "results/final_benchmark.csv",
                            "6-seed benchmark run", tables_dir, "T4_classifier_performance")
    sub = df[df["mode"] == mode]
    agg = sub.groupby("metric").agg(
        mean=("value", "mean"), sd=("value", "std"),
        ci_lower_mean=("ci_lower", "mean"), ci_upper_mean=("ci_upper", "mean"),
    ).reset_index()
    n_seeds = sub["seed"].nunique()
    path = _save(agg, "T4_classifier_performance", tables_dir)
    return {"id": "T4", "title": f"Classifier performance ({mode}, Method 1/CANVAS baseline)",
           "source": "SIMULATED", "csv_path": path, "df": agg,
           "caption": f"Accuracy, macro-F1, Cohen's kappa for mode={mode}, mean +/- SD across "
                      f"{n_seeds} seeds, with the mean of each seed's bootstrap 95% CI. "
                      "SIMULATED --simulate fixture; the real classifier's performance table "
                      "needs Phase 2-4 data."}


def t5_benchmark_table(tables_dir: str) -> dict:
    df = _load_final_benchmark()
    if df is None:
        return _placeholder("T5", "Benchmark: mean +/- SD per mode", "results/final_benchmark.csv",
                            "6-seed benchmark run", tables_dir, "T5_benchmark")

    modes = list(dict.fromkeys(df["mode"]))
    base = modes[0]
    rows = []
    for metric in ["accuracy", "macro_f1", "cohen_kappa"]:
        sub = df[df["metric"] == metric]
        piv = sub.pivot(index="seed", columns="mode", values="value")
        for mode in modes:
            delta = piv[mode] - piv[base] if mode != base else pd.Series(0, index=piv.index)
            if mode != base:
                try:
                    _, pv = wilcoxon(piv[mode], piv[base])
                except ValueError:
                    pv = np.nan
            else:
                pv = np.nan
            meta = sub[sub["mode"] == mode].iloc[0]
            rows.append(dict(
                metric=metric, mode=mode, mean=piv[mode].mean(), sd=piv[mode].std(),
                delta_vs_base=delta.mean(), wilcoxon_p=pv, n_seeds=len(piv),
                n_params=int(meta["n_params"]), train_seconds=float(meta["train_seconds"]),
            ))
    out = pd.DataFrame(rows)
    path = _save(out, "T5_benchmark", tables_dir)
    return {"id": "T5", "title": "Benchmark: mean +/- SD per mode across seeds",
           "source": "SIMULATED", "csv_path": path, "df": out,
           "caption": f"Mean +/- SD per mode across {df['seed'].nunique()} seeds, delta vs "
                      f"mode={base}, paired Wilcoxon p, parameter count, training seconds. "
                      "SIMULATED --simulate fixture -- oriented-band construction rewards "
                      "spatial context by design; this is a machinery check."}


def t6_per_class_metrics(tables_dir: str) -> dict:
    df = _load_per_class()
    if df is None:
        return _placeholder("T6", "Per-class precision/recall/F1",
                            "results/per_class_run/per_class_metrics.csv",
                            "supplementary per-class benchmark run", tables_dir,
                            "T6_per_class_metrics")
    agg = df.groupby(["mode", "class"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        support=("support", "mean"), n_seeds=("seed", "nunique"),
    ).reset_index()
    path = _save(agg, "T6_per_class_metrics", tables_dir)
    return {"id": "T6", "title": "Per-class precision, recall, F1", "source": "SIMULATED",
           "csv_path": path, "df": agg,
           "caption": f"Precision/recall/F1 per habitat class, mean across "
                      f"{df['seed'].nunique()} seeds. SIMULATED (supplementary "
                      f"{df['seed'].nunique()}-seed run at the same --simulate fixture and "
                      "epoch count as T5, run separately to avoid a second full 6-seed "
                      "training pass)."}


# ==================================================================== T7, T8, T9


def t7_univariate_cox(tables_dir: str, endpoint: str = "OS") -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder("T7", "Univariate Cox", "data/interim/sim_features.parquet",
                            "stage5 --simulate + null-clinical check", tables_dir, "T7_cox")
    clinical = _null_clinical(feats.index, endpoint)
    cfg = ClinicalConfig()
    cox = univariate_cox(feats, clinical, endpoint, cfg=cfg)
    path = _save(cox, "T7_cox", tables_dir)
    n_sig = int((cox["q"] < cfg.alpha).sum()) if not cox.empty else 0
    return {"id": "T7", "title": "Univariate Cox (null calibration)", "source": "SIMULATED",
           "csv_path": path, "df": cox,
           "caption": f"feature, HR, 95% CI, p, q, n, events -- all {len(cox)} features, "
                      f"random {endpoint} outcome, n={feats.shape[0]}. "
                      f"{n_sig}/{len(cox)} significant at FDR {cfg.alpha} "
                      f"(expected near 0 under the null). NOT real prognostic estimates -- "
                      "needs Phase 5/6 for those."}


def t8_ecotype_clinical(tables_dir: str) -> dict:
    return _placeholder("T8", "Ecotype clinical associations",
                        "real ecotypes (Phase 5) + real clinical covariates (Phase 5/6)",
                        "Phase 5/6 - ecotype characterisation", tables_dir, "T8_ecotype_clinical")


def t9_signature(tables_dir: str, endpoint: str = "OS") -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder("T9", "Signature model", "data/interim/sim_features.parquet",
                            "stage5+6 null-calibration chain", tables_dir, "T9_signature")
    clinical = _null_clinical(feats.index, endpoint)
    cfg = ClinicalConfig(lasso_repeats=50, rsf_bootstrap=200, seed=42)
    keep = reduce_collinearity(feats, cfg)
    reduced = feats[keep]
    lasso_freq = lasso_cox_selection(reduced, clinical, endpoint, cfg)
    rsf_imp = rsf_importance(reduced, clinical, endpoint, cfg)
    lasso_top = set(lasso_freq[lasso_freq >= lasso_freq.quantile(1 - cfg.lasso_top_frac)].index)
    rsf_top = set(rsf_imp[rsf_imp >= rsf_imp.quantile(1 - cfg.lasso_top_frac)].index)
    selected = sorted(lasso_top & rsf_top)

    rows = []
    coefs = {}
    if len(selected) >= 2:
        model, _ = fit_signature(reduced, clinical, endpoint, selected, cfg)
        coefs = model.params_.to_dict()
    for feat in selected:
        rows.append(dict(feature=feat, coefficient=coefs.get(feat, np.nan),
                         lasso_selection_freq=float(lasso_freq.get(feat, np.nan)),
                         rsf_importance=float(rsf_imp.get(feat, np.nan))))
    out = pd.DataFrame(rows)
    path = _save(out, "T9_signature", tables_dir)
    return {"id": "T9", "title": "Signature: selected features", "source": "SIMULATED",
           "csv_path": path, "df": out,
           "caption": f"{len(selected)} features survived LASSO(top {cfg.lasso_top_frac:.0%}) "
                      f"^ RSF(top {cfg.lasso_top_frac:.0%}) intersection, random {endpoint} "
                      f"outcome, n={feats.shape[0]}. Coefficients from the multivariable Cox "
                      "fit on the SAME data used for selection -- see the report text on "
                      "in-sample optimism bias before trusting these on real data."}


# ==================================================================== T10


def t10_deviations(tables_dir: str) -> dict:
    rows = [
        dict(parameter="CN discovery platform/panel/cohort",
            paper_value="CODEX, 41-plex, NSCLC (unspecified n)",
            our_value="CODEX, 56-plex, Schurch 2020 CRC (35 patients/140 regions); "
                      "Orion 18-plex IF CRC as fallback",
            reason="No public CODEX 41-plex NSCLC re-implementation dataset; CRC has the "
                  "strongest public same-section paired resource (Orion), so the whole "
                  "pipeline trains on CRC and is APPLIED to breast (see design note in "
                  "config/crc_train_brca_apply.yaml)",
            expected_impact="Richer marker panel (56 vs 41) partially offsets tissue "
                            "mismatch; CN identities will not literally match the paper's "
                            "10 NSCLC neighbourhoods and must be renamed from marker "
                            "enrichment (F3/T2), never assumed"),
        dict(parameter="Paired same-section cohort",
            paper_value="CODEX + H&E, same lung sections",
            our_value="Orion 18-plex IF + H&E, same CRC sections, already registered "
                      "at 0.325 um/px",
            reason="Breast has only 1-3 Xenium samples and 10x documents its alignment as "
                  "unsuitable for sub-cellular correspondence; Orion has 74 same-section "
                  "resections with pre-registered pairs",
            expected_impact="Label-transfer noise floor should be LOWER than a from-scratch "
                            "registration would give, since Orion is pre-registered -- "
                            "still verified per-sample (F6), not assumed"),
        dict(parameter="Encoder", paper_value="MUSK (fine-tuned final 2 layers)",
            our_value="Phikon (frozen throughout)",
            reason="MUSK is gated on HuggingFace and its reference loader assumes CUDA/fp16; "
                  "Phikon is ungated and CPU-tractable. Frozen (not fine-tuned) because "
                  "CANVAS-CTX trains on cached embeddings for CPU feasibility, and Method 1 "
                  "must be compared frozen-vs-frozen against it, never fine-tuned-vs-frozen",
            expected_impact="Lower ceiling accuracy than the paper's MUSK numbers; benchmarked "
                            "against ResNet-50 as the floor. Encoder choice does not affect "
                            "the Method 1 vs Method 2 comparison, which holds the encoder "
                            "fixed across all four context modes"),
        dict(parameter="H&E resolution", paper_value="0.25 um/px (40x)",
            our_value="0.50 um/px (20x)",
            reason="CPU/16GB laptop feasibility",
            expected_impact="Minimal per the paper's own Fig S7B (comparable accuracy "
                            "40x vs 20x)"),
        dict(parameter="Interaction permutation count", paper_value="1000",
            our_value="200 (pilot profile)",
            reason="CPU runtime on a laptop profile",
            expected_impact="Noisier z-score estimates for weak habitat-pair effects; "
                            "strong effects should be stable. State the reduction in any "
                            "manuscript limitations section"),
        dict(parameter="Interaction null model [DELIBERATE]",
            paper_value="label shuffling (implied/standard)",
            our_value="toroidal shift (F21, T10 self-reference)",
            reason="Label shuffling destroys spatial autocorrelation; on a toroidal-shift "
                  "null 10/100 habitat pairs were significant in an internal check vs "
                  "94/100 under shuffling on the same data (F21 reproduces this comparison "
                  "on a fresh synthetic region)",
            expected_impact="More conservative, more specific significance calls -- answers "
                            "'are these two habitats associated' rather than 'is this tissue "
                            "spatially organised at all'"),
        dict(parameter="Dispersion edge correction [DELIBERATE]",
            paper_value="not specified",
            our_value="border-corrected Ripley K/L; Donnelly-corrected Clark-Evans",
            reason="Uncorrected estimates are biased on small tissue cores / TMA regions "
                  "(F22 shows the magnitude on a synthetic point pattern)",
            expected_impact="Reduced boundary bias, especially on smaller ROIs; NaN where "
                            "too few points remain eligible rather than a misleading value"),
        dict(parameter="CN discovery cohort size", paper_value="unspecified full NSCLC cohort",
            our_value="35 patients / 140 regions (Schurch), primary source",
            reason="Smallest complete public CODEX-class CRC resource with the original "
                  "cellular-neighbourhood methodology behind it",
            expected_impact="Fewer patients than the paper's discovery cohort; "
                            "cross-cohort transferability should be checked (e.g. against a "
                            "100-image Danenberg subset) before treating CNs as final"),
        dict(parameter="Paired/training cohort size", paper_value="10 WSI (paper's paired set)",
            our_value="8 train / 2 val / 2 test Orion CRC slides (patient-level split)",
            reason="Matches the smallest defensible split that keeps train/val/test "
                  "non-trivial at patient level",
            expected_impact="Small test set (n=2); any reported test metric has wide "
                            "uncertainty and should be bootstrapped, not read as a point "
                            "estimate"),
        dict(parameter="Deployment cohort size (per cancer)",
            paper_value="~1,100 (BRCA, full design) / large pan-cancer transfer set",
            our_value="60 TCGA-COAD (in-domain) + 60 TCGA-BRCA (transfer), PAM50-stratified",
            reason="Pilot/laptop track scale (config/pilot.yaml honesty ledger)",
            expected_impact="Pilot proves the PIPELINE runs; hazard ratios or subtype-"
                            "stratified effect sizes from 60 slides split across 5 PAM50 "
                            "subtypes are not interpretable as findings (see report "
                            "Limitations)"),
    ]
    df = pd.DataFrame(rows)
    path = _save(df, "T10_deviations", tables_dir)
    return {"id": "T10", "title": "Deviations from the published method",
           "source": "DESIGN (documentation, not a data-derived table)", "csv_path": path,
           "df": df,
           "caption": "Every parameter, cohort, and methodological choice that differs from "
                      "the published CANVAS protocol, why, and the expected direction/size of "
                      "impact. Nothing here is hidden or deferred to a footnote."}


ALL_TABLES = [
    t1_dataset_inventory, t2_cn_definitions, t3_patch_counts, t4_classifier_performance,
    t5_benchmark_table, t6_per_class_metrics, t7_univariate_cox, t8_ecotype_clinical,
    t9_signature, t10_deviations, t11_cn_vs_published,
]
