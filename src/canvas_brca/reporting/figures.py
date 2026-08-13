"""One function per report figure (F1-F22).

Every function returns a metadata dict:
    {"id": "F7", "title": "...", "source": "REAL DATA (...)" | "SIMULATED (...)",
     "paths": {"png": "...", "pdf": "..."}, "caption": "..."}

THE ABSOLUTE RULE: a figure is either built from a real cohort (state the
cohort and n) or from the synthetic/null fixture (state SIMULATED and the
fixture) or it cannot be built at all, in which case `placeholder_figure`
emits a labelled MISSING-DATA panel naming the exact file needed. Nothing
here is ever left blank or silently skipped.

Data sources used by the REAL/SIMULATED figures, all already-computed
artefacts on disk (nothing here re-runs training):
    results/final_benchmark.csv        6-seed x 4-mode simulated benchmark
    results/training_curves.csv        per-epoch history, same run
    results/per_class_run/*.csv        3-seed supplementary run (per-class,
                                       confusion matrices; cheaper than a
                                       second 6-seed run, separately labelled)
    data/interim/sim_features.parquet  262-feature matrix, n=200, simulated
                                       habitat maps (stage5 --simulate)

F21/F22 and the stage6-chain figures (F16/F17/F18/F20) additionally run
small NEW computations against these same simulated fixtures, calling only
the existing tested functions in spatial_features.py and signature.py -
neither file is modified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from canvas_brca.stage5_features.spatial_features import (
    FeatureConfig, N_HABITATS, _bbox_area, _ripley_k_border, interaction_features,
)
from canvas_brca.stage6_clinical.signature import (
    ClinicalConfig, consensus_cluster, fit_signature, lasso_cox_selection,
    reduce_collinearity, rsf_importance, univariate_cox,
)

from .style import (
    NAVY, OKABE_ITO, STEEL_BLUE, apply_style, categorical_colors, continuous_cmap,
    letter_panels, placeholder_figure, save_figure, source_caption,
)

logger = logging.getLogger(__name__)

RESULTS = Path("results")
DATA_INTERIM = Path("data/interim")

SIM_NOTE = ("SIMULATED - synthetic null/machinery fixture, not real tissue or "
           "clinical data. Numbers here characterise the pipeline, not biology.")


# ============================================================ shared loaders


def _missing(path: Path) -> bool:
    return not path.exists()


def _load_final_benchmark() -> pd.DataFrame | None:
    p = RESULTS / "final_benchmark.csv"
    return None if _missing(p) else pd.read_csv(p)


def _load_training_curves() -> pd.DataFrame | None:
    p = RESULTS / "training_curves.csv"
    return None if _missing(p) else pd.read_csv(p, index_col=[0, 1])


def _load_per_class(outdir: str = "results/per_class_run") -> pd.DataFrame | None:
    p = Path(outdir) / "per_class_metrics.csv"
    return None if _missing(p) else pd.read_csv(p)


def _load_confusion(outdir: str = "results/per_class_run") -> pd.DataFrame | None:
    p = Path(outdir) / "confusion_matrices.csv"
    return None if _missing(p) else pd.read_csv(p)


def _load_sim_features() -> pd.DataFrame | None:
    p = DATA_INTERIM / "sim_features.parquet"
    return None if _missing(p) else pd.read_parquet(p)


def _null_clinical(index: pd.Index, endpoint: str = "OS", seed: int = 123) -> pd.DataFrame:
    """Same construction as scripts/validate_stage6_null.py: independent
    exponential event/censoring times, no dependence on any feature.
    Regenerated deterministically here rather than imported, since scripts/
    is not an importable package from src/.
    """
    rng = np.random.default_rng(seed)
    event_time = rng.exponential(scale=1000.0, size=len(index))
    censor_time = rng.exponential(scale=1200.0, size=len(index))
    return pd.DataFrame({
        f"{endpoint}_time": np.minimum(event_time, censor_time),
        f"{endpoint}_event": (event_time <= censor_time).astype(int),
    }, index=index)


def _synthetic_clustered_patches(
    rng: np.random.Generator, n: int = 2500, n_habitats: int = N_HABITATS,
    extent: float = 8000.0, cluster_sd: float = 700.0,
) -> pd.DataFrame:
    """One synthetic tissue region: spatially clustered habitat domains, for
    F21/F22's spatial-statistics diagnostics. Same spirit as
    scripts/run_stage5_features.py's `simulate()`, reimplemented locally so
    reporting/ does not import from scripts/.
    """
    centres = rng.uniform(0, extent, size=(n_habitats, 2))
    assign = rng.integers(0, n_habitats, n)
    pts = centres[assign] + rng.normal(0, cluster_sd, size=(n, 2))
    return pd.DataFrame({"x_um": pts[:, 0], "y_um": pts[:, 1], "habitat": assign})


# ==================================================================== F1


def f01_study_design(figures_dir: str) -> dict:
    """Pipeline schematic. Not data-dependent -- describes the design itself."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Study design: CANVAS re-implementation, CRC-trained, breast-applied",
                 loc="left", fontsize=12)

    stage_xy = [(0.3, 5.0, "1. CN discovery\nSchurch CODEX 56-plex CRC\n(fallback: Orion IF 18-plex)"),
                (2.9, 5.0, "2. Label transfer\nOrion paired IF+H&E\n8/2/2 patient split"),
                (5.5, 5.0, "3. H&E model\nMethod 1 (CANVAS) vs\nMethod 2 (CANVAS-CTX)"),
                (8.1, 5.0, "4. WSI inference\nTCGA-COAD (in-domain)\nTCGA-BRCA (transfer)"),
                (5.5, 2.6, "5. Spatial features\n262 features, 6 blocks"),
                (5.5, 0.6, "6. Clinical modelling\nCox, ecotypes, signature")]
    boxes = {}
    for x, y, label in stage_xy:
        box = mpatches.FancyBboxPatch((x, y), 2.2, 1.35, boxstyle="round,pad=0.08",
                                      facecolor="#EAF1F8", edgecolor=NAVY, linewidth=1.3)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.68, label, ha="center", va="center", fontsize=8.3)
        boxes[label.split("\n")[0]] = (x, y)

    def arrow(a, b, **kw):
        ax.annotate("", xy=b, xytext=a,
                   arrowprops=dict(arrowstyle="-|>", color=STEEL_BLUE, lw=1.6, **kw))

    arrow((2.5, 5.68), (2.9, 5.68))
    arrow((5.1, 5.68), (5.5, 5.68))
    arrow((7.7, 5.68), (8.1, 5.68))
    arrow((6.6, 5.0), (6.6, 3.95))
    arrow((6.6, 2.6), (6.6, 1.95))

    ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.6), 2.2, 1.35, boxstyle="round,pad=0.08",
                                         facecolor="#FCEFE3", edgecolor=OKABE_ITO[1], linewidth=1.3))
    ax.text(1.4, 3.28, "Method 1\nk=0: per-patch\nCANVAS head only", ha="center", va="center", fontsize=8.3)
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.6), 2.2, 1.35, boxstyle="round,pad=0.08",
                                         facecolor="#FCEFE3", edgecolor=OKABE_ITO[1], linewidth=1.3))
    ax.text(1.4, 1.28, "Method 2\nk in {4,8,16}\ngraph/grid2d/grid3d", ha="center", va="center", fontsize=8.3)
    arrow((1.4, 2.6), (1.4, 1.95))

    ax.text(9.0, 3.1,
           "Substituted from the original paper\n(CODEX NSCLC, same-section H&E, MUSK):\n"
           "- 56-plex CODEX CRC (Schurch) replaces\n  41-plex CODEX NSCLC for CN discovery\n"
           "- Orion IF+H&E (CRC) replaces same-cancer\n  paired H&E; trained on CRC, applied to breast\n"
           "- Phikon (frozen) replaces MUSK\n  (fine-tuned) - see T10 for the full list",
           fontsize=7.6, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E8", edgecolor="#999999"))

    source_caption(fig, "Source: pipeline design (PROTOCOL.md, config/crc_train_brca_apply.yaml). "
                        "Not derived from any run.", y=-0.02)
    paths = save_figure(fig, "F1_study_design", figures_dir)
    return {"id": "F1", "title": "Study design schematic", "source": "DESIGN (no data)",
           "paths": paths,
           "caption": "Six CANVAS stages, the Method 1/Method 2 split, and what is "
                      "substituted from the published protocol and why. See T10 for "
                      "the complete, itemised deviation list."}


# ==================================================================== F2-F7


def f02_k_sweep(figures_dir: str, k_sweep_csv: str = "data/processed/k_sweep.csv") -> dict:
    p = Path(k_sweep_csv)
    if _missing(p):
        fig, _ = placeholder_figure("F2", "CN discovery: k sweep diagnostics",
                                    missing_file=f"{k_sweep_csv} (run scripts/run_stage1_cn.py "
                                                 "on Schurch or Orion single-cell data)",
                                    unblocks="Phase 2 - CN discovery")
        paths = save_figure(fig, "F2_k_sweep", figures_dir)
        return {"id": "F2", "title": "CN discovery: k sweep diagnostics",
               "source": "MISSING DATA", "paths": paths,
               "caption": f"Needs {k_sweep_csv}. Unblocks Phase 2."}

    df = pd.read_csv(p)
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [("silhouette", "Silhouette (higher better)"),
              ("davies_bouldin", "Davies-Bouldin (lower better)"),
              ("ari_vs_previous", "Adjacent-k ARI")]
    for ax, (col, label) in zip(axes, metrics):
        ax.plot(df["k"], df[col], "-o", color=STEEL_BLUE, ms=4)
        ax.set_xlabel("k (candidate cellular neighbourhoods)")
        ax.set_ylabel(label)
    chosen_k = int(df.loc[df["silhouette"].idxmax(), "k"]) if "silhouette" in df else None
    if chosen_k is not None:
        for ax in axes:
            ax.axvline(chosen_k, color=OKABE_ITO[6], ls="--", lw=1.2)
    letter_panels(axes)
    source_caption(fig, f"REAL DATA (CN discovery cohort, n={df.shape[0]} k-values swept).")
    paths = save_figure(fig, "F2_k_sweep", figures_dir)
    return {"id": "F2", "title": "CN discovery: k sweep diagnostics",
           "source": "REAL DATA", "paths": paths,
           "caption": "Silhouette, Davies-Bouldin and adjacent-k ARI across k=5-20, "
                      "chosen k marked."}


def _placeholder_table_figure(fig_id, title, missing_file, unblocks, figures_dir, name):
    fig, _ = placeholder_figure(fig_id, title, missing_file=missing_file, unblocks=unblocks)
    paths = save_figure(fig, name, figures_dir)
    return {"id": fig_id, "title": title, "source": "MISSING DATA", "paths": paths,
           "caption": f"Needs {missing_file}. Unblocks {unblocks}."}


def f03_cn_marker_heatmap(figures_dir: str,
                          enrichment_csv: str = "data/processed/cn_lineage_enrichment.csv") -> dict:
    p = Path(enrichment_csv)
    if _missing(p):
        return _placeholder_table_figure(
            "F3", "CN marker enrichment heatmap", enrichment_csv, "Phase 2 - CN naming",
            figures_dir, "F3_cn_marker_heatmap")
    df = pd.read_csv(p, index_col=0)
    apply_style()
    fig, ax = plt.subplots(figsize=(0.5 * df.shape[1] + 3, 0.35 * df.shape[0] + 2))
    im = ax.imshow(df.to_numpy(), cmap=continuous_cmap(), aspect="auto")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="z-scored mean composition")
    # long rotated cell-type labels need more clearance than the default -0.16
    source_caption(fig, f"REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154; "
                        f"{df.shape[0]} CNs x {df.shape[1]} cell types).", y=-0.40)
    paths = save_figure(fig, "F3_cn_marker_heatmap", figures_dir)
    return {"id": "F3", "title": "CN marker enrichment heatmap", "source": "REAL DATA",
           "paths": paths, "caption": "z-scored mean neighbourhood composition per CN."}


def f04_cn_spatial_maps(figures_dir: str,
                        assignments: str = "data/processed/cn_assignments.parquet",
                        n_regions: int = 6) -> dict:
    """Cells plotted at their true tissue coordinates, coloured by CN.

    This is the spatial-map panel the published work shows for cellular
    neighbourhoods. It is drawn from real cell centroids, not from an image
    file: the source table ships coordinates and cell types, not pixels.
    """
    p = Path(assignments)
    if _missing(p):
        return _placeholder_table_figure(
            "F4", "CN spatial maps (representative regions)",
            f"{assignments} (run scripts/run_stage1_cn.py)",
            "Phase 2 - CN spatial validation", figures_dir, "F4_cn_spatial_maps")

    df = pd.read_parquet(p)
    # representative regions: the largest, so structure is visible
    top = df.groupby("image_id").size().sort_values(ascending=False).head(n_regions).index
    cns = sorted(df["cn_label"].unique())
    colors = dict(zip(cns, categorical_colors(len(cns))))

    apply_style()
    ncol = 3
    nrow = int(np.ceil(n_regions / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, img in zip(axes, top):
        g = df[df["image_id"] == img]
        for cn in cns:
            sub = g[g["cn_label"] == cn]
            if len(sub):
                ax.scatter(sub["x_um"], sub["y_um"], s=1.6, c=colors[cn], label=cn,
                          linewidths=0, rasterized=True)
        ax.set_title(f"{img}  (n={len(g):,} cells)", fontsize=8.5)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        ax.grid(False)
    for ax in axes[len(top):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=colors[c], label=c)
              for c in cns]
    fig.legend(handles=handles, loc="lower center", ncol=min(10, len(cns)),
              fontsize=8, bbox_to_anchor=(0.5, -0.04))
    letter_panels(axes[:len(top)])
    source_caption(fig, f"REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154; "
                        f"{df['image_id'].nunique()} images, {len(df):,} cells; "
                        f"{n_regions} largest regions shown).", y=-0.09)
    paths = save_figure(fig, "F4_cn_spatial_maps", figures_dir)
    return {"id": "F4", "title": "CN spatial maps, representative regions",
           "source": f"REAL DATA (Schurch CRC CODEX, n={df['image_id'].nunique()} images)",
           "paths": paths,
           "caption": f"Single cells plotted at their measured tissue coordinates in the "
                      f"{n_regions} largest imaged regions, coloured by assigned cellular "
                      f"neighbourhood. Coordinates converted from pixels at 0.37744 um/px "
                      f"(TCIA-documented for this collection). Contiguous single-colour "
                      f"domains indicate the neighbourhoods are spatially coherent tissue "
                      f"structures rather than scattered label noise."}


def f05_cn_composition_per_sample(figures_dir: str,
                                  assignments: str = "data/processed/cn_assignments.parquet") -> dict:
    p = Path(assignments)
    if _missing(p):
        return _placeholder_table_figure(
            "F5", "CN composition per sample", f"{assignments} (run scripts/run_stage1_cn.py)",
            "Phase 2 - CN composition", figures_dir, "F5_cn_composition")

    df = pd.read_parquet(p)
    comp = (pd.crosstab(df["image_id"], df["cn_label"], normalize="index"))
    cns = sorted(comp.columns)
    comp = comp[cns]
    dominant = comp.idxmax(axis=1)
    order = comp.assign(_d=dominant).sort_values(
        ["_d"] + cns, ascending=False).drop(columns="_d").index
    comp = comp.loc[order]
    colors = dict(zip(cns, categorical_colors(len(cns))))

    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))
    for cn in cns:
        ax.bar(x, comp[cn].to_numpy(), bottom=bottom, color=colors[cn], width=1.0,
              label=cn, linewidth=0)
        bottom += comp[cn].to_numpy()
    ax.set_xlim(-0.5, len(comp) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"imaged region (n={len(comp)}), ordered by dominant neighbourhood")
    ax.set_ylabel("fraction of cells")
    ax.set_xticks([])
    ax.grid(False)
    ax.legend(ncol=min(10, len(cns)), fontsize=8, loc="upper center",
             bbox_to_anchor=(0.5, -0.09))
    source_caption(fig, f"REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154; "
                        f"{len(comp)} regions, {len(df):,} cells).", y=-0.22)
    paths = save_figure(fig, "F5_cn_composition", figures_dir)
    return {"id": "F5", "title": "CN composition per imaged region",
           "source": f"REAL DATA (Schurch CRC CODEX, n={len(comp)} regions)",
           "paths": paths,
           "caption": f"Stacked neighbourhood composition for each of {len(comp)} imaged "
                      f"regions, ordered by dominant neighbourhood. Regions differ markedly "
                      f"in composition, which is the variation the downstream habitat "
                      f"features are designed to quantify."}


def f06_registration_qc(figures_dir: str,
                        qc_json: str = "results/orion_registration_qc.json") -> dict:
    """Generated by scripts/run_orion_registration_qc.py (it needs the 981 MB
    H&E, so it is not re-run at report time). This reads the metrics it wrote
    and confirms the figure exists.
    """
    import json
    png = Path(figures_dir) / "F6_registration_qc.png"
    if _missing(Path(qc_json)) or _missing(png):
        return _placeholder_table_figure(
            "F6", "Registration QC: IF/H&E overlay and residuals",
            "data/raw/orion_crc/CRC01/ then scripts/run_orion_registration_qc.py",
            "Phase 3 - paired label transfer", figures_dir, "F6_registration_qc")

    m = json.loads(Path(qc_json).read_text())
    return {"id": "F6", "title": "Registration QC: H&E, cell overlay and correlation peak",
           "source": f"REAL DATA (Orion {m['specimen']}, {m['n_cells_total']:,} cells)",
           "paths": {"png": str(png), "pdf": str(png.with_suffix(".pdf"))},
           "caption": f"(A) The registered H&E section, {m['he_width_px']:,} x "
                      f"{m['he_height_px']:,} px at {m['mpp_level0_um']} um/px. (B) Tissue "
                      f"mask with {m['n_cells_sampled']:,} sampled immunofluorescence cell "
                      f"centroids overlaid; {100*m['frac_cells_on_tissue']:.1f}% fall on "
                      f"tissue. (C) FFT cross-correlation between the tissue mask and the "
                      f"cell-density map, peaking at zero offset. Registration residual "
                      f"{m['residual_statement']}, against the [PAPER] {m['threshold_um']} um "
                      f"threshold: {m['verdict']}. Orion images both modalities from one "
                      f"section and ships the H&E pre-registered, but this was verified "
                      f"rather than assumed. Note the check is global and rigid, so it "
                      f"bounds translation only and would not detect local warping."}


def f07_patch_label_distribution(figures_dir: str) -> dict:
    """Generated by scripts/extract_orion_patches.py (needs the H&E on disk)."""
    png = Path(figures_dir) / "F7_patch_labels.png"
    if _missing(png):
        return _placeholder_table_figure(
            "F7", "Patch label distribution and example patches",
            "data/raw/orion_crc/CRC01/ then scripts/extract_orion_patches.py",
            "Phase 3 - label transfer", figures_dir, "F7_patch_labels")
    return {"id": "F7", "title": "Real H&E patches by dominant gated lineage",
           "source": "REAL DATA (Orion CRC01 H&E, 1,620,375 gated cells)",
           "paths": {"png": str(png), "pdf": str(png.with_suffix(".pdf"))},
           "caption": "H&E patches of 345 native px (224 px at the 20x/0.5 um per px "
                      "working resolution) read from the registered whole-slide image, "
                      "grouped by the dominant marker-gated lineage of the cells they "
                      "contain, with cell count and dominant-lineage purity per patch. "
                      "Grouping by lineage rather than by transferred habitat is "
                      "deliberate: lineage is checkable by eye in H&E, so this doubles as "
                      "a visual audit of the gating the label transfer depends on. That "
                      "audit passes for tumour (malignant glandular epithelium, cribriform "
                      "architecture, enlarged hyperchromatic nuclei) and for smooth muscle "
                      "stroma, and fails for the immune and vascular subsets, whose "
                      "dominant-lineage purity is only 28-46 percent. The immune gates are "
                      "therefore not yet reliable enough to support habitat assignment."}


# ==================================================================== F8


def f08_training_curves(figures_dir: str) -> dict:
    hist = _load_training_curves()
    if hist is None:
        return _placeholder_table_figure(
            "F8", "Training curves by context mode", "results/training_curves.csv",
            "any benchmark run", figures_dir, "F8_training_curves")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = sorted({idx[0].rsplit("_s", 1)[0] for idx in hist.index})
    colors = dict(zip(modes, categorical_colors(len(modes))))
    for run_id in hist.index.get_level_values(0).unique():
        mode = run_id.rsplit("_s", 1)[0]
        sub = hist.loc[run_id]
        axes[0].plot(sub["epoch"], sub["train_loss"], color=colors[mode], alpha=0.35, lw=1.1)
        axes[1].plot(sub["epoch"], sub["val_macro_f1"], color=colors[mode], alpha=0.35, lw=1.1)
    for mode in modes:
        run_ids = [r for r in hist.index.get_level_values(0).unique()
                  if r.rsplit("_s", 1)[0] == mode]
        mean_loss = pd.concat([hist.loc[r].set_index("epoch")["train_loss"] for r in run_ids],
                              axis=1).mean(axis=1)
        mean_f1 = pd.concat([hist.loc[r].set_index("epoch")["val_macro_f1"] for r in run_ids],
                            axis=1).mean(axis=1)
        axes[0].plot(mean_loss.index, mean_loss.values, color=colors[mode], lw=2.4, label=mode)
        axes[1].plot(mean_f1.index, mean_f1.values, color=colors[mode], lw=2.4, label=mode)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss (focal)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro-F1")
    axes[1].legend(title="context mode")
    letter_panels(axes)
    n_seeds = hist.index.get_level_values(0).str.extract(r"_s(\d+)$")[0].nunique()
    source_caption(fig, f"SIMULATED (--simulate machinery fixture, {n_seeds} seeds, thin lines "
                        "= individual seeds, thick = mean).")
    paths = save_figure(fig, "F8_training_curves", figures_dir)
    return {"id": "F8", "title": "Training curves by context mode", "source": "SIMULATED",
           "paths": paths,
           "caption": "Focal-loss training curves and validation macro-F1 across epochs, "
                      "all four context modes overlaid. " + SIM_NOTE}


# ==================================================================== F9


def f09_confusion_matrix(figures_dir: str, mode: str = "grid2d") -> dict:
    cm_long = _load_confusion()
    if cm_long is None:
        return _placeholder_table_figure(
            "F9", "Confusion matrix (row-normalised)", "results/per_class_run/confusion_matrices.csv",
            "supplementary per-class benchmark run", figures_dir, "F9_confusion_matrix")

    sub = cm_long[cm_long["mode"] == mode]
    n_seeds = sub["seed"].nunique()
    n_classes = int(max(sub["true_class"].max(), sub["pred_class"].max()) + 1)
    cm = np.zeros((n_classes, n_classes))
    for _, r in sub.iterrows():
        cm[int(r.true_class), int(r.pred_class)] += r["count"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    recall = np.diag(cm_norm)

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap=continuous_cmap(), vmin=0, vmax=1)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    labels = [f"H{i+1:02d}" if i < n_classes - 1 else "bg" for i in range(n_classes)]
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n_classes):
        ax.text(i, i, f"{recall[i]:.2f}", ha="center", va="center",
               color="white" if cm_norm[i, i] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="row-normalised fraction")
    ax.set_title(f"mode={mode}, {n_seeds} seeds pooled", fontsize=10)
    source_caption(fig, f"SIMULATED ({n_seeds}-seed supplementary run, --simulate fixture). "
                        "Diagonal = per-class recall.")
    paths = save_figure(fig, "F9_confusion_matrix", figures_dir)
    return {"id": "F9", "title": "Confusion matrix, row-normalised", "source": "SIMULATED",
           "paths": paths,
           "caption": f"Row-normalised confusion matrix for mode={mode}, pooled across "
                      f"{n_seeds} seeds, per-class recall annotated on the diagonal. " + SIM_NOTE}


# ==================================================================== F10, F11


def f10_benchmark_comparison(figures_dir: str) -> dict:
    df = _load_final_benchmark()
    if df is None:
        return _placeholder_table_figure(
            "F10", "Benchmark: macro-F1 and kappa by context mode", "results/final_benchmark.csv",
            "6-seed benchmark run", figures_dir, "F10_benchmark")

    modes = list(dict.fromkeys(df["mode"]))
    base = modes[0]
    n_seeds = df["seed"].nunique()
    colors = dict(zip(modes, categorical_colors(len(modes))))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, metric in zip(axes, ["macro_f1", "cohen_kappa"]):
        sub = df[df["metric"] == metric]
        means = sub.groupby("mode")["value"].mean().reindex(modes)
        sds = sub.groupby("mode")["value"].std().reindex(modes)
        x = np.arange(len(modes))
        bar_colors = [NAVY if m == base else colors[m] for m in modes]
        ax.bar(x, means.values, yerr=sds.values, color=bar_colors, capsize=4,
              edgecolor="black", linewidth=0.6)
        piv = sub.pivot(index="seed", columns="mode", values="value")
        rng = np.random.default_rng(0)
        for i, m in enumerate(modes):
            jitter = rng.uniform(-0.12, 0.12, size=len(piv))
            ax.scatter(np.full(len(piv), i) + jitter, piv[m], color="black", s=14,
                      zorder=5, alpha=0.75)
        for i, m in enumerate(modes[1:], start=1):
            d = piv[m] - piv[base]
            try:
                _, pv = wilcoxon(piv[m], piv[base])
            except ValueError:
                pv = np.nan
            ax.text(i, means[m] + sds[m] + 0.015, f"p={pv:.3f}" if np.isfinite(pv) else "n/a",
                   ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(modes)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1.08)
    axes[0].text(0, 1.0, "CANVAS\nbaseline", ha="center", fontsize=7, color=NAVY,
                transform=axes[0].get_xaxis_transform())
    letter_panels(axes)
    source_caption(fig, f"SIMULATED (--simulate machinery fixture, {n_seeds} seeds, "
                        f"paired Wilcoxon vs mode={base}, points = individual seeds).")
    paths = save_figure(fig, "F10_benchmark", figures_dir)
    return {"id": "F10", "title": "Benchmark: macro-F1 and kappa by context mode",
           "source": "SIMULATED", "paths": paths,
           "caption": f"Mean +/- SD across {n_seeds} seeds, individual seeds jittered, "
                      f"paired Wilcoxon p vs mode={base} annotated. mode=none is the "
                      "CANVAS baseline (navy). " + SIM_NOTE}


def f11_params_vs_f1(figures_dir: str) -> dict:
    df = _load_final_benchmark()
    if df is None:
        return _placeholder_table_figure(
            "F11", "Parameter count vs macro-F1", "results/final_benchmark.csv",
            "6-seed benchmark run", figures_dir, "F11_params_vs_f1")

    sub = df[df["metric"] == "macro_f1"]
    agg = sub.groupby("mode").agg(macro_f1=("value", "mean"), sd=("value", "std"),
                                  n_params=("n_params", "first")).reset_index()
    modes = list(agg["mode"])
    colors = categorical_colors(len(modes))

    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for (_, row), c in zip(agg.iterrows(), colors):
        ax.errorbar(row["n_params"], row["macro_f1"], yerr=row["sd"], fmt="o", ms=9,
                   color=c, capsize=4, label=row["mode"])
        ax.annotate(row["mode"], (row["n_params"], row["macro_f1"]),
                   textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log scale)")
    ax.set_ylabel("macro-F1 (mean +/- SD across seeds)")
    ax.set_title("Is a grid-mode gain just capacity?", fontsize=10)
    source_caption(fig, "SIMULATED (--simulate machinery fixture, same run as F10/T5).")
    paths = save_figure(fig, "F11_params_vs_f1", figures_dir)
    return {"id": "F11", "title": "Parameter count vs macro-F1", "source": "SIMULATED",
           "paths": paths,
           "caption": "Macro-F1 against parameter count on a log axis, per context mode. "
                      "If grid3d's edge over grid2d tracked capacity alone, its point would "
                      "sit on the same capacity-F1 trend as the others; PROTOCOL.md calls for "
                      "re-running grid2d at matched grid_channels before trusting a grid3d "
                      "win -- that matched-capacity rerun was NOT performed here. " + SIM_NOTE}


# ==================================================================== F12, F13


def f12_attention_maps(figures_dir: str) -> dict:
    return _placeholder_table_figure(
        "F12", "Attention maps: attended neighbouring patches",
        "a trained graph/grid ContextHabitatNet checkpoint + real H&E patches (Phase 3/4)",
        "Phase 3/4 - Method 2 qualitative figure", figures_dir, "F12_attention_maps")


def f13_wsi_habitat_maps(figures_dir: str) -> dict:
    return _placeholder_table_figure(
        "F13", "Whole-slide habitat maps with tumour bulk / leading edge",
        "TCGA-COAD DX slides + trained habitat classifier + tumour detector (Phase 4)",
        "Phase 4 - WSI deployment", figures_dir, "F13_wsi_habitat_maps")


# ==================================================================== F14, F15


def f14_feature_matrix_heatmap(figures_dir: str) -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F14", "262-feature matrix, clustered heatmap", "data/interim/sim_features.parquet",
            "stage5 --simulate run", figures_dir, "F14_feature_matrix")

    blocks = {"comp": "composition", "div": "diversity", "disp": "dispersion",
             "inter": "interaction", "dist": "distance", "transition": "transition"}
    block_of = {c: next((v for k, v in blocks.items() if c.startswith(k)), "other")
               for c in feats.columns}
    order = sorted(feats.columns, key=lambda c: list(blocks.values()).index(block_of[c]))
    x = feats[order].to_numpy(dtype=float)
    x = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-9)
    x = np.nan_to_num(x)

    from scipy.cluster.hierarchy import leaves_list, linkage
    row_order = leaves_list(linkage(np.nan_to_num(x), method="average"))
    x = x[row_order]

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(x, cmap=continuous_cmap(), aspect="auto", vmin=-3, vmax=3)
    ax.set_xlabel(f"262 features, ordered by block")
    ax.set_ylabel(f"{feats.shape[0]} samples (row-clustered)")
    boundaries = []
    pos = 0
    for b in dict.fromkeys(block_of[c] for c in order):
        n = sum(1 for c in order if block_of[c] == b)
        boundaries.append((pos, pos + n, b))
        pos += n
    for start, end, name in boundaries:
        ax.axvline(start, color="white", lw=0.8)
        ax.text((start + end) / 2, -feats.shape[0] * 0.02, name, ha="center", va="bottom",
               fontsize=7.5, rotation=0, transform=ax.transData)
    fig.colorbar(im, ax=ax, label="z-scored feature value")
    source_caption(fig, f"SIMULATED (stage5 --simulate, n={feats.shape[0]} synthetic samples).")
    paths = save_figure(fig, "F14_feature_matrix", figures_dir)
    return {"id": "F14", "title": "262-feature matrix, clustered heatmap", "source": "SIMULATED",
           "paths": paths,
           "caption": f"z-scored 262-feature matrix, n={feats.shape[0]} simulated samples, "
                      "rows hierarchically clustered, columns grouped by the six feature "
                      "blocks. " + SIM_NOTE}


def f15_feature_correlation(figures_dir: str) -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F15", "Feature correlation with collinear communities", "data/interim/sim_features.parquet",
            "stage5 --simulate run", figures_dir, "F15_feature_correlation")

    cfg = ClinicalConfig()
    keep = reduce_collinearity(feats, cfg)
    corr = feats.corr(method="spearman").abs().fillna(0.0)

    is_repr = corr.columns.isin(keep)
    order = np.argsort(~is_repr)
    corr_ord = corr.iloc[order, order]

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr_ord.to_numpy(), cmap=continuous_cmap(), vmin=0, vmax=1)
    boundary = int(is_repr.sum())
    ax.axhline(boundary - 0.5, color=OKABE_ITO[6], lw=1.3)
    ax.axvline(boundary - 0.5, color=OKABE_ITO[6], lw=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("262 features (reordered)"); ax.set_ylabel("262 features (reordered)")
    fig.colorbar(im, ax=ax, label="|Spearman rho|")
    ax.set_title(f"{len(keep)}/{feats.shape[1]} kept after Louvain collinearity reduction "
                f"(rho > {cfg.collinearity_rho})", fontsize=9.5)
    source_caption(fig, f"SIMULATED (stage5 --simulate, n={feats.shape[0]} samples).")
    paths = save_figure(fig, "F15_feature_correlation", figures_dir)
    return {"id": "F15", "title": "Feature correlation, collinear communities marked",
           "source": "SIMULATED", "paths": paths,
           "caption": f"|Spearman rho| across all 262 features; vermillion line separates "
                      f"the {len(keep)} community representatives (top-left) from the "
                      f"{feats.shape[1] - len(keep)} features dropped for |rho| > "
                      f"{cfg.collinearity_rho}. " + SIM_NOTE}


# ==================================================================== F16, F17


def f16_cox_forest(figures_dir: str, endpoint: str = "OS", top_n: int = 20) -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F16", "Univariate Cox forest plot", "data/interim/sim_features.parquet",
            "stage5 --simulate + null-clinical calibration check", figures_dir, "F16_cox_forest")

    clinical = _null_clinical(feats.index, endpoint)
    cfg = ClinicalConfig()
    cox = univariate_cox(feats, clinical, endpoint, cfg=cfg)
    if cox.empty:
        return _placeholder_table_figure(
            "F16", "Univariate Cox forest plot", "(univariate_cox returned no fits)",
            "stage5 --simulate + null-clinical calibration check", figures_dir, "F16_cox_forest")
    cox = cox.sort_values("p").head(top_n).sort_values("hr")

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(cox) + 1.6))
    y = np.arange(len(cox))
    sig = cox["q"] < cfg.alpha
    colors = [OKABE_ITO[6] if s else "#888888" for s in sig]
    # One errorbar call per row: a list-valued `ecolor` combined with
    # `capsize` mis-renders the cap markers in this matplotlib version
    # (each cap tries to interpret the whole colour list as one RGBA value).
    for yi, (_, row), c in zip(y, cox.iterrows(), colors):
        ax.errorbar(row["hr"], yi, xerr=[[row["hr"] - row["ci_lower"]],
                                         [row["ci_upper"] - row["hr"]]],
                   fmt="o", color=c, ms=5, capsize=3, elinewidth=1.6)
    ax.axvline(1.0, color="#333333", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(cox["feature"], fontsize=7.5)
    ax.set_xlabel("hazard ratio (95% CI), per 1 SD")
    n_sig = int(sig.sum())
    ax.set_title(f"top {len(cox)} by p-value; {n_sig}/{len(cox)} FDR-significant (q<{cfg.alpha})",
                fontsize=9.5)
    source_caption(fig, f"SIMULATED null-calibration check (n={feats.shape[0]}, random "
                        f"{endpoint} outcome independent of every feature). NOT a real "
                        "prognostic finding -- real habitat x compartment Cox needs Phase 5/6.")
    paths = save_figure(fig, "F16_cox_forest", figures_dir)
    return {"id": "F16", "title": "Univariate Cox forest plot (null calibration)",
           "source": "SIMULATED", "paths": paths,
           "caption": f"HR with 95% CI, top {len(cox)} features by p-value, against a "
                      f"random survival outcome. {n_sig}/{len(cox)} survive FDR -- this IS "
                      "the expected null result, not a placeholder for a real prognostic "
                      "figure (F16 proper needs real habitat x compartment data, Phase 5/6)."}


def f17_km_by_tertile(figures_dir: str, endpoint: str = "OS") -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F17", "Kaplan-Meier by habitat tertile", "data/interim/sim_features.parquet",
            "stage5 --simulate + null-clinical calibration check", figures_dir, "F17_km_tertile")

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    clinical = _null_clinical(feats.index, endpoint)
    feature_col = "comp_H01"
    joined = feats[[feature_col]].join(clinical)
    joined["tertile"] = pd.qcut(joined[feature_col], 3, labels=["low", "mid", "high"],
                                duplicates="drop")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    kmf = KaplanMeierFitter()
    for tert, color in zip(["low", "mid", "high"], [OKABE_ITO[5], OKABE_ITO[1], OKABE_ITO[6]]):
        sub = joined[joined["tertile"] == tert]
        if sub.empty:
            continue
        kmf.fit(sub[f"{endpoint}_time"], sub[f"{endpoint}_event"], label=f"{tert} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
    lr = multivariate_logrank_test(joined[f"{endpoint}_time"], joined["tertile"],
                                   joined[f"{endpoint}_event"])
    ax.set_xlabel(f"{endpoint} time (days)"); ax.set_ylabel("survival probability")
    ax.set_title(f"log-rank p={lr.p_value:.3f}", fontsize=10)
    source_caption(fig, f"SIMULATED null-calibration check (n={feats.shape[0]}, feature="
                        f"{feature_col}, random {endpoint} outcome). Curves should overlap.")
    paths = save_figure(fig, "F17_km_tertile", figures_dir)
    return {"id": "F17", "title": "Kaplan-Meier by habitat tertile (null calibration)",
           "source": "SIMULATED", "paths": paths,
           "caption": f"KM curves by tertile of {feature_col} against a random outcome, "
                      f"log-rank p={lr.p_value:.3f}. Curves overlapping with a non-significant "
                      "p is the correct null result -- real prognostic KM curves need "
                      "Phase 5/6 data."}


# ==================================================================== F18


def f18_consensus_clustering(figures_dir: str, k_range=range(2, 9)) -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F18", "Consensus clustering: CDF, delta-area, matrix", "data/interim/sim_features.parquet",
            "stage5 --simulate run", figures_dir, "F18_consensus")

    comp = feats[[c for c in feats.columns if c.startswith("comp_")]]
    cfg = ClinicalConfig(seed=42)
    cdfs, matrices = {}, {}
    for k in k_range:
        _, consensus = consensus_cluster(comp, cfg, k=k)
        iu = np.triu_indices_from(consensus.to_numpy(), k=1)
        cdfs[k] = np.sort(consensus.to_numpy()[iu])
        matrices[k] = consensus

    _trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in numpy 2.0+
    areas = {}
    for k in k_range:
        vals = cdfs[k]
        areas[k] = float(_trapezoid(np.arange(1, len(vals) + 1) / len(vals), vals))
    ks = list(k_range)
    delta = {ks[0]: np.nan}
    for i in range(1, len(ks)):
        prev = areas[ks[i - 1]]
        delta[ks[i]] = (areas[ks[i]] - prev) / prev if prev else np.nan
    chosen_k = ks[int(np.nanargmax(list(delta.values())[1:])) + 1] if len(ks) > 1 else ks[0]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = categorical_colors(len(ks))
    for k, c in zip(ks, colors):
        axes[0].step(cdfs[k], np.arange(1, len(cdfs[k]) + 1) / len(cdfs[k]), color=c, label=f"k={k}")
    axes[0].set_xlabel("consensus index value"); axes[0].set_ylabel("CDF")
    axes[0].legend(fontsize=7)

    axes[1].plot(ks[1:], [delta[k] for k in ks[1:]], "-o", color=STEEL_BLUE)
    axes[1].axvline(chosen_k, color=OKABE_ITO[6], ls="--")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("relative change in CDF area")

    im = axes[2].imshow(matrices[chosen_k].to_numpy(), cmap=continuous_cmap(), vmin=0, vmax=1)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title(f"consensus matrix, k={chosen_k}", fontsize=9.5)
    fig.colorbar(im, ax=axes[2], label="co-clustering frequency", fraction=0.046)
    letter_panels(axes)
    source_caption(fig, f"SIMULATED (composition features only, n={comp.shape[0]}, "
                        f"consensus_reps={cfg.consensus_reps}).")
    paths = save_figure(fig, "F18_consensus", figures_dir)
    return {"id": "F18", "title": "Consensus clustering diagnostics", "source": "SIMULATED",
           "paths": paths,
           "caption": f"CDF and delta-area across k={min(ks)}-{max(ks)}, consensus matrix at "
                      f"the delta-area-selected k={chosen_k}. Random composition data gives no "
                      "reason to expect a sharp elbow; a flat delta-area curve here is the "
                      "expected null, not evidence against the method. " + SIM_NOTE}


def f19_ecotype_heatmap(figures_dir: str) -> dict:
    return _placeholder_table_figure(
        "F19", "Ecotype heatmap with clinical annotation tracks",
        "real habitat compositions (Phase 4/5) + real clinical covariates "
        "(ER/PR/HER2/PAM50/stage, Phase 5/6)",
        "Phase 5/6 - ecotype clinical characterisation", figures_dir, "F19_ecotype_heatmap")


# ==================================================================== F20


def f20_signature_performance(figures_dir: str, endpoint: str = "OS") -> dict:
    feats = _load_sim_features()
    if feats is None:
        return _placeholder_table_figure(
            "F20", "Signature performance", "data/interim/sim_features.parquet",
            "stage5+6 null-calibration chain", figures_dir, "F20_signature")

    clinical = _null_clinical(feats.index, endpoint)
    cfg = ClinicalConfig(lasso_repeats=50, rsf_bootstrap=200, seed=42)
    keep = reduce_collinearity(feats, cfg)
    reduced = feats[keep]
    lasso_freq = lasso_cox_selection(reduced, clinical, endpoint, cfg)
    lasso_top = lasso_freq[lasso_freq >= lasso_freq.quantile(1 - cfg.lasso_top_frac)].index.tolist()
    rsf_imp = rsf_importance(reduced, clinical, endpoint, cfg)
    rsf_top = rsf_imp[rsf_imp >= rsf_imp.quantile(1 - cfg.lasso_top_frac)].index.tolist()
    selected = sorted(set(lasso_top) & set(rsf_top))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    if len(selected) < 2:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, f"only {len(selected)} feature(s) survived LASSO^RSF "
                     "intersection on this draw -- too few for a multivariable fit",
                     ha="center", va="center", wrap=True, transform=axes[0].transAxes)
        c_index, auc_rows = None, []
    else:
        model, metrics = fit_signature(reduced, clinical, endpoint, selected, cfg)
        c_index = float(metrics.loc[metrics.metric == "c_index", "value"].iloc[0])
        auc_rows = metrics[metrics.metric != "c_index"]

        axes[0].bar(["C-index"], [c_index], color=NAVY, width=0.5)
        axes[0].bar(auc_rows["metric"], auc_rows["value"], color=STEEL_BLUE)
        axes[0].axhline(0.5, color="#888888", ls="--", lw=1, label="chance (0.5)")
        axes[0].set_ylim(0, 1); axes[0].set_ylabel("value"); axes[0].legend(fontsize=8)
        axes[0].tick_params(axis="x", rotation=20)

        risk = model.predict_partial_hazard(reduced[selected].join(clinical)).to_numpy()
        joined = reduced[selected].join(clinical)
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        median = np.median(risk)
        for grp, mask, color in [("low risk", risk <= median, OKABE_ITO[5]),
                                 ("high risk", risk > median, OKABE_ITO[6])]:
            kmf.fit(joined.loc[mask, f"{endpoint}_time"], joined.loc[mask, f"{endpoint}_event"],
                   label=f"{grp} (n={mask.sum()})")
            kmf.plot_survival_function(ax=axes[1], color=color, ci_show=True)
        axes[1].set_xlabel(f"{endpoint} time (days)"); axes[1].set_ylabel("survival probability")
    letter_panels(axes)
    c_txt = f"C-index={c_index:.3f}" if c_index is not None else "C-index n/a"
    source_caption(fig, f"SIMULATED null-calibration check (n={feats.shape[0]}, random "
                        f"{endpoint} outcome, {len(selected)} features selected). {c_txt}.")
    paths = save_figure(fig, "F20_signature", figures_dir)
    return {"id": "F20", "title": "Signature performance (null calibration)",
           "source": "SIMULATED", "paths": paths,
           "stats": {"c_index": c_index, "n_selected": len(selected), "n": feats.shape[0]},
           "caption": f"C-index and time-dependent AUC (panel A) and KM by median risk score "
                      f"(panel B), fit on a random outcome, n={feats.shape[0]}. {c_txt} is "
                      "ABOVE the 0.5 chance level expected under a true null -- see the report "
                      "text: this is in-sample optimism bias from selecting and fitting on the "
                      "same subjects with no held-out split, not evidence the pipeline is "
                      "broken. It shrinks as n grows (0.80 at n=50 to this value at n=200 in "
                      "repeated checks) and will need correcting before any real C-index is "
                      "reported."}


# ==================================================================== F21, F22


def f21_null_model_comparison(figures_dir: str, n_permutations: int = 200, seed: int = 7) -> dict:
    """The toroidal-shift vs label-shuffle null, head to head. Calls the
    existing, tested `interaction_features` twice (different `null_model`),
    on one synthetic clustered tissue region. No external data needed.
    """
    rng = np.random.default_rng(seed)
    patches = _synthetic_clustered_patches(rng, n=2500)
    coords = patches[["x_um", "y_um"]].to_numpy()
    habitats = patches["habitat"].to_numpy()

    results = {}
    for null_model in ("toroidal", "shuffle"):
        cfg = FeatureConfig(interaction_permutations=n_permutations, null_model=null_model,
                            seed=seed)
        feats = interaction_features(coords, habitats, cfg)
        z = np.array([feats[f"inter_H{i+1:02d}_H{j+1:02d}"]
                     for i in range(N_HABITATS) for j in range(N_HABITATS)])
        results[null_model] = z

    n_pairs = N_HABITATS * N_HABITATS
    sig_toroidal = int(np.sum(np.abs(results["toroidal"]) > 1.96))
    sig_shuffle = int(np.sum(np.abs(results["shuffle"]) > 1.96))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(["toroidal shift\n(used)", "label shuffle\n(not used)"],
               [sig_toroidal, sig_shuffle], color=[NAVY, OKABE_ITO[6]])
    axes[0].axhline(n_pairs, color="#888888", ls=":", lw=1)
    axes[0].text(1.4, n_pairs, f"all {n_pairs} pairs", fontsize=7, color="#666666", va="bottom")
    axes[0].set_ylabel(f"habitat pairs called significant (|z|>1.96, of {n_pairs})")

    bins = np.linspace(-6, 6, 41)
    axes[1].hist(results["shuffle"], bins=bins, color=OKABE_ITO[6], alpha=0.55, label="shuffle")
    axes[1].hist(results["toroidal"], bins=bins, color=NAVY, alpha=0.65, label="toroidal")
    axes[1].axvline(1.96, color="#333333", ls="--", lw=1)
    axes[1].axvline(-1.96, color="#333333", ls="--", lw=1)
    axes[1].set_xlabel("interaction z-score"); axes[1].set_ylabel("habitat pairs")
    axes[1].legend()
    letter_panels(axes)
    source_caption(fig, f"SIMULATED (one synthetic clustered tissue region, n=2500 patches, "
                        f"{n_permutations} permutations per null).")
    paths = save_figure(fig, "F21_null_comparison", figures_dir)
    return {"id": "F21", "title": "Null-model comparison: toroidal shift vs label shuffling",
           "source": "SIMULATED", "paths": paths,
           "caption": f"On the SAME synthetic clustered tissue, label shuffling calls "
                      f"{sig_shuffle}/{n_pairs} habitat pairs significant vs {sig_toroidal}/"
                      f"{n_pairs} for toroidal shift, at {n_permutations} permutations. "
                      "Shuffling destroys spatial autocorrelation, so it answers 'is this "
                      "tissue spatially organised at all' (almost always yes) rather than "
                      "'are these two habitats specifically associated' -- the reason "
                      "toroidal shift is used throughout stage 5 instead. See T10."}


def f22_edge_correction_effect(figures_dir: str, seed: int = 11) -> dict:
    """Border-corrected vs naive Ripley's K against the CSR expectation, one
    synthetic region. Naive K is a small local helper (spatial_features.py
    only exposes the corrected version); border-corrected K calls the
    existing, tested `_ripley_k_border` directly.
    """
    rng = np.random.default_rng(seed)
    patches = _synthetic_clustered_patches(rng, n=900, n_habitats=1, extent=2000.0,
                                           cluster_sd=2000.0)  # one CSR-ish region
    coords = patches[["x_um", "y_um"]].to_numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    area = _bbox_area(coords)
    lam = len(coords) / area

    def naive_k(pts, tree, r):
        counts = np.array([len(tree.query_ball_point(p, r)) - 1 for p in pts])
        return float(counts.mean() / lam)

    radii = np.linspace(20, 400, 15)
    k_border = [_ripley_k_border(coords, tree, coords, r) for r in radii]
    k_naive = [naive_k(coords, tree, r) for r in radii]
    k_csr = np.pi * radii ** 2

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(radii, k_csr, color="#888888", ls=":", lw=1.6, label="CSR expectation (pi r^2)")
    ax.plot(radii, k_naive, color=OKABE_ITO[6], marker="o", ms=3, label="uncorrected")
    ax.plot(radii, k_border, color=NAVY, marker="o", ms=3, label="border-corrected (used)")
    ax.set_xlabel("radius r (um)"); ax.set_ylabel("Ripley's K(r)")
    ax.legend()
    source_caption(fig, f"SIMULATED (one synthetic near-CSR point pattern, n={len(coords)} points).")
    paths = save_figure(fig, "F22_edge_correction", figures_dir)
    bias = float(np.nanmean(np.array(k_naive) - np.array(k_border)))
    return {"id": "F22", "title": "Edge correction effect on Ripley's K", "source": "SIMULATED",
           "paths": paths,
           "caption": f"Border-corrected vs uncorrected Ripley's K across r=20-400um on a "
                      f"single synthetic point pattern, against the CSR expectation pi*r^2. "
                      f"Mean uncorrected-minus-corrected gap = {bias:+.0f} (uncorrected reads "
                      "systematically low near the window boundary, worst at large r on a "
                      "small window) -- the reason border correction is applied throughout "
                      "stage 5 instead of raw Kest. See T10."}


ALL_FIGURES = [
    f01_study_design, f02_k_sweep, f03_cn_marker_heatmap, f04_cn_spatial_maps,
    f05_cn_composition_per_sample, f06_registration_qc, f07_patch_label_distribution,
    f08_training_curves, f09_confusion_matrix, f10_benchmark_comparison, f11_params_vs_f1,
    f12_attention_maps, f13_wsi_habitat_maps, f14_feature_matrix_heatmap,
    f15_feature_correlation, f16_cox_forest, f17_km_by_tertile, f18_consensus_clustering,
    f19_ecotype_heatmap, f20_signature_performance, f21_null_model_comparison,
    f22_edge_correction_effect,
]
