# canvas-ctx

Spatial tumour habitat inference from routine H&E histology, with a
context-aware extension.

This repository re-implements the CANVAS approach — discovering cellular
neighbourhoods (CNs) from spatial proteomics, transferring those labels onto
co-registered H&E, and predicting habitats from morphology alone — and adds
**CANVAS-CTX**, which predicts the habitat of a patch from that patch *plus its
spatial neighbours*.

## Motivation

CANVAS defines each habitat label from a 40 µm cellular neighbourhood, but
predicts it from a single 224×224 patch viewed in isolation. The label is
contextual; the predictor is not. Information that defines the target is
withheld from the model.

CANVAS-CTX closes that gap with three interchangeable context encoders feeding
the **unchanged** CANVAS classification head:

| mode | context representation | invariance |
|---|---|---|
| `none` | per-patch (this is CANVAS exactly) | n/a |
| `graph` | k-NN deep set + distance-biased attention | permutation invariant |
| `grid2d` | local W×W feature lattice, 2D convolution | orientation aware |
| `grid3d` | local S×W×W multi-scale cube, 3D convolution | orientation + scale aware |

At `k_neighbours = 0` the context branch degenerates to passing the raw
embedding through, which is architecturally identical to `none`. The neighbour
sweep is therefore a controlled ablation rather than a comparison between
different models — same encoder, same cached embeddings, same loss, same
sampler, same sample-level splits, same head. This equivalence is enforced by a
test, not asserted in prose.

## Study design: train on colorectal, apply to breast

The pipeline requires spatial proteomics and H&E **from the same tissue
section**. Breast cancer has no public resource of that kind at usable scale.
Colorectal cancer does. So CNs and the habitat classifier are derived in
colorectal tissue and applied unchanged to breast, making breast the
generalisation test rather than the foundation. The original work performed the
structurally equivalent experiment, training on lung and applying across other
tumour types.

## Data

All primary data are public. Nothing in this repository redistributes them.

| Cohort | Platform | Role | Accession |
|---|---|---|---|
| Schürch 2020 | CODEX, 56-plex, colorectal | CN discovery | Mendeley Data `mpjzbtfgfr` |
| Orion (Lin 2023) | 18-plex IF + same-section H&E, colorectal | Paired label transfer | `s3://lin-2023-orion-crc` |
| TCGA-COAD / TCGA-BRCA | Diagnostic H&E WSI | Deployment and transfer | GDC Data Portal |
| TCGA-CDR | Curated clinical outcomes | Survival endpoints | GDC PanCanAtlas |

Coordinate units are verified before analysis in every stage. Applying a
micron radius to pixel coordinates collapses every neighbourhood while still
producing clean-looking clusters, so the loaders refuse to proceed on an
implausible coordinate span rather than guessing.

## Installation

```bash
git clone <repository-url>
cd canvas-ctx
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10+. The pipeline is CPU-only by default; no GPU is required.

For Google Colab, open `notebooks/canvas_ctx_colab.ipynb` and run the setup
cell. It installs dependencies, clones the repository and detects any available
GPU automatically.

## Usage

Stages are independent and each writes a cached artefact, so any stage can be
re-run without repeating the ones before it.

```bash
# Stage 1 — discover cellular neighbourhoods
python scripts/prepare_schurch.py
python scripts/run_stage1_cn.py --config config/crc_train_brca_apply.yaml \
    --cells data/interim/schurch_crc_cells.parquet
python scripts/validate_cn_vs_published.py       # external validation

# Stage 2 — paired cohort: registration QC, habitats, patch labels
python scripts/download_orion.py --specimens CRC01 CRC02 CRC03
python scripts/run_orion_registration_qc.py --level 3
python scripts/run_orion_cohort.py

# Stage 3 — cache patch embeddings
python scripts/encode_orion_patches.py --encoder phikon

# Method 1 vs Method 2 — the controlled ablation
python scripts/run_final_benchmark.py \
    --embeddings data/interim/orion_embeddings \
    --modes none graph grid2d grid3d --seeds 1 2 3 4 5 6 --epochs 30

# Stages 5 and 6 — spatial features and clinical modelling
python scripts/run_stage5_features.py --habitats data/processed/habitats.parquet
python scripts/run_stage6_clinical.py --features data/processed/spatial_features.parquet \
    --clinical data/raw/clinical/tcga_cdr.csv --endpoint OS

# Regenerate the full report (reads cached artefacts, recomputes nothing)
python scripts/run_report.py
```

Every script supports `--help`. Stages are idempotent: an existing output is
reused unless `--force` is passed.

## Repository layout

The distribution is named `canvas-ctx`; the import package remains
`canvas_brca` from the project's origin as a breast-cancer adaptation. Renaming
it would touch every tested module for no functional gain, so the two names are
left to differ.

```
config/           YAML configuration, resolved through an `inherit:` chain
src/canvas_brca/
  stage1_cn/      neighbourhood construction, spatial-LDA, k sweep
  stage2_pair/    registration, CN-to-nucleus transfer, patch purity rules
  stage3_model/   encoders, embedding extraction, CANVAS head, evaluation
  stage4_infer/   whole-slide intake, compartment assignment
  stage5_features/  the 262 spatial features
  stage6_clinical/  Cox models, ecotypes, prognostic signature
  method2_context/  graph, 2D grid and 3D grid context encoders
  reporting/      figures, tables, report and manuscript generation
scripts/          command-line entry point for each stage
tests/            pytest suite, synthetic fixtures only, no downloads
notebooks/        Colab notebook
```

## Methodological parameters

`PROTOCOL.md` is the authoritative specification. Parameters marked `[PAPER]`
are taken from the published STAR Methods and are not changed without recording
the change in the deviation table (`T10` in the generated report).

Two deviations are deliberate and are quantified in the report rather than
buried:

1. **Toroidal-shift null** for habitat-pair interaction, instead of label
   shuffling. Shuffling destroys spatial autocorrelation, so the null describes
   randomly scattered habitats — a configuration real tissue never adopts — and
   nearly every pair is declared significant against it.
2. **Edge-corrected spatial statistics**: border-corrected Ripley's K and L,
   Donnelly-corrected Clark–Evans. Uncorrected estimators are biased on small
   tissue regions because boundary points have artificially few neighbours.

## Reporting standards

The report generator enforces two rules mechanically:

- Every figure and table states its data source: the cohort and *n* for real
  data, or an explicit `SIMULATED` label for synthetic fixtures. A result that
  cannot be produced becomes a labelled placeholder naming the exact file
  required, and stays numbered in sequence.
- Nothing generated may be silently omitted. `build_report()` raises if a
  figure or table exists but no section references it.

Classifier performance is reported as macro-F1 and Cohen's kappa with bootstrap
confidence intervals and per-class recall, never accuracy alone: the background
class dominates and accuracy conceals a collapsed minority habitat. Model
comparisons use at least six random seeds with a paired Wilcoxon test.

## Tests

```bash
python -m pytest tests/ -q
```

Synthetic fixtures only. No test downloads data or contacts the network.

## Status

Implemented and validated on real data: cellular neighbourhood discovery,
registration verification, patch extraction and label transfer, the full 262
-feature engine, and the clinical statistics chain.

The habitat classifier benchmark on real tissue requires the multi-specimen
paired cohort and is the current work in progress. Benchmark numbers presently
in the report come from a synthetic fixture that is deliberately constructed to
reward spatial context; they characterise the machinery, not tissue, and are
labelled as such throughout.

A known caveat is documented in the report: the prognostic signature chain
selects features and evaluates the fitted model on the same subjects, which
inflates the concordance index on data with no real signal. Any reported
C-index must come from held-out data or carry an explicit optimism correction.

## References

- Schürch CM, et al. Coordinated cellular neighborhoods orchestrate antitumoral
  immunity at the colorectal cancer invasive front. *Cell* 2020;182(5):1341-1359.
  PMID 32763154.
- Lin JR, et al. High-plex immunofluorescence imaging and traditional histology
  of the same tissue section for discovering image-based biomarkers.
  *Nat Cancer* 2023;4(7):1036-1052. doi:10.1038/s43018-023-00576-1
- Liu J, et al. An integrated TCGA pan-cancer clinical data resource to drive
  high-quality survival outcome analytics. *Cell* 2018;173(2):400-416.
  PMID 29625055.
- Filiot A, et al. Scaling self-supervised learning for histopathology with
  masked image modeling. *medRxiv* 2023. doi:10.1101/2023.07.21.23292757

## License

MIT. See `LICENSE`.
