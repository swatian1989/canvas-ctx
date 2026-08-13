# Pipeline reference

Every script is a command-line entry point with `--help`, is idempotent
(an existing output is reused unless `--force` is given), and writes its
artefacts under `data/` or `results/`. Stages communicate only through those
files, so any stage can be re-run without repeating the ones before it.

Run order for a full reproduction is top to bottom.

---

## `run_pipeline.py` — single entry point

Runs every stage below in order, in VS Code, a terminal, or Colab. It shells
out to the same scripts documented here rather than reimplementing them, so
there is one source of truth per stage.

```bash
python scripts/run_pipeline.py                 # everything
python scripts/run_pipeline.py --dry-run       # plan only, nothing executes
python scripts/run_pipeline.py --quick         # fewer seeds/epochs/samples
python scripts/run_pipeline.py --from 9        # resume from a stage
python scripts/run_pipeline.py --stages 3 4    # selected stages only
python scripts/run_pipeline.py --specimens CRC01 CRC02 CRC03
python scripts/run_pipeline.py --force         # rerun completed stages
```

Completion is judged per stage by whether its declared outputs exist. A stage
that writes one shard per specimen, such as patch encoding, is treated as
complete only when every shard is present; testing the containing directory
would report a half-finished encoding run as done and let training proceed on
partial data. Partial stages print `resuming, N/M shards present`.

A stage whose inputs are missing is reported `BLOCKED` with the missing paths
named, rather than failing obscurely part way through.

### `download_schurch.py`
Fetches the discovery cohort table (223 MB) from Mendeley Data and verifies it
by SHA-256 against the published checksum. A truncated or corrupted download is
deleted rather than kept, and a complete file is left alone.

```bash
python scripts/download_schurch.py [--force] [--skip-checksum]
```
**Writes** `data/raw/CRC_clusters_neighborhoods_markers.csv`

---

## Stage 1 — cellular neighbourhood discovery

### `prepare_schurch.py`
Converts the Schürch colorectal CODEX table into the tidy schema stage 1
expects (`image_id, cell_id, x_um, y_um, cell_type`).

The source stores coordinates in **image pixels** on a 1920×1440 grid; stage 1
requires **microns**. Conversion uses the 377.44 nm/pixel lateral resolution
documented for that collection. The scale is then checked against the data
itself, since a wrong scale still yields clean-looking clusters: median
nearest-neighbour spacing lands at 6.8 µm, consistent with packed tissue.
Cells labelled as an imaging-artefact class are excluded.

```bash
python scripts/prepare_schurch.py [--mpp 0.37744] [--keep-artefacts]
```
**Writes** `data/interim/schurch_crc_cells.parquet`

### `run_stage1_cn.py`
Builds 40 µm neighbourhoods, decomposes composition into latent topics, sweeps
k over the configured range scoring silhouette, Davies–Bouldin and adjacent-k
adjusted Rand index, then assigns neighbourhoods at the selected k.

All three selection criteria are reported rather than one being optimised;
they routinely disagree, and the choice of k is a judgement to be made with the
table in view.

```bash
python scripts/run_stage1_cn.py --config config/crc_train_brca_apply.yaml \
    --cells data/interim/schurch_crc_cells.parquet [--skip-sweep] [--force]
```
**Writes** `data/processed/{cn_assignments.parquet, k_sweep.csv, cn_lineage_enrichment.csv}`

### `validate_cn_vs_published.py`
External validation with no extra data: the source table ships the original
authors' own neighbourhood assignments, so the rediscovered ones are scored
against them by adjusted Rand index and normalised mutual information, with a
contingency table naming each correspondence.

Agreement is partial by construction — the published method built windows from
the 10 nearest neighbours, this one uses a fixed 40 µm radius plus a topic
decomposition — so the comparison asks whether comparable structure is
recovered, not whether the original code was reproduced.

```bash
python scripts/validate_cn_vs_published.py
```
**Writes** `results/tables/T11_*.csv`

---

## Stage 2 — paired cohort, registration and label transfer

### `download_orion.py`
Fetches the registered H&E and single-cell table for each requested specimen,
verifying every file against the S3 listing and skipping anything already
complete. Deliberately **does not** fetch the 19-channel multiplex OME-TIFFs:
they are 44–147 GB each and nothing downstream reads them.

```bash
python scripts/download_orion.py --specimens CRC01 CRC02 CRC03
```
**Writes** `data/raw/orion_crc/<specimen>/`

### `run_orion_registration_qc.py`
Verifies the immunofluorescence-to-H&E registration rather than trusting it.
Orion images both modalities from one section and ships the H&E pre-registered,
so the expected transform is the identity — but a silent half-cell offset would
corrupt every patch label while still looking plausible.

Builds a tissue mask from the H&E, bins cell centroids onto the same grid, and
locates the FFT cross-correlation peak. A zero-pixel peak is reported as a
**bound** (`< one pixel at the level used`), not as a false-precision zero.
The check is global and rigid: it bounds translation and would not detect local
warping.

```bash
python scripts/run_orion_registration_qc.py [--level 3] [--sample 400000]
```
**Writes** `results/orion_registration_qc.json`, `figures/F6_registration_qc.*`

### `run_orion_cohort.py`
Derives **one shared habitat taxonomy** across every specimen, then applies it
to each. Per-specimen derivation would give each slide a private label set,
making "CN03" mean something different on every slide and the downstream
classifier meaningless.

Two concessions to cohort scale (~15 M cells), both stated in the script: the
topic model and k-means are fitted on a pooled random subsample and then applied
to every cell, and the k sweep is skipped because k was already selected in
stage 1.

Patch labels follow the CANVAS purity rules and are capped per specimen as a
**contiguous lattice block**, not a random subsample: the grid and graph context
encoders read spatial neighbours off the lattice, and random thinning would
punch holes in exactly the signal the benchmark measures.

```bash
python scripts/run_orion_cohort.py [--max-patches 2000] [--force]
```
**Writes** `data/interim/orion_cohort/{cohort_cn_assignments,cohort_patch_labels}.parquet`

### `extract_orion_patches.py`
Reads real H&E patches through a zarr view of level 0, so the full-resolution
image (13.5 GB for one specimen) is never materialised. Patches are grouped by
**dominant gated lineage** rather than transferred habitat, which makes the
figure a visual audit of the marker gating: tumour epithelium and smooth muscle
are checkable by eye in H&E.

```bash
python scripts/extract_orion_patches.py [--per-group 4] [--min-cells 25]
```
**Writes** `figures/F7_patch_labels.*`, `data/interim/orion_patches/`

### `run_orion_label_transfer.py`
Cross-platform transfer of a CN taxonomy from one cohort onto another. **This
approach is retained as a documented negative result.** Mapping a 56-plex
taxonomy onto a 16-plex panel requires collapsing both onto the lineages the
thinner panel can express, which leaves one neighbourhood with a featureless
profile that acts as a nearest-neighbour attractor and absorbs most of the
tissue. `run_orion_cohort.py` supersedes it.

### `run_stage2_pair.py`
Generic runner for registration plus label transfer across several samples,
wiring the tested `assign_cn_to_he_nuclei` and `label_patches` functions and
performing the patient-level split. A per-sample failure is skipped rather than
aborting the batch, and partial results are saved if the split cannot be formed.

---

## Stage 3 — embeddings and the habitat classifier

### `encode_orion_patches.py`
Encodes labelled patches to cached embeddings in the exact schema the benchmark
reads (`slide_id, x_um, y_um, patch_x, patch_y, label, emb_*`). Patches are read
through zarr, stain-normalised by the Macenko method, then passed through the
encoder **with that encoder's own preprocessing statistics**; substituting a
generic transform degrades embeddings silently rather than raising. Resumable
per specimen.

```bash
python scripts/encode_orion_patches.py --encoder phikon [--batch-size 32]
```
**Writes** `data/interim/orion_embeddings/<specimen>.parquet`

### `run_stage3_extract.py`
Equivalent for slides readable by OpenSlide (for example TCGA SVS): tissue
segmentation, non-overlapping tiling at the configured resolution, artefact
filtering, stain normalisation and streaming encoding to resumable per-slide
shards.

```bash
python scripts/run_stage3_extract.py --slides "data/raw/tcga/*.svs" --encoder phikon
```

### `run_final_benchmark.py`
The controlled ablation. `none` is CANVAS exactly; `graph`, `grid2d` and
`grid3d` differ only in how neighbourhood evidence reaches the same
classification head. Splits at **slide level** — a patch-level split combined
with spatial context places neighbouring patches on both sides of the split and
makes the resulting accuracy meaningless.

Use at least six seeds: a paired signed-rank test cannot fall below p = 0.0312
with six pairs, and with three the floor is 0.25.

```bash
python scripts/run_final_benchmark.py \
    --embeddings data/interim/orion_embeddings \
    --modes none graph grid2d grid3d --seeds 1 2 3 4 5 6 --epochs 30 --window 7
```
**Writes** `results/{final_benchmark,training_curves,per_class_metrics,confusion_matrices}.csv`

---

## Stage 4 — whole-slide inference

### `run_stage4_infer.py`
Filters to diagnostic slides, validates each slide's native resolution rather
than assuming it, caches embeddings, and — only when a trained head is supplied
— predicts habitats and assigns tumour-bulk and leading-edge compartments.
Without a checkpoint it stops after embedding and says so, rather than
fabricating compartments.

```bash
python scripts/run_stage4_infer.py --slides "data/raw/tcga_coad/*.svs" \
    [--habitat-model models/head.pt --tumour-habitat-idx 1]
```

---

## Stages 5 and 6 — spatial features and clinical modelling

### `run_stage5_features.py`
Computes the 262 features per sample: composition (10), diversity (6),
dispersion (90), interaction (100), distance (55) and transition entropy (1).
The total is asserted at runtime so a block producing the wrong count fails
loudly instead of quietly changing the feature space. `--simulate` generates
random habitat maps to exercise the machinery before real predictions exist.

```bash
python scripts/run_stage5_features.py --simulate --n-samples 200
python scripts/run_stage5_features.py --habitats data/processed/habitats.parquet
```

### `run_stage6_clinical.py`
Univariate Cox with Benjamini–Hochberg correction, consensus clustering into
ecotypes, and collinearity reduction, optionally within clinical strata.

```bash
python scripts/run_stage6_clinical.py --features <features.parquet> \
    --clinical <clinical.csv> --endpoint OS [--stratify PAM50]
```

### `validate_stage6_null.py`
Runs the whole stage 6 chain against an **independently random** survival
outcome. Every statistic should come out null; anything significant indicates a
bug rather than a finding.

This check found a real one. Univariate testing behaves correctly, but the
terminal signature model returns concordance indices of 0.60–0.80 on pure noise,
decreasing as sample size grows — in-sample optimism, because feature selection
and model evaluation share the same subjects. Any reported C-index must
therefore come from held-out data or carry an explicit optimism correction.

```bash
python scripts/validate_stage6_null.py --features data/interim/sim_features.parquet
```

---

## Reporting

### `run_report.py`
Regenerates all figures and tables from cached artefacts and assembles the
report in markdown, self-contained HTML and Word. Trains nothing and re-encodes
nothing. Raises if any generated figure or table is not referenced by a section,
so nothing can be silently dropped.

```bash
python scripts/run_report.py
```
**Writes** `figures/`, `results/tables/`, `reports/analysis_report.{md,html,docx}`

### `run_manuscript.py`
Builds the manuscript draft with figures embedded. Sections requiring data that
does not exist are emitted as explicit `[RESULTS PENDING]` markers naming the
file needed, shown in red in the Word version, so an incomplete claim cannot be
mistaken for a finished one. References that could not be verified against
PubMed are flagged in the same way.

```bash
python scripts/run_manuscript.py
```
**Writes** `reports/manuscript.{md,docx}`
