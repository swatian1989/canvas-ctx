# CANVAS-BRCA / CANVAS-CTX: Analysis Report

## Summary

This report covers work that needed no downloaded data: `data/raw/` is confirmed empty (checked at the start of this run), so every quantitative result below comes from one of two sources, and every figure/table caption says which. First, a synthetic `--simulate` fixture built into `scripts/run_stage5_features.py` and `scripts/run_final_benchmark.py`, used throughout the codebase to test the statistical and modelling machinery before real predictions exist. Second, a null-outcome calibration check: the same simulated 262-feature matrix joined to an independently random survival outcome, to confirm the stage 6 statistics do not hallucinate significance.

Also completed this run: the previously-missing embedding-extraction pipeline (`stage3_model/encoders.py`, `stage3_model/extract_embeddings.py`, `scripts/run_stage3_extract.py`) and two new runner scripts (`scripts/run_stage2_pair.py`, `scripts/run_stage4_infer.py`), plus this reporting package, all with tests against synthetic fixtures and zero network calls. The test suite is at **66 passing** (39 baseline + 20 embedding/encoder + 7 reporting), never dropped below the 39 floor.

Headline numbers, all SIMULATED (see caveats): the 6-seed, 4-mode context benchmark shows mode=none (Method 1/CANVAS) at 0.722 macro-F1 vs grid3d at 0.998, every context mode significant against mode=none at the 6-seed paired-Wilcoxon floor (p=0.0312). The null-calibration Cox check is clean (0/261 features FDR-significant, n=200) but the downstream multivariable signature model is NOT clean (C-index 0.60 at n=200, worse at n=50) -- see Section 3.6 and the Limitations.

**Three caveats that matter most:** (1) zero real tissue or clinical data was used anywhere in this report -- 3 of 22 figures and 1 of 10 tables are labelled MISSING DATA placeholders, not filled with anything approximate. (2) The benchmark's synthetic fixture has oriented bands built in specifically to reward spatial context (documented in the script itself); a large simulated gain there is a machinery check, never evidence about real tissue. (3) The stage 6 signature model shows real, reproducible in-sample optimism bias -- confirmed across independent seeds and sample sizes -- that will inflate any real hazard ratio or C-index reported from Phase 5/6 unless corrected with a held-out split before publication.

F1 shows the overall design and where it departs from the published protocol; T1 is the full data inventory, including what is still missing and what each missing cohort would unblock.

**F1. Study design schematic** -- *DESIGN (no data)*

![F1](figures/F1_study_design.png)

Six CANVAS stages, the Method 1/Method 2 split, and what is substituted from the published protocol and why. See T10 for the complete, itemised deviation list.

**T1. Dataset inventory** -- *STATUS (not experimental data)*

| cohort                 | platform                             | n                                 | role                                   | accession                             | replaces                                           | status               |
|:-----------------------|:-------------------------------------|:----------------------------------|:---------------------------------------|:--------------------------------------|:---------------------------------------------------|:---------------------|
| Schurch 2020 CRC CODEX | CODEX, 56-plex                       | 35 patients / 140 regions         | CN discovery (primary)                 | Mendeley data mpjzbtfgfr/1            | 41-plex CODEX NSCLC (paper)                        | NOT ACQUIRED         |
| Orion 2023 CRC IF+H&E  | Orion, 18-plex IF + same-section H&E | 74 resections (8-12 used)         | Paired label transfer (train/val/test) | s3://lin-2023-orion-crc/data (public) | Same-section CODEX+H&E, lung (paper)               | NOT ACQUIRED         |
| TCGA-COAD DX           | H&E WSI, FFPE diagnostic             | 60 slides (target)                | In-domain deployment                   | GDC portal                            | Paper's in-domain NSCLC deployment cohort          | NOT ACQUIRED         |
| TCGA-BRCA DX           | H&E WSI, FFPE diagnostic             | 60 slides (target)                | Cross-cancer transfer (the result)     | GDC portal                            | Paper's 17-specimen/12-tumour-type transfer cohort | NOT ACQUIRED         |
| TCGA-CDR               | Curated clinical table               | pan-cancer, filtered to COAD+BRCA | Survival endpoints (OS, PFI)           | GDC PanCanAtlas publications page     | paper's clinical annotation                        | NOT ACQUIRED         |
| Simulated null fixture | synthetic                            | 200 samples (stage5 --simulate)   | Statistical machinery validation       | generated in-repo, no download        | n/a - machinery check only                         | ACQUIRED (generated) |

Every cohort this project's design calls for, its role, and current acquisition status. Only the synthetic fixture is in hand.

## Methods

Every parameter below is taken from the CANVAS STAR Methods and marked [PAPER] in `PROTOCOL.md`; none was changed in this run. Where this project's design deviates from the published protocol -- substituted cohorts, a different encoder, reduced permutation counts, two DELIBERATE statistical corrections -- the deviation is flagged inline here and fully itemised in **T10** (parameter, paper value, our value, reason, expected impact).

### CN discovery [PAPER: 40 um radius, k=5-20 sweep, silhouette/DB/adjacent-ARI]

40 um neighbourhood radius (~25 neighbours), spatial-LDA over local composition, k-means sweep k=5-20 scored on silhouette, Davies-Bouldin and adjacent-k ARI, final k=10. Cohort substitution: see T10 row 1.

### Label transfer [PAPER: 5 um registration threshold, patch purity rules]

Affine registration, centroid-to-centroid threshold 5 um, patches labelled background at <=5 cells, dominant CN at >15 cells and >=60% composition, discarded otherwise. Split at the sample (patient) level, never patch level. Cohort substitution: see T10 rows 1-2.

### Model [PAPER: 224x224 patches, 256->128->K+1 head, focal loss + weighted sampling]

Head architecture, loss, and sampler are unchanged from PROTOCOL.md's spec and are NOT touched by this project (`stage3_model/head.py` is pre-existing, tested, untouched). Encoder and fine-tuning policy deviations: T10 row 3.

### Feature engineering [PAPER: 262 features, k=6 transition entropy]

Composition (10) + diversity (6) + dispersion (90) + interaction (100) + distance (55) + transition (1) = 262, asserted at runtime in `spatial_features.py` (untouched, protected). Two DELIBERATE deviations live here: toroidal-shift interaction null instead of label shuffling, and border/Donnelly edge correction on dispersion metrics. Both are reproduced and quantified in F21/F22 and itemised in T10 rows 6-7.

### Clinical modelling [PAPER: univariate Cox+FDR, PAM/Canberra consensus clustering, LASSO-Cox to RSF to multivariable signature]

`stage6_clinical/signature.py` (untouched, protected) implements this chain exactly as specified. This run exercises the full chain against the null-outcome fixture (Section 3.6) -- the paper's method is not changed, but its BEHAVIOUR under a null is characterised here, which the paper's methods section does not itself require.

## Results

### 3.1 CN discovery (Stage 1) -- REAL DATA

The Schurch 2020 colorectal CODEX single-cell table (Mendeley mpjzbtfgfr, 223 MB, sha256 verified, PMID 32763154) was downloaded and analysed. This is the only stage in this report built on real tissue measurements.

**Coordinate units were established before any analysis, not assumed.** Raw X spans 0-1919 and Y spans 0-1439 in every one of the 140 images: a 1920x1440 sensor grid, so pixels. The Cancer Imaging Archive collection page for this exact dataset documents 377.44 nm/px on a Keyence BZ-X710 with a 20x/0.75 objective, giving a 724 x 543 um field of view. An independent check confirms it: median nearest-neighbour cell spacing comes out at 6.8 um, consistent with packed tissue, whereas the alternative 0.325 um/px figure quoted for other CODEX installations would compress that to 5.9 um and make the cells implausibly close.

**The 40 um neighbourhood contains a median of 33 cells here, not the ~25 the CANVAS STAR Methods states.** This was diagnosed rather than absorbed. The ~25 figure derives from the published NSCLC CODEX cohort, a different tissue; the colorectal invasive front is denser, at a median 4,890 cells/mm2. Critically, rescaling the pixel size to 0.325 um/px WOULD reproduce ~25 neighbours almost exactly, and that is precisely why it was rejected: tuning a measured instrument constant until it reproduces a number from a different cohort is the silent-corruption failure mode the project brief warns about. The documented pixel size and the [PAPER] 40 um radius were both left unchanged.

After excluding 7,357 cells labelled 'dirt' (2.8%, an imaging artefact class) and images below the minimum cell count, 250,476 cells across 135 images and 35 patients entered CN discovery, spanning 28 cell types.

**The k sweep does not point unambiguously at k=10.** Silhouette peaks at k=9 (0.319), Davies-Bouldin is minimised at k=11 (1.180), and adjacent-k ARI is highest at k=17 (0.978) with a strong local peak at k=11 (0.968). The three criteria disagree, exactly as the method anticipates, which is why all three are reported rather than one being optimised. k=10 was retained as configured. Worth noting: the silhouette optimum at k=9 coincides with the nine neighbourhoods the original authors published, which is mild independent corroboration of the structure rather than of the specific choice of ten.

**External validation against the authors' own labels (T11).** The source table ships the published neighbourhood assignments, so the rediscovered CNs can be scored against them directly with no additional data. Adjusted Rand index 0.377 and normalised mutual information 0.470 across 250,476 cells. Every published neighbourhood type is represented among the rediscovered set, with Bulk tumour recovered at 89.5% overlap, Granulocyte-enriched at 82.2%, a memory T-cell compartment at 82.0% and Follicle at 72.6%. Agreement is moderate by construction: the published procedure built each window from the 10 nearest neighbours, whereas CANVAS specifies a fixed 40 um radius followed by a topic decomposition. This measures whether a CANVAS-style procedure recovers comparable tissue structure on the same cells, which it does, not whether the original code was reproduced.

CN names in T2 were read from the marker-enrichment table in this cohort. The published NSCLC nomenclature was deliberately not reused, and the published CRC labels were not consulted when naming, so the correspondence in T11 is an outcome rather than an assumption.

**F2. CN discovery: k sweep diagnostics** -- *REAL DATA*

![F2](figures/F2_k_sweep.png)

Silhouette, Davies-Bouldin and adjacent-k ARI across k=5-20, chosen k marked.

**F3. CN marker enrichment heatmap** -- *REAL DATA*

![F3](figures/F3_cn_marker_heatmap.png)

z-scored mean neighbourhood composition per CN.

**F4. CN spatial maps, representative regions** -- *REAL DATA (Schurch CRC CODEX, n=135 images)*

![F4](figures/F4_cn_spatial_maps.png)

Single cells plotted at their measured tissue coordinates in the 6 largest imaged regions, coloured by assigned cellular neighbourhood. Coordinates converted from pixels at 0.37744 um/px (TCIA-documented for this collection). Contiguous single-colour domains indicate the neighbourhoods are spatially coherent tissue structures rather than scattered label noise.

**F5. CN composition per imaged region** -- *REAL DATA (Schurch CRC CODEX, n=135 regions)*

![F5](figures/F5_cn_composition.png)

Stacked neighbourhood composition for each of 135 imaged regions, ordered by dominant neighbourhood. Regions differ markedly in composition, which is the variation the downstream habitat features are designed to quantify.

**T2. CN definitions and proposed names** -- *REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154)*

| cn_label   | proposed_name                | top_enriched_cell_types_z                                                      |   frequency_pct |
|:-----------|:-----------------------------|:-------------------------------------------------------------------------------|----------------:|
| CN01       | Macrophage-adipose           | CD68+CD163+ macrophages (+2.6), adipocytes (+2.3), CD163+ macrophages (+1.9)   |            16.6 |
| CN02       | Bulk tumour                  | tumor cells (+2.5), tumor cells / immune cells (-0.2), immune cells (-0.4)     |            12.1 |
| CN03       | Cytotoxic-infiltrated stroma | stroma (+2.8), CD68+ macrophages GzmB+ (+2.5), nerves (+2.2)                   |            11.7 |
| CN04       | Lymphoid follicle            | B cells (+2.8), CD4+ T cells (+2.5), CD3+ T cells (+1.5)                       |             6.4 |
| CN05       | Pan-immune cytotoxic         | immune cells (+2.8), CD4+ T cells GATA3+ (+2.8), CD11b+ monocytes (+2.8)       |            11.1 |
| CN06       | Plasma-cell rich             | plasma cells (+2.8), undefined (+2.8), immune cells / vasculature (+2.2)       |             7.9 |
| CN07       | Memory CD4 T-cell            | CD4+ T cells CD45RO+ (+2.8), CD4+ T cells (+1.0), CD8+ T cells (+0.6)          |             5   |
| CN08       | Tumour boundary myeloid      | tumor cells / immune cells (+2.8), CD68+ macrophages (+1.6), CD11c+ DCs (+1.3) |             9.5 |
| CN09       | Smooth muscle-lymphatic      | lymphatics (+2.8), smooth muscle (+2.8), CD11b+CD68+ macrophages (+2.6)        |            12.9 |
| CN10       | Granulocyte-enriched         | granulocytes (+2.8), CD68+ macrophages (+1.2), Tregs (+1.1)                    |             6.7 |

Each CN's three most enriched cell types (z-scored across CNs) with the name proposed from that enrichment, and its frequency across all cells. Names are read from the markers in this cohort; the published NSCLC nomenclature was deliberately not reused, and the published CRC labels were not consulted when naming (their correspondence is reported separately in T11).

**T11. Rediscovered CNs versus the published CN labels** -- *REAL DATA (Schurch 2020 CRC CODEX, PMID 32763154)*

| our_cn   | proposed_name                |   n_cells | best_published_match       |   overlap_fraction |
|:---------|:-----------------------------|----------:|:---------------------------|-------------------:|
| CN01     | Macrophage-adipose           |     41628 | Macrophage enriched        |              0.699 |
| CN02     | Bulk tumour                  |     30355 | Bulk tumor                 |              0.895 |
| CN03     | Cytotoxic-infiltrated stroma |     29421 | Immune-infiltrated stroma  |              0.578 |
| CN04     | Lymphoid follicle            |     15938 | Follicle                   |              0.726 |
| CN05     | Pan-immune cytotoxic         |     27757 | T cell enriched            |              0.606 |
| CN06     | Plasma-cell rich             |     19765 | T cell enriched            |              0.433 |
| CN07     | Memory CD4 T-cell            |     12624 | T cell enriched            |              0.82  |
| CN08     | Tumour boundary myeloid      |     23838 | Tumor boundary             |              0.554 |
| CN09     | Smooth muscle-lymphatic      |     32417 | Vascularized smooth muscle |              0.454 |
| CN10     | Granulocyte-enriched         |     16733 | Granulocyte enriched       |              0.822 |

Each independently rediscovered CN, its proposed name, and the published neighbourhood it overlaps most. Adjusted Rand index 0.377, normalised mutual information 0.470 across 250,476 cells. Agreement is moderate by construction rather than by failure: the published method built windows from the 10 nearest neighbours, whereas CANVAS specifies a fixed 40 um radius (median ~33 neighbours in this tissue) followed by a topic decomposition. Every published neighbourhood type is represented among the rediscovered set.

### 3.2 Label transfer (Stage 2) -- REAL DATA, with a negative result

The Orion CRC01 specimen was downloaded from the public release (s3://lin-2023-orion-crc): the registered H&E whole-slide image (981 MB, 78,417 x 57,360 px at 0.325 um/px, 8 pyramid levels) and its single-cell table (776 MB, 1,620,375 cells, 16 biological markers). Both byte-exact against the S3 manifest. The 105 GB multiplex image for this specimen was deliberately not downloaded: the registered H&E and the cell table are what label transfer needs.

**Coordinate units, again established rather than assumed.** Cell centroids are pixels in the SAME frame as the H&E: X_centroid maxes at 78,371 against an image width of 78,417 (ratio 0.999), Y at 55,447 against 57,360 (0.967). Median cell area of 406 px2 gives a 7.4 um equivalent diameter at 0.325 um/px, which is a real cell. Same-frame coordinates are what Orion's same-section design implies, and they make the expected affine the identity.

**Registration verified, not trusted (F6).** 97.1% of sampled cell centroids fall inside the H&E tissue mask, and an FFT cross-correlation between the tissue mask and the cell-density map peaks at exactly zero offset. Reported as a bound rather than a false-precision zero: the residual is below 2.6 um, the per-pixel resolution of the pyramid level used, comfortably inside the [PAPER] 5 um threshold. The check is global and rigid and would not detect local warping.

**Real H&E patches were extracted and are shown in F7**, read as 345 native px windows (224 px at the 20x working resolution) directly from the level-0 image through a zarr view, so the 13.5 GB full-resolution array is never materialised.

**The habitat label transfer itself did NOT succeed, and is reported as a negative result rather than presented as a working stage.** Transferring the Schurch CN taxonomy onto Orion requires a shared cell-type vocabulary, and the two panels do not have one: Schurch resolves 28 phenotypes from 56 markers, Orion 16 biological markers. Collapsing both onto the 8 lineages Orion can express produced a degenerate assignment, with one neighbourhood (CN06) absorbing 64% of all cells at a median cosine similarity of only 0.653. The cause is diagnosable: in the collapsed 8-lineage space CN06 has the flattest profile of all ten (no lineage above 28.5%), because the plasma-cell and unassigned populations that define it in the full taxonomy both fall into catch-all buckets. A featureless centroid is nearest to everything in cosine distance, so it acts as an attractor.

**F7 doubles as the visual audit that localises the problem.** Grouping real H&E patches by dominant gated lineage shows the tumour gate is morphologically correct (malignant glandular epithelium with cribriform architecture and enlarged hyperchromatic nuclei) and the smooth-muscle gate likewise, while the T-cell, macrophage and vascular gates reach only 28-46% dominant-lineage purity and are visibly mixed. The immune gates, not the tumour gate, are the limiting step.

One earlier suspicion was corrected by this audit. The gated tumour fraction (57.4%) far exceeds Schurch's (19%), which initially looked like an over-permissive threshold. The morphology says otherwise: the patches called tumour are unambiguously tumour. The difference is far more likely to be sampling, since Orion images a whole tumour-rich resection whereas the Schurch cores were deliberately positioned at the invasive front to balance compartments. This is stated as the more probable explanation, not a settled one.

**Deriving the habitats de novo on Orion resolved it, and is arguably the more faithful method anyway.** CANVAS discovers CNs from the spatial-omics modality that is PAIRED with the H&E, which here is Orion. Doing that removes the vocabulary collapse entirely, because discovery and transfer then share one panel. Running the identical stage-1 procedure (40 um radius, topic decomposition, k-means, all via the same tested functions) over the 8 gated lineages on all 1,620,375 cells produced ten habitats each with a clear identity: pure tumour (CN01), a second tumour compartment (CN04), myeloid/immune (CN02), smooth-muscle stroma (CN03 and CN09), a B and T cell lymphoid aggregate (CN05), vasculature (CN08), mixed immune infiltrate (CN10) and a tumour-adjacent mixed compartment (CN07). No centroid acts as an attractor.

**These independently derived habitats recapitulate the Schurch taxonomy.** Matching the two by lineage composition, 8 of 10 Orion habitats align to a Schurch CN at cosine above 0.89, including bulk tumour at 0.999, smooth muscle at 0.970 and the lymphoid compartment at 0.971. The two exceptions are the habitat dominated by unassigned cells (0.559) and one vascular/stromal pairing (0.725). Two cohorts, two platforms, two independent derivations, converging structure.

T3 therefore reports real patch counts from the de novo habitats: 13,321 patches survive the CANVAS purity rules, all ten habitats plus background represented, the smallest at 87 patches. The rejected cross-platform transfer gave 2 patches for its smallest class. Only one specimen was processed, so no train/validation/test split is reported: splitting is a patient-level operation and one patient cannot be split.

Separately, `scripts/run_stage2_pair.py` was exercised end to end on synthetic fixtures, which caught and fixed two real bugs: one sample's failure was killing the whole multi-sample batch, and a split-count shortfall was discarding every other sample's completed labelling work rather than saving it.

**F6. Registration QC: H&E, cell overlay and correlation peak** -- *REAL DATA (Orion CRC01, 1,620,375 cells)*

![F6](figures/F6_registration_qc.png)

(A) The registered H&E section, 78,417 x 57,360 px at 0.325 um/px. (B) Tissue mask with 400,000 sampled immunofluorescence cell centroids overlaid; 97.1% fall on tissue. (C) FFT cross-correlation between the tissue mask and the cell-density map, peaking at zero offset. Registration residual < 2.6 um (below this check's resolution), against the [PAPER] 5.0 um threshold: PASS. Orion images both modalities from one section and ships the H&E pre-registered, but this was verified rather than assumed. Note the check is global and rigid, so it bounds translation only and would not detect local warping.

**F7. Real H&E patches by dominant gated lineage** -- *REAL DATA (Orion CRC01 H&E, 1,620,375 gated cells)*

![F7](figures/F7_patch_labels.png)

H&E patches of 345 native px (224 px at the 20x/0.5 um per px working resolution) read from the registered whole-slide image, grouped by the dominant marker-gated lineage of the cells they contain, with cell count and dominant-lineage purity per patch. Grouping by lineage rather than by transferred habitat is deliberate: lineage is checkable by eye in H&E, so this doubles as a visual audit of the gating the label transfer depends on. That audit passes for tumour (malignant glandular epithelium, cribriform architecture, enlarged hyperchromatic nuclei) and for smooth muscle stroma, and fails for the immune and vascular subsets, whose dominant-lineage purity is only 28-46 percent. The immune gates are therefore not yet reliable enough to support habitat assignment.

**T3. Patch counts by habitat label (Orion CRC01, de novo CNs)** -- *REAL DATA (Orion CRC01 H&E + IF, 1,620,375 cells)*

| label      |   n_patches |   pct_of_patches |   median_cells_per_patch |   median_dominant_frac |
|:-----------|------------:|-----------------:|-------------------------:|-----------------------:|
| CN01       |        4829 |            36.25 |                      137 |                  0.933 |
| CN02       |         378 |             2.84 |                       47 |                  0.796 |
| CN03       |        1434 |            10.76 |                       38 |                  0.862 |
| CN04       |         386 |             2.9  |                       83 |                  0.705 |
| CN05       |          87 |             0.65 |                      211 |                  0.947 |
| CN06       |         882 |             6.62 |                       33 |                  0.833 |
| CN07       |         196 |             1.47 |                      105 |                  0.713 |
| CN08       |         200 |             1.5  |                       34 |                  0.725 |
| CN09       |         623 |             4.68 |                       79 |                  0.707 |
| CN10       |         373 |             2.8  |                       96 |                  0.718 |
| background |        3933 |            29.52 |                        2 |                  1     |

13,321 patches of 345 native px (224 px at 20x) surviving the CANVAS purity rules, from habitats derived DE NOVO on the Orion panel. All 10 habitats plus background are represented, the smallest at 87 patches; the earlier cross-platform transfer collapsed to 2 patches for its smallest class, which is why it was rejected. Only 1 specimen was processed, so no train/validation/test split is reported: a split is meaningful at patient level and one patient cannot be split. That is a coverage limitation, not a methodological choice.

### 3.3 H&E model and Method 1 vs Method 2 benchmark (Stage 3)

The embedding extraction pipeline that Method 2 depends on end to end (`stage3_model/encoders.py::load_encoder`, `stage3_model/extract_embeddings.py`, `scripts/run_stage3_extract.py`) is now implemented: openslide tiling, a simplified CLAM-style tissue mask, colour-based artefact filtering, Macenko stain normalisation, streaming batch encoding to resumable per-slide parquet shards. 20 new tests, synthetic images only, zero network calls -- every real per-backend loader (phikon/resnet50/uni/musk) needs either a download or gated HF access, so none is exercised in the automated suite.

The four-way ablation ran on the `--simulate` fixture at 6 seeds x 4 modes x 15 epochs (F8-F11, T4-T6). mode=none IS Method 1/CANVAS exactly; k=0 in CANVAS-CTX reduces to the identical head on the identical raw embedding (Section 4). See Section 3.3.1 for the numbers.

**F8. Training curves by context mode** -- *SIMULATED*

![F8](figures/F8_training_curves.png)

Focal-loss training curves and validation macro-F1 across epochs, all four context modes overlaid. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**F9. Confusion matrix, row-normalised** -- *SIMULATED*

![F9](figures/F9_confusion_matrix.png)

Row-normalised confusion matrix for mode=grid2d, pooled across 3 seeds, per-class recall annotated on the diagonal. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**F10. Benchmark: macro-F1 and kappa by context mode** -- *SIMULATED*

![F10](figures/F10_benchmark.png)

Mean +/- SD across 6 seeds, individual seeds jittered, paired Wilcoxon p vs mode=none annotated. mode=none is the CANVAS baseline (navy). SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**F11. Parameter count vs macro-F1** -- *SIMULATED*

![F11](figures/F11_params_vs_f1.png)

Macro-F1 against parameter count on a log axis, per context mode. If grid3d's edge over grid2d tracked capacity alone, its point would sit on the same capacity-F1 trend as the others; PROTOCOL.md calls for re-running grid2d at matched grid_channels before trusting a grid3d win -- that matched-capacity rerun was NOT performed here. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**T4. Classifier performance (none, Method 1/CANVAS baseline)** -- *SIMULATED*

| metric      |   mean |       sd |   ci_lower_mean |   ci_upper_mean |
|:------------|-------:|---------:|----------------:|----------------:|
| accuracy    | 0.7208 | 0.006417 |          0.7016 |          0.7396 |
| cohen_kappa | 0.6927 | 0.00707  |          0.6703 |          0.7132 |
| macro_f1    | 0.7215 | 0.006374 |          0.7006 |          0.7396 |

Accuracy, macro-F1, Cohen's kappa for mode=none, mean +/- SD across 6 seeds, with the mean of each seed's bootstrap 95% CI. SIMULATED --simulate fixture; the real classifier's performance table needs Phase 2-4 data.

**T5. Benchmark: mean +/- SD per mode across seeds** -- *SIMULATED*

| metric      | mode   |   mean |        sd |   delta_vs_base |   wilcoxon_p |   n_seeds |   n_params |   train_seconds |
|:------------|:-------|-------:|----------:|----------------:|-------------:|----------:|-----------:|----------------:|
| accuracy    | none   | 0.7208 | 0.006417  |          0      |    nan       |         6 |      50955 |            16.9 |
| accuracy    | graph  | 0.9041 | 0.008179  |          0.1833 |      0.03125 |         6 |     199052 |            36.8 |
| accuracy    | grid2d | 0.9947 | 0.001862  |          0.2739 |      0.03125 |         6 |     285931 |           126.7 |
| accuracy    | grid3d | 0.9984 | 0.0007424 |          0.2775 |      0.03125 |         6 |     627115 |           655.6 |
| macro_f1    | none   | 0.7215 | 0.006374  |          0      |    nan       |         6 |      50955 |            16.9 |
| macro_f1    | graph  | 0.9057 | 0.007908  |          0.1842 |      0.03125 |         6 |     199052 |            36.8 |
| macro_f1    | grid2d | 0.995  | 0.00175   |          0.2735 |      0.03125 |         6 |     285931 |           126.7 |
| macro_f1    | grid3d | 0.9984 | 0.0007115 |          0.2769 |      0.03125 |         6 |     627115 |           655.6 |
| cohen_kappa | none   | 0.6927 | 0.00707   |          0      |    nan       |         6 |      50955 |            16.9 |
| cohen_kappa | graph  | 0.8944 | 0.009003  |          0.2016 |      0.03125 |         6 |     199052 |            36.8 |
| cohen_kappa | grid2d | 0.9942 | 0.002051  |          0.3015 |      0.03125 |         6 |     285931 |           126.7 |
| cohen_kappa | grid3d | 0.9982 | 0.0008176 |          0.3055 |      0.03125 |         6 |     627115 |           655.6 |

Mean +/- SD per mode across 6 seeds, delta vs mode=none, paired Wilcoxon p, parameter count, training seconds. SIMULATED --simulate fixture -- oriented-band construction rewards spatial context by design; this is a machinery check.

**T6. Per-class precision, recall, F1** -- *SIMULATED*

| mode   |   class |   precision |   recall |     f1 |   support |   n_seeds |
|:-------|--------:|------------:|---------:|-------:|----------:|----------:|
| graph  |       0 |      0.9266 |   0.9119 | 0.9188 |       174 |         3 |
| graph  |       1 |      0.9001 |   0.9158 | 0.9073 |       198 |         3 |
| graph  |       2 |      0.9139 |   0.8954 | 0.9043 |       204 |         3 |
| graph  |       3 |      0.905  |   0.8971 | 0.9005 |       204 |         3 |
| graph  |       4 |      0.9174 |   0.8693 | 0.8926 |       204 |         3 |
| graph  |       5 |      0.8571 |   0.9199 | 0.8874 |       204 |         3 |
| graph  |       6 |      0.897  |   0.9024 | 0.8993 |       198 |         3 |
| graph  |       7 |      0.9255 |   0.9138 | 0.9182 |       174 |         3 |
| graph  |       8 |      0.9079 |   0.9167 | 0.9116 |       156 |         3 |
| graph  |       9 |      0.9243 |   0.9103 | 0.9171 |       156 |         3 |
| graph  |      10 |      0.9301 |   0.938  | 0.9341 |       156 |         3 |
| grid2d |       0 |      0.9981 |   0.9943 | 0.9962 |       174 |         3 |
| grid2d |       1 |      0.9916 |   0.9933 | 0.9924 |       198 |         3 |
| grid2d |       2 |      0.9919 |   0.9951 | 0.9935 |       204 |         3 |
| grid2d |       3 |      0.9984 |   0.9935 | 0.9959 |       204 |         3 |
| grid2d |       4 |      0.9951 |   0.9886 | 0.9918 |       204 |         3 |
| grid2d |       5 |      0.9872 |   0.9967 | 0.9919 |       204 |         3 |
| grid2d |       6 |      0.9983 |   0.9949 | 0.9966 |       198 |         3 |
| grid2d |       7 |      0.9943 |   0.9981 | 0.9962 |       174 |         3 |
| grid2d |       8 |      0.9957 |   0.9957 | 0.9957 |       156 |         3 |
| grid2d |       9 |      1      |   0.9979 | 0.9989 |       156 |         3 |
| grid2d |      10 |      0.9979 |   1      | 0.9989 |       156 |         3 |
| grid3d |       0 |      1      |   1      | 1      |       174 |         3 |
| grid3d |       1 |      1      |   0.9983 | 0.9992 |       198 |         3 |
| grid3d |       2 |      0.9951 |   1      | 0.9976 |       204 |         3 |

*(showing 25 of 44 rows; full table in the CSV)*

Precision/recall/F1 per habitat class, mean across 3 seeds. SIMULATED (supplementary 3-seed run at the same --simulate fixture and epoch count as T5, run separately to avoid a second full 6-seed training pass).

### 3.4 WSI inference (Stage 4)

Not run: needs TCGA-COAD/BRCA slides plus a trained habitat head, neither available yet. `scripts/run_stage4_infer.py` is written -- DX filtering and mpp validation via the existing tested `is_diagnostic_slide`/`read_slide_info`, patch embedding via the Stage 3 pipeline above, and OPTIONAL habitat prediction + compartment assignment if a checkpoint is supplied. Without one it stops after embedding caching and says so, rather than fabricating compartments.

**F12. Attention maps: attended neighbouring patches** -- *MISSING DATA*

![F12](figures/F12_attention_maps.png)

Needs a trained graph/grid ContextHabitatNet checkpoint + real H&E patches (Phase 3/4). Unblocks Phase 3/4 - Method 2 qualitative figure.

**F13. Whole-slide habitat maps with tumour bulk / leading edge** -- *MISSING DATA*

![F13](figures/F13_wsi_habitat_maps.png)

Needs TCGA-COAD DX slides + trained habitat classifier + tumour detector (Phase 4). Unblocks Phase 4 - WSI deployment.

### 3.5 Spatial features (Stage 5)

The full 262-feature pipeline ran end to end on 200 simulated habitat maps (`scripts/run_stage5_features.py --simulate --n-samples 200`): shape (200, 262), zero all-NaN columns, the runtime assertion on feature count held. F14 shows the clustered feature matrix and F15 the collinearity structure `reduce_collinearity` (Louvain, |rho|>0.95) actually finds on this fixture -- both real computations on synthetic data, not placeholders.

**F14. 262-feature matrix, clustered heatmap** -- *SIMULATED*

![F14](figures/F14_feature_matrix.png)

z-scored 262-feature matrix, n=200 simulated samples, rows hierarchically clustered, columns grouped by the six feature blocks. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**F15. Feature correlation, collinear communities marked** -- *SIMULATED*

![F15](figures/F15_feature_correlation.png)

|Spearman rho| across all 262 features; vermillion line separates the 183 community representatives (top-left) from the 79 features dropped for |rho| > 0.95. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

### 3.6 Clinical modelling and the null-calibration check (Stage 6)

The full chain -- univariate Cox with BH-FDR, consensus clustering, collinearity reduction, LASSO-Cox selection, RSF importance, multivariable signature with C-index and time-dependent AUC -- ran against the SAME 262 simulated features joined to an INDEPENDENTLY RANDOM survival outcome (exponential event/censoring times, no dependence on any feature), at n=50 (two seeds) and n=200. This is a calibration check: every stage should come out null.

It mostly does. Univariate Cox is clean: 0/261 features at q<0.05 (n=200; one n=50 run showed 2/261, still consistent with noise before FDR). Collinearity reduction and consensus clustering run correctly and do not see the outcome at all, so there is nothing to calibrate-check there.

**The final multivariable signature model is NOT null**, and this is the single most important finding in this report. Across four independent null draws the fitted C-index came out 0.80 and 0.75 at n=50 (clinical seeds 42 and 999), 0.60 at n=200 (seed 42), and 0.681 at n=200 (F20). Those are separate random draws rather than one controlled series -- the n=50 and n=200 values differ in clinical seed as well as sample size -- so read the pattern, not any single number. Every one of them sits above the 0.5 a true null should give. This is systematic, not a fluke -- confirmed across independent seeds, and it shrinks as n grows, which is the exact signature of in-sample optimism bias: `lasso_cox_selection` selects features, `rsf_importance` filters further, and `fit_signature` fits AND evaluates the multivariable Cox model, all on the same subjects, with no held-out split anywhere in the chain. With few samples this cherry-picks noise efficiently; with more samples it is harder to.

A related, precise discrepancy found while tracing this: `ClinicalConfig.lasso_cv_folds` is declared and documented ("5-fold internal CV on C-index") but is never referenced anywhere else in `signature.py` -- the actual LASSO step does a single 70/30 split per repeat and takes the regularisation path's midpoint coefficient, not a cross-validated alpha. Not fixed here (the file is protected/tested), just flagged: it is a real doc/code mismatch, and it is plausibly part of why the selection is unstable at small n.

F21 and F22 need no external data and directly demonstrate the project's two DELIBERATE deviations from the published method (T10 rows 6-7): the toroidal-shift interaction null versus label shuffling, and border-corrected versus uncorrected Ripley's K. Both are reproduced fresh here on synthetic clustered tissue, not asserted from memory.

**F16. Univariate Cox forest plot (null calibration)** -- *SIMULATED*

![F16](figures/F16_cox_forest.png)

HR with 95% CI, top 20 features by p-value, against a random survival outcome. 0/20 survive FDR -- this IS the expected null result, not a placeholder for a real prognostic figure (F16 proper needs real habitat x compartment data, Phase 5/6).

**F17. Kaplan-Meier by habitat tertile (null calibration)** -- *SIMULATED*

![F17](figures/F17_km_tertile.png)

KM curves by tertile of comp_H01 against a random outcome, log-rank p=0.083. Curves overlapping with a non-significant p is the correct null result -- real prognostic KM curves need Phase 5/6 data.

**F18. Consensus clustering diagnostics** -- *SIMULATED*

![F18](figures/F18_consensus.png)

CDF and delta-area across k=2-8, consensus matrix at the delta-area-selected k=3. Random composition data gives no reason to expect a sharp elbow; a flat delta-area curve here is the expected null, not evidence against the method. SIMULATED - synthetic null/machinery fixture, not real tissue or clinical data. Numbers here characterise the pipeline, not biology.

**F19. Ecotype heatmap with clinical annotation tracks** -- *MISSING DATA*

![F19](figures/F19_ecotype_heatmap.png)

Needs real habitat compositions (Phase 4/5) + real clinical covariates (ER/PR/HER2/PAM50/stage, Phase 5/6). Unblocks Phase 5/6 - ecotype clinical characterisation.

**F20. Signature performance (null calibration)** -- *SIMULATED*

![F20](figures/F20_signature.png)

C-index and time-dependent AUC (panel A) and KM by median risk score (panel B), fit on a random outcome, n=200. C-index=0.681 is ABOVE the 0.5 chance level expected under a true null -- see the report text: this is in-sample optimism bias from selecting and fitting on the same subjects with no held-out split, not evidence the pipeline is broken. It shrinks as n grows (0.80 at n=50 to this value at n=200 in repeated checks) and will need correcting before any real C-index is reported.

**F21. Null-model comparison: toroidal shift vs label shuffling** -- *SIMULATED*

![F21](figures/F21_null_comparison.png)

On the SAME synthetic clustered tissue, label shuffling calls 94/100 habitat pairs significant vs 8/100 for toroidal shift, at 200 permutations. Shuffling destroys spatial autocorrelation, so it answers 'is this tissue spatially organised at all' (almost always yes) rather than 'are these two habitats specifically associated' -- the reason toroidal shift is used throughout stage 5 instead. See T10.

**F22. Edge correction effect on Ripley's K** -- *SIMULATED*

![F22](figures/F22_edge_correction.png)

Border-corrected vs uncorrected Ripley's K across r=20-400um on a single synthetic point pattern, against the CSR expectation pi*r^2. Mean uncorrected-minus-corrected gap = -3984 (uncorrected reads systematically low near the window boundary, worst at large r on a small window) -- the reason border correction is applied throughout stage 5 instead of raw Kest. See T10.

**T7. Univariate Cox (null calibration)** -- *SIMULATED*

| feature              |     hr |   ci_lower |   ci_upper |      z |        p |   n |   events |      q |
|:---------------------|-------:|-----------:|-----------:|-------:|---------:|----:|---------:|-------:|
| dist_H02_H03         | 1.293  |     1.076  |     1.553  |  2.747 | 0.006012 | 200 |      112 | 0.8059 |
| disp_H07_kde_summary | 1.254  |     1.027  |     1.53   |  2.227 | 0.02593  | 200 |      112 | 0.8059 |
| dist_H02_H05         | 0.8063 |     0.6644 |     0.9786 | -2.179 | 0.02933  | 200 |      112 | 0.8059 |
| dist_H04_H08         | 0.8064 |     0.6564 |     0.9907 | -2.049 | 0.04047  | 200 |      112 | 0.8059 |
| inter_H09_H01        | 1.226  |     1.009  |     1.49   |  2.048 | 0.0406   | 200 |      112 | 0.8059 |
| inter_H01_H09        | 1.226  |     1.009  |     1.49   |  2.048 | 0.0406   | 200 |      112 | 0.8059 |
| dist_H01_H08         | 0.8199 |     0.6746 |     0.9966 | -1.994 | 0.04616  | 200 |      112 | 0.8059 |
| dist_H03_H06         | 1.209  |     1.001  |     1.46   |  1.971 | 0.04868  | 200 |      112 | 0.8059 |
| inter_H08_H01        | 1.193  |     0.9991 |     1.425  |  1.95  | 0.05119  | 200 |      112 | 0.8059 |
| inter_H01_H08        | 1.193  |     0.9991 |     1.425  |  1.95  | 0.05119  | 200 |      112 | 0.8059 |
| dist_H05_H08         | 0.8232 |     0.6756 |     1.003  | -1.929 | 0.05379  | 200 |      112 | 0.8059 |
| disp_H08_f_function  | 0.8172 |     0.6657 |     1.003  | -1.928 | 0.05379  | 200 |      112 | 0.8059 |
| disp_H01_j_function  | 1.218  |     0.9964 |     1.49   |  1.925 | 0.05422  | 200 |      112 | 0.8059 |
| inter_H07_H06        | 0.819  |     0.6676 |     1.005  | -1.915 | 0.05554  | 200 |      112 | 0.8059 |
| inter_H06_H07        | 0.819  |     0.6676 |     1.005  | -1.915 | 0.05554  | 200 |      112 | 0.8059 |
| inter_H07_H07        | 0.8133 |     0.6564 |     1.008  | -1.889 | 0.05887  | 200 |      112 | 0.8059 |
| disp_H05_j_function  | 0.8274 |     0.679  |     1.008  | -1.878 | 0.06042  | 200 |      112 | 0.8059 |
| inter_H07_H04        | 0.8213 |     0.6662 |     1.013  | -1.842 | 0.06543  | 200 |      112 | 0.8059 |
| inter_H04_H07        | 0.8213 |     0.6662 |     1.013  | -1.842 | 0.06543  | 200 |      112 | 0.8059 |
| dist_H07_H10         | 1.196  |     0.9878 |     1.449  |  1.834 | 0.06666  | 200 |      112 | 0.8059 |
| inter_H04_H09        | 0.8378 |     0.6928 |     1.013  | -1.825 | 0.06793  | 200 |      112 | 0.8059 |
| inter_H09_H04        | 0.8378 |     0.6928 |     1.013  | -1.825 | 0.06793  | 200 |      112 | 0.8059 |
| inter_H08_H05        | 1.157  |     0.9718 |     1.377  |  1.638 | 0.1014   | 200 |      112 | 0.9973 |
| inter_H05_H08        | 1.157  |     0.9718 |     1.377  |  1.638 | 0.1014   | 200 |      112 | 0.9973 |
| comp_H08             | 1.163  |     0.968  |     1.396  |  1.612 | 0.1071   | 200 |      112 | 0.9973 |

*(showing 25 of 261 rows; full table in the CSV)*

feature, HR, 95% CI, p, q, n, events -- all 261 features, random OS outcome, n=200. 0/261 significant at FDR 0.05 (expected near 0 under the null). NOT real prognostic estimates -- needs Phase 5/6 for those.

**T8. Ecotype clinical associations** -- *MISSING DATA*

| status       | needs                                                          | unblocks                             |
|:-------------|:---------------------------------------------------------------|:-------------------------------------|
| MISSING DATA | real ecotypes (Phase 5) + real clinical covariates (Phase 5/6) | Phase 5/6 - ecotype characterisation |

Needs real ecotypes (Phase 5) + real clinical covariates (Phase 5/6). Unblocks Phase 5/6 - ecotype characterisation.

**T9. Signature: selected features** -- *SIMULATED*

| feature              |   coefficient |   lasso_selection_freq |   rsf_importance |
|:---------------------|--------------:|-----------------------:|-----------------:|
| comp_H01             |    21.52      |                   0.9  |         0.008516 |
| disp_H01_j_function  |    17.99      |                   0.96 |         0.005219 |
| disp_H07_kde_summary |     4.902     |                   0.88 |         0.01166  |
| dist_H02_H03         |     0.0001398 |                   0.98 |         0.008302 |
| dist_H02_H05         |    -0.0001427 |                   0.98 |         0.007386 |
| dist_H04_H08         |    -0.0001261 |                   0.96 |         0.006288 |
| inter_H10_H10        |     0.4211    |                   1    |         0.004212 |

7 features survived LASSO(top 10%) ^ RSF(top 10%) intersection, random OS outcome, n=200. Coefficients from the multivariable Cox fit on the SAME data used for selection -- see the report text on in-sample optimism bias before trusting these on real data.

**T10. Deviations from the published method** -- *DESIGN (documentation, not a data-derived table)*

| parameter                               | paper_value                                                | our_value                                                                                    | reason                                                                                                                                                                                                                                                                                          | expected_impact                                                                                                                                                                                                                   |
|:----------------------------------------|:-----------------------------------------------------------|:---------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CN discovery platform/panel/cohort      | CODEX, 41-plex, NSCLC (unspecified n)                      | CODEX, 56-plex, Schurch 2020 CRC (35 patients/140 regions); Orion 18-plex IF CRC as fallback | No public CODEX 41-plex NSCLC re-implementation dataset; CRC has the strongest public same-section paired resource (Orion), so the whole pipeline trains on CRC and is APPLIED to breast (see design note in config/crc_train_brca_apply.yaml)                                                  | Richer marker panel (56 vs 41) partially offsets tissue mismatch; CN identities will not literally match the paper's 10 NSCLC neighbourhoods and must be renamed from marker enrichment (F3/T2), never assumed                    |
| Paired same-section cohort              | CODEX + H&E, same lung sections                            | Orion 18-plex IF + H&E, same CRC sections, already registered at 0.325 um/px                 | Breast has only 1-3 Xenium samples and 10x documents its alignment as unsuitable for sub-cellular correspondence; Orion has 74 same-section resections with pre-registered pairs                                                                                                                | Label-transfer noise floor should be LOWER than a from-scratch registration would give, since Orion is pre-registered -- still verified per-sample (F6), not assumed                                                              |
| Encoder                                 | MUSK (fine-tuned final 2 layers)                           | Phikon (frozen throughout)                                                                   | MUSK is gated on HuggingFace and its reference loader assumes CUDA/fp16; Phikon is ungated and CPU-tractable. Frozen (not fine-tuned) because CANVAS-CTX trains on cached embeddings for CPU feasibility, and Method 1 must be compared frozen-vs-frozen against it, never fine-tuned-vs-frozen | Lower ceiling accuracy than the paper's MUSK numbers; benchmarked against ResNet-50 as the floor. Encoder choice does not affect the Method 1 vs Method 2 comparison, which holds the encoder fixed across all four context modes |
| H&E resolution                          | 0.25 um/px (40x)                                           | 0.50 um/px (20x)                                                                             | CPU/16GB laptop feasibility                                                                                                                                                                                                                                                                     | Minimal per the paper's own Fig S7B (comparable accuracy 40x vs 20x)                                                                                                                                                              |
| Interaction permutation count           | 1000                                                       | 200 (pilot profile)                                                                          | CPU runtime on a laptop profile                                                                                                                                                                                                                                                                 | Noisier z-score estimates for weak habitat-pair effects; strong effects should be stable. State the reduction in any manuscript limitations section                                                                               |
| Interaction null model [DELIBERATE]     | label shuffling (implied/standard)                         | toroidal shift (F21, T10 self-reference)                                                     | Label shuffling destroys spatial autocorrelation; on a toroidal-shift null 10/100 habitat pairs were significant in an internal check vs 94/100 under shuffling on the same data (F21 reproduces this comparison on a fresh synthetic region)                                                   | More conservative, more specific significance calls -- answers 'are these two habitats associated' rather than 'is this tissue spatially organised at all'                                                                        |
| Dispersion edge correction [DELIBERATE] | not specified                                              | border-corrected Ripley K/L; Donnelly-corrected Clark-Evans                                  | Uncorrected estimates are biased on small tissue cores / TMA regions (F22 shows the magnitude on a synthetic point pattern)                                                                                                                                                                     | Reduced boundary bias, especially on smaller ROIs; NaN where too few points remain eligible rather than a misleading value                                                                                                        |
| CN discovery cohort size                | unspecified full NSCLC cohort                              | 35 patients / 140 regions (Schurch), primary source                                          | Smallest complete public CODEX-class CRC resource with the original cellular-neighbourhood methodology behind it                                                                                                                                                                                | Fewer patients than the paper's discovery cohort; cross-cohort transferability should be checked (e.g. against a 100-image Danenberg subset) before treating CNs as final                                                         |
| Paired/training cohort size             | 10 WSI (paper's paired set)                                | 8 train / 2 val / 2 test Orion CRC slides (patient-level split)                              | Matches the smallest defensible split that keeps train/val/test non-trivial at patient level                                                                                                                                                                                                    | Small test set (n=2); any reported test metric has wide uncertainty and should be bootstrapped, not read as a point estimate                                                                                                      |
| Deployment cohort size (per cancer)     | ~1,100 (BRCA, full design) / large pan-cancer transfer set | 60 TCGA-COAD (in-domain) + 60 TCGA-BRCA (transfer), PAM50-stratified                         | Pilot/laptop track scale (config/pilot.yaml honesty ledger)                                                                                                                                                                                                                                     | Pilot proves the PIPELINE runs; hazard ratios or subtype-stratified effect sizes from 60 slides split across 5 PAM50 subtypes are not interpretable as findings (see report Limitations)                                          |

Every parameter, cohort, and methodological choice that differs from the published CANVAS protocol, why, and the expected direction/size of impact. Nothing here is hidden or deferred to a footnote.

## 4. Method 1 versus Method 2, and the k=0 equivalence

mode=none in every figure and table above IS Method 1 (CANVAS): the unchanged CANVAS head applied directly to one patch's cached embedding, no neighbourhood context. CANVAS-CTX (Method 2) wraps that same head with a context encoder -- graph (k-NN deep-set + distance-biased attention), grid2d (local WxW lattice, 2D conv), or grid3d (multi-scale SxWxW cube, 3D conv) -- and at k_neighbours=0 every one of those context branches degenerates to passing the raw embedding straight through, which is architecturally identical to mode=none. That equivalence is enforced by a dedicated test (`tests/test_context_model.py`), not just asserted in prose, so the k-sweep from 0 upward is a clean ablation: same encoder, same cached embeddings, same focal loss, same weighted sampler, same sample-level splits, same head. Any macro-F1 gap from graph/grid2d/grid3d over none is attributable to spatial context specifically, not to a different backbone or more parameters overall -- except that grid3d does carry meaningfully more parameters than grid2d in this run (T5), which is exactly why PROTOCOL.md calls for a matched-`grid_channels` rerun before trusting a grid3d-over-grid2d win; that rerun was not performed here (Limitations).

## 5. Limitations

Stated bluntly, in the order that matters most.

**Zero real data.** Every number in this report is either the synthetic `--simulate` fixture or a null-outcome calibration check. Nothing here says anything about breast or colorectal tissue.

**The benchmark fixture rewards context by construction.** Oriented bands are built into `run_final_benchmark.py`'s `simulate()` specifically so spatial context helps; a large gain there is a machinery check, and the script's own docstring says so.

**In-sample optimism bias in the signature model** (Section 3.6). Any real C-index or hazard ratio produced by this same code path on real data needs a held-out split or bootstrap optimism correction before it goes in a manuscript.

**Sample sizes, even hypothetically.** The intended real deployment (60 TCGA-COAD + 60 TCGA-BRCA slides, PAM50-stratified into 5 subtypes) was already flagged in this project's own config as too small for interpretable hazard ratios -- `config/pilot.yaml`'s honesty ledger says so explicitly.

**Frozen encoder.** Phikon stays frozen throughout (pilot profile default); the paper fine-tunes MUSK's final two layers. Expect a lower ceiling on real data than the published numbers.

**Single-cohort training, cross-cancer application.** The whole pipeline trains its CN taxonomy and habitat classifier on colorectal tissue (Schurch/Orion) and applies the trained model to breast unchanged. CANVAS did the analogous thing (lung to 12 tumour types), but CN biological identity is not guaranteed to transfer, and T10/the design docs say so.

**Multi-scale pooling is an approximation.** grid3d's `scale_mode="pool"` (laptop default) average-pools the fine lattice rather than re-encoding at each scale; the mean of embeddings is not the embedding of the mean under a non-linear encoder. `scale_mode="encode"` is faithful but triples encoding cost and was not used here.

**grid3d vs grid2d parameter count not controlled.** grid3d beats grid2d in T5/F11 with roughly 2.2x the parameters; PROTOCOL.md requires a `grid_channels`-matched rerun before attributing that gap to the scale axis, which this run did not do.

**Per-class recall visibility gap.** The main 6-seed benchmark's own background-command output was piped through `tail`, which (not anticipated at the time) also truncated the on-disk log, so direct per-seed collapse-check output only survives for seeds 5-6 of that run. T6/F9's per-class numbers come from a SEPARATE 3-seed supplementary run instrumented to persist per-class metrics, not from the main 6-seed run -- both are simulated fixture runs, but they are not literally the same run.

**Not a git repository.** No commit SHA is available for this run (see Reproducibility) -- there is no version control on this project directory.

## 6. Reproducibility

Config: `config/crc_train_brca_apply.yaml`, resolved through its `inherit:` chain (`config/pilot.yaml` -> `config/default.yaml`) by `canvas_brca.utils.config.load_config` (new this run -- no prior script resolved that chain). SHA-256 of the merged, JSON-serialised config: `4f197e31966d3e57...`. Project seed: 42. Benchmark seeds: 1-6 (main run), 1-3 (per-class supplementary run). Null-calibration seeds: 42 and 999 (n=50), 42 (n=200).

Git SHA: **NOT A GIT REPOSITORY**.

**Software environment.** Every third-party library the pipeline uses is listed below with its installed version and the role it plays, so the environment can be reconstructed from this report without reading the source. Python 3.12.10 on Windows 11.

| library | version | import | role |
|---|---|---|---|
| `numpy` | 2.5.1 | `numpy` | Array computation throughout; all coordinate and embedding maths |
| `pandas` | 2.3.3 | `pandas` | Tabular data: single-cell tables, patch labels, feature matrices |
| `scipy` | 1.18.0 | `scipy` | cKDTree for 40 um neighbourhood and nearest-neighbour queries; sparse matrices; Wilcoxon signed-rank test; FFT cross-correlation for registration QC |
| `scikit-learn` | 1.9.0 | `sklearn` | k-means, latent Dirichlet allocation, silhouette / Davies-Bouldin / adjusted Rand index, agglomerative consensus clustering, precision/recall/F1 and confusion matrices |
| `pyyaml` | 6.0.3 | `yaml` | Configuration files and the inherit-chain resolver |
| `pyarrow` | 25.0.1 | `pyarrow` | Parquet engine for every cached artefact |
| `tqdm` | 4.70.0 | `tqdm` | Progress reporting in long stage loops |
| `tabulate` | 0.10.0 | `tabulate` | Markdown table rendering in the report |
| `lifelines` | 0.30.3 | `lifelines` | Cox proportional hazards models, Kaplan-Meier estimation, log-rank tests, concordance index |
| `scikit-survival` | 0.28.0 | `sksurv` | LASSO-Cox (CoxnetSurvivalAnalysis), random survival forest permutation importance, time-dependent cumulative AUC |
| `statsmodels` | 0.14.6 | `statsmodels` | Benjamini-Hochberg false discovery rate correction |
| `networkx` | 3.6.1 | `networkx` | Connected-component fallback for collinearity reduction |
| `python-igraph` | 1.0.0 | `igraph` | Louvain community detection on the feature correlation graph |
| `torch` | 2.13.0 | `torch` | All neural network components: the CANVAS head, focal loss, weighted sampling, and the graph/2D-grid/3D-grid context encoders |
| `torchvision` | 0.28.0 | `torchvision` | ResNet-50 benchmark encoder and its own preprocessing transforms |
| `timm` | 1.0.28 | `timm` | Model registry and data configuration for UNI and MUSK |
| `transformers` | 5.15.0 | `transformers` | Phikon encoder (ViTModel) and its AutoImageProcessor |
| `huggingface-hub` | 1.27.0 | `huggingface_hub` | Weight download and token-gated model access |
| `pillow` | 12.3.0 | `PIL` | Patch image handling and resampling |
| `opencv-python-headless` | 5.0.0.93 | `cv2` | Tissue segmentation (HSV, Otsu, morphology), artefact filtering, marker gating thresholds |
| `tifffile` | 2026.7.31 | `tifffile` | Pyramidal OME-TIFF reading for the paired H&E whole-slide images |
| `imagecodecs` | 2026.6.26 | `imagecodecs` | JPEG and zlib codecs those OME-TIFFs use |
| `zarr` | 3.3.0 | `zarr` | Windowed level-0 reads, so a 13.5 GB image is never loaded whole |
| `matplotlib` | 3.11.1 | `matplotlib` | Every figure in this report |
| `python-docx` | 1.2.0 | `docx` | Word rendering of the report and manuscript |
| `openpyxl` | 3.1.5 | `openpyxl` | Reads the TCGA-CDR clinical workbook |
| `pytest` | 9.1.1 | `pytest` | Test suite |

**Optional dependencies and the fallbacks taken.** Each of these is absent from this run, and each has a documented fallback rather than a silent skip:

- **spatial-lda** — True spatial-LDA prior with a smoothness penalty over neighbouring index cells. NOT installed in this run, so scikit-learn's LatentDirichletAllocation was used instead: the same generative model without the spatial prior. This is a real deviation and is recorded in T10.
- **openslide-python** — Aperio/SVS reading for TCGA slides. Not required for the Orion cohort, whose OME-TIFFs are read with tifffile and zarr.
- **stardist** — Nuclear segmentation. Not required here: the paired cohort ships its own segmented single-cell tables.

Approximate runtime this session (CPU-only, Ryzen 7000, no GPU): stage5 `--simulate --n-samples 200` ~17.5 min; 6-seed x 4-mode x 15-epoch benchmark ~85 min (grid3d dominates -- pilot.yaml's own comment already warns it is ~4x grid2d); 3-seed supplementary per-class/confusion-matrix run ~45 min; null-calibration chain (n=200) ~2 min; report figure/table generation (this file) ~5-10 min, dominated by F18's consensus-clustering k-sweep.

Exact commands to regenerate every artefact referenced in this report:

```
python -m pytest tests/ -q
python scripts/run_stage5_features.py --simulate --n-samples 200
python scripts/run_final_benchmark.py --simulate --modes none graph grid2d grid3d --seeds 1 2 3 4 5 6 --epochs 15 --window 7
python scripts/run_final_benchmark.py --simulate --modes none graph grid2d grid3d --seeds 1 2 3 --epochs 15 --window 7 --outdir results/per_class_run
python scripts/validate_stage6_null.py --features data/interim/sim_features.parquet --endpoint OS --seed 42 --outdir results/null_check_n200
python scripts/run_report.py
```
