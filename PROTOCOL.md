# PROTOCOL.md — canvas-ctx: methodological specification

Authoritative methodological specification for this project. Parameters marked [PAPER] are taken from the published STAR Methods and must not be changed without recording the change in the deviation table.

## What this project is

A faithful re-implementation of **CANVAS** (Li et al., *Cell* 2026,
doi:10.1016/j.cell.2026.05.031) adapted from NSCLC to **breast cancer**, using
only public data.

CANVAS infers spatial tumour ecological habitats from routine H&E histology.
Original design:

1. Discover 10 cellular neighbourhoods (CNs) from 41-plex CODEX spatial proteomics.
2. Co-register CODEX and H&E on the same section, transfer CN labels to H&E
   patches, train a pathology foundation model to predict habitats from H&E alone.
3. Apply to large H&E cohorts, engineer 262 spatial features, model prognosis,
   ecotypes and treatment response.

Reference implementation: https://github.com/lilab-stanford/CANVAS
Consult it when a spec here is ambiguous, but do not assume file paths match.

## The critical substitution

CANVAS module 2 requires paired spatial-omics + H&E from the same tissue. We do
not have CODEX. Breast cancer substitutes, all public:

| CANVAS component | Breast substitute | Access |
|---|---|---|
| CODEX 41-plex atlas (CN discovery) | Danenberg 2022 IMC, 693 METABRIC tumours, 37 markers, 32 cell phenotypes, single-cell tables with x/y | Zenodo 10.5281/zenodo.5850952 |
| Secondary CN cohort | Jackson 2020 Basel IMC, 352 tumours | Zenodo 10.5281/zenodo.3518284 |
| Paired spatial + same-section H&E | 10x Xenium FFPE human breast, post-Xenium H&E with registration file, CC BY 4.0 | 10xgenomics.com/datasets |
| TCGA/PLCO/NLST deployment | TCGA-BRCA diagnostic WSI, n≈1,100 | GDC portal |
| Molecular correlates | TCGA-BRCA RNA-seq, MC3 mutations, GISTIC CNA | cBioPortal / GDC |

**Consequence:** CN discovery is from IMC (protein, 37-plex) not CODEX (41-plex),
and the label-transfer cohort is Xenium (RNA) not CODEX. This is a defensible
methodological adaptation, not a shortcut. State it explicitly in any manuscript.
Do not describe the output as "CODEX-derived".

## Compute constraints — read this before designing anything

Target machine is **CPU-only, Ryzen 7000, 16 GB RAM, no GPU**. Every stage must
have a `--profile` flag with three tiers:

- `laptop` — ≤50 WSIs, ResNet-50 or Phikon-S encoder, 20× tiles, non-overlapping
  stride, batch 16, feature caching to disk-backed parquet. Must complete overnight.
- `colab` — free T4/L4, ≤300 WSIs, UNI or CONCH encoder, mixed precision.
- `hpc` — full cohort, MUSK encoder, matches the paper.

Never load a whole WSI into RAM. Always stream tiles. Always cache encoder
embeddings to disk so the classifier head can be retrained without re-encoding.

Foundation model access: MUSK, UNI, Virchow and CONCH are **gated** on HuggingFace
and need approved access plus a write token. Phikon and CTransPath are ungated.
Default to Phikon on `laptop`, fall back to ResNet-50 ImageNet if no HF token.
Never hardcode a token; read `HF_TOKEN` from the environment.

## Non-negotiable methodological parameters

Taken directly from the CANVAS STAR Methods. Do not silently change these.

**CN discovery (stage 1)**
- Neighbourhood radius: 40 µm around each index cell (yields ~25 neighbours).
- Model: spatial-LDA over local cell-type composition, then k-means on the
  resulting topic-proportion profiles.
- k sweep: k = 5–20. Select using Silhouette, Davies–Bouldin, and Adjusted Rand
  Index between adjacent k partitions. Report all three; do not auto-pick on one.
- Cross-cohort: co-cluster each TMA cohort separately with the paired cohort to
  test transferability.

**Pairwise interaction / co-occurrence**
- 40 µm radius; 1,000 label permutations preserving coordinates; enriched pairs =
  attraction, depleted = avoidance.

**Triplet motifs**
- Build global cell-type interaction graph, GCN trained to regress observed node
  degree (MSE), node scores = topological prominence.
- Enumerate all 3-cell-type combinations. With C cell types this is C-choose-3
  (455 for C=15). Integrate metric rankings with robust rank aggregation, keep
  FDR < 0.01.

**Barrier score**
- Spatial graph of CD8 T cells, proliferating tumour cells, CAFs. For each
  cytotoxic T cell not adjacent to tumour, find shortest immune→tumour path;
  barrier-positive if ≥1 CAF lies on it. Score = fraction barrier-positive.

**H&E preprocessing (stage 4)**
- Standardise to 40× equivalent, 0.25 µm/px. Rescale lower-res slides up to this
  reference. On `laptop` profile you may work at 20× (0.50 µm/px) — the paper
  shows accuracy is comparable between 40× and 20×, cite Figure S7B.
- Remove pen marks, folds, blur by colour-based filtering.
- Tissue segmentation via CLAM. Non-overlapping 224×224 patches. Stride 224 is
  fine — the paper shows 50% and 75% overlap give no meaningful gain (Fig S7C).
- Stain normalisation before encoding.
- Nuclear segmentation: CellViT. On `laptop`, substitute StarDist or HoVer-Net-lite.

**Label transfer (stage 2)**
- Affine registration, source = spatial-omics cell centroids, target = H&E space.
  `xdst = A · xsrc`, A is 2×3.
- Assign each H&E-segmented cell the CN of its nearest registered spatial cell,
  centroid-to-centroid threshold **5 µm**. Drop anything beyond.
- Patch labelling rules, apply exactly:
  - ≤5 annotated cells → `background` class.
  - >15 annotated cells AND dominant CN ≥60% of local composition → keep, label
    = dominant CN.
  - Everything in between → discard.
- Split train/val **at the sample level**, never at patch level.

**Model (stage 3)**
- Input 224×224 patches, resized to 384×384 for MUSK.
- Head: `Dropout(ReLU(W1 h + b1))` → 256, `Dropout(ReLU(W2 z1 + b2))` → 128,
  linear → K+1 logits (10 habitats + background).
- Class imbalance: weighted random sampling (weight = 1/class frequency) **and**
  focal loss `L = -α(1-p_t)^γ log(p_t)`. Use both, not one.
- Augmentation: random h/v flip, rotation, brightness/contrast/saturation/hue
  jitter. Normalise with the encoder's own channel statistics.
- Optimiser Adam, lr 1e-4, exponential decay 0.95, batch 64, dropout 0.5.
- 60 epochs, two-stage: epochs 1–10 head only with backbone frozen; epochs 11–60
  unfreeze the **final two encoder layers** only.
- Metrics: accuracy, macro-F1, Cohen's kappa. Bootstrap 1,000× for CIs.
- Benchmark against UNI, Virchow, ResNet-50 under identical splits and labels.

**Feature engineering (stage 5) — 262 features total**
1. Composition, n=10: frequency of each habitat per image.
2. Diversity, n=6: richness, Shannon, Simpson, inverse Simpson, Fisher's alpha,
   Pielou's evenness, computed on habitat counts as species abundances.
3. Spatial dispersion, n=90: per habitat as a planar point pattern — Ripley's K
   and L summaries, pair correlation, G, F, J functions, Clark–Evans index,
   quadrat dispersion statistics, kernel density summaries. 9 per habitat × 10.
4. Interaction, n=100: full 10×10 ordered habitat-pair matrix, permutation
   scored with 1,000 iterations against a randomised null.
5. Distance, n=55: pairwise nearest-neighbour Euclidean distances between habitat
   pairs, 10 self + 45 unordered pairs.
6. Transition, n=1: spatial transition entropy — build habitat transition matrix
   from patch-level k-NN (**k=6**), take Shannon entropy of the global transition
   probability distribution.

Assert `len(feature_matrix.columns) == 262` in a test. If your dispersion block
does not produce exactly 90, fix the per-habitat count, do not pad.

**Clinical modelling (stage 6)**
- Partition each slide into **tumour bulk** and **leading edge** by unsupervised
  spatial clustering of aggregated patch-level tumour probability, then compute
  habitat profiles per compartment. Exclude samples with insufficient patch
  coverage in either compartment.
- Univariate Cox per habitat × compartment. Report HR, 95% CI, Wald p.
- Endpoints from TCGA-CDR: OS, DSS, PFI, DFI. Analyse within subtype.
- Ecotypes: z-score the habitat × compartment matrix, ConsensusClusterPlus with
  **PAM and Canberra distance**. Then multivariable Cox adjusted for age, stage,
  and for breast: ER/PR/HER2 status, PAM50, grade. (Paper used smoking status —
  drop it, it is not a breast covariate.)
- Immunogenomics: join on TCGA barcodes. ESTIMATE stromal/immune/purity,
  CIBERSORTx LM22, xCell 64 signatures, GSVA on MSigDB/KEGG/Reactome. ssGSEA for
  IFN-γ, TLS, EMT-TGFβ, angiogenesis signatures. Covariate-adjusted ANOVA.
- Genomics: gene-level CNA as del/neutral/amp, MC3 binarised mutations, Fisher's
  exact per gene per ecotype.
- Signature model: Spearman correlation matrix → drop |ρ| > 0.95 by Louvain
  community detection, keep one representative per community → 100× resampled
  LASSO-Cox (5-fold internal CV on C-index), keep top 10% by selection frequency
  → 1,000-iteration permutation random survival forest importance → intersect the
  two lists → multivariable Cox on the intersection. Report C-index and
  time-dependent AUC at 6, 12, 24 months. Stratify by median risk score.

**Statistics**
- Two-sided throughout. t-test or Mann–Whitney for two groups, ANOVA or
  Kruskal–Wallis for more. Benjamini–Hochberg FDR. α = 0.05.

## Breast-specific changes you must make

- Cell-type taxonomy comes from Danenberg's 32 phenotypes, not the CODEX panel.
  Collapse to a working lineage set in `config/celltypes_brca.yaml`. Keep tumour
  epithelial subsets separate from myoepithelial — myoepithelial presence is
  diagnostically meaningful in breast (DCIS vs invasive) and has no lung analogue.
- Stratify every analysis by **PAM50 subtype and by ER/HER2 status**. Habitat
  prognostic direction is unlikely to be uniform across Luminal A and TNBC. The
  paper stratified by LUAD/LUSC; that is the structural equivalent.
- The immunotherapy arm (CANVAS module 3, ICB cohorts) has no clean public breast
  equivalent. Either drop it, or substitute neoadjuvant chemotherapy response
  (pCR) using a public NAC breast cohort. Do not silently claim ICB results.
- TNBC has a genuine public spatial resource (Keren 2018 MIBI-TOF) — use it as an
  independent CN-reproducibility check.

## Code standards

- Python 3.10. Type hints on public functions. Numpy-style docstrings.
- Config-driven: no magic numbers in function bodies. Everything tunable lives in
  `config/default.yaml` and is loaded once into a dataclass.
- Every stage script is idempotent and resumable. Check for the output artefact
  and skip unless `--force`.
- Log with `logging`, not `print`. One log file per run under `logs/`.
- Set and record seeds. Write a `run_manifest.json` per stage recording config
  hash, git SHA, package versions, input checksums.
- Tests in `tests/` with pytest. Synthetic fixtures, no data downloads in tests.
- No notebooks in the pipeline path. Notebooks are for figures only.

## Order of work

Do not build stages 3–6 before stage 1 produces validated CNs. Suggested order:

1. `stage1_cn` on Danenberg data. This runs on CPU in minutes and needs no
   images. Get 10 interpretable breast CNs first — everything downstream is
   meaningless without them.
2. `stage5_features` and `stage6_clinical` against **simulated** habitat maps, so
   the statistical machinery is tested and correct before real predictions exist.
3. `stage2_pair` on one Xenium breast sample end to end.
4. `stage3_model` with Phikon on that one sample, verify the training loop.
5. `stage4_infer` on 20 TCGA-BRCA slides, then scale.

## What to ask before assuming

If you hit any of these, stop and ask rather than guessing:
- Whether a downloaded Danenberg file uses µm or pixel coordinates, and its
  pixel size. The 40 µm radius is wrong if units are wrong, and everything
  downstream silently degrades.
- Whether TCGA slides are 40× or 20× native. Check `openslide` properties per
  slide, do not assume.
- Which TCGA-BRCA slides are diagnostic (DX) versus frozen (TS/BS). Use DX only.

---

# METHOD 2 — CANVAS-CTX (context-aware habitat inference)

This project runs **two methods** on the same data, same labels, same splits.

## The gap being targeted

CANVAS defines habitats from 40 µm *cellular neighbourhoods*, then predicts them
from H&E with a classifier that sees one 224×224 patch in isolation. The label is
contextual; the predictor is not. Meanwhile the field has moved to spatial
aggregation: DeepSpot uses deep-set networks over neighbourhood morphology,
SEPAL and EGGN propagate context across histology-derived neighbourhood graphs,
and TITAN arranges patch features into a 2D grid to preserve spatial context.

CANVAS-CTX closes that gap: predict habitat of patch *i* from patch *i* plus its
*k* spatial neighbours, via a deep-set branch and a distance-biased attention
branch feeding the **unchanged CANVAS head**.

## Why the comparison is clean

`k_neighbours = 0` reduces CANVAS-CTX **exactly** to the CANVAS head on the raw
embedding. Same encoder, same cached embeddings, same focal loss, same weighted
sampler, same sample-level splits, same head dimensions. So the k-sweep *is* the
ablation, and any gain is attributable to spatial context rather than to extra
parameters or a different backbone.

Run `k ∈ {0, 4, 8, 16}` and report the curve, not a single number. There is a
tested contract enforcing the k=0 equivalence (`tests/test_context_model.py`).

## Confound you must control

The paper's Method 1 fine-tunes the final two encoder layers. CANVAS-CTX trains
on frozen cached embeddings because that is what makes it CPU-feasible. **Do not
compare a frozen CANVAS-CTX against a fine-tuned Method 1.** Compare
frozen-vs-frozen. If you later get GPU time, fine-tune both or neither.

## Second confound: neighbour leakage

Neighbour indices are built per slide. Context must never cross slides, and the
train/val split must remain at sample level. A patch-level split plus spatial
context is catastrophic leakage — neighbouring patches land in both sets and
accuracy becomes meaningless. There is a test for this.

## What Method 2 gives you that Method 1 cannot

The attention weights. `ContextHabitatNet.attention_map()` returns which
neighbouring patches the model consulted to call a habitat. That is the natural
figure for the manuscript: an H&E region with its attended context overlaid.
Method 1 has no such output.

## The four context modes

Method 2 is not one model. It is three context encoders sharing the unchanged
CANVAS head, benchmarked against Method 1 in a single controlled ablation.

| mode | what it is | invariance |
|---|---|---|
| `none` | **Method 1. CANVAS exactly.** Per-patch. | n/a |
| `graph` | k-NN deep-set + distance-biased attention | permutation invariant |
| `grid2d` | local W×W feature grid, 2D conv + learned positional encoding | **orientation aware** |
| `grid3d` | local S×W×W multi-scale cube, 3D conv across magnification | orientation + scale aware |

### Why grid2d differs from graph even at matched neighbour count

k-NN with k=48 and a 7×7 window see the same 48 neighbours. The graph pools them
permutation-invariantly, so it can tell you *what* is nearby but not *which
direction* it lies in. The grid keeps the lattice, so it can represent
directional structure. Invasive fronts, duct walls and tumour–stroma interfaces
are directional by definition, which is exactly what CANVAS habitats CN07
(tumour–immune interface) and the barrier score are about. There is a test
asserting the grid is not rotation invariant.

### The third axis in grid3d is magnification, not depth

Habitats are hierarchical: nuclear detail at 40×, glandular architecture at 10×,
compartment structure at 2.5×. CANVAS uses one scale. A (D, S, W, W) cube lets a
3D kernel mix evidence across adjacent scales at nearby positions — a region
that reads as stroma at high magnification but sits inside a glandular structure
at low magnification is a different habitat from stroma at every scale.

Two ways to fill the scale axis, and you must state which you used:

- `scale_mode="pool"` — average-pool the fine lattice in 2×2 and 4×4 blocks.
  **Free**, no extra encoding. Approximate, because the mean of embeddings is
  not the embedding of the mean and the encoder is non-linear. Laptop default.
- `scale_mode="encode"` — encode 448 px and 896 px regions downsampled to 224 px
  separately. Faithful, **3× the encoding cost**. Use with GPU time.

There is a test asserting pooled scale-1 slots equal the mean of their 2×2
parents, so the pooling itself is verified even though the approximation is not.

**True z-axis 3D** (serial sections stacked into a tissue volume) is a different
thing and TCGA is single-section, so it is not available. If you obtain serial
sections, the same `Grid3DEncoder` applies with the scale axis replaced by z;
only the grid builder changes. Do not describe the multi-scale cube as 3D tissue.

## Files

    method2_context/context_model.py    graph aggregators, kNN index
    method2_context/grid.py             lattice, windows, pyramid pooling
    method2_context/habitat_net.py      unified model, all four modes
    method2_context/unified_dataset.py  dataset, loaders, training
    scripts/run_final_benchmark.py      THE four-way ablation
    tests/test_context_model.py         7 tests
    tests/test_grid.py                  14 tests

## Traps the tests already guard

- **Edge windows.** A corner patch has a quarter of its window off-slide. Those
  slots are parked on index 0 and masked. Gathering them unmasked gives every
  edge patch context from patch 0 of the slide.
- **Lattice holes.** Artefact-filtered patches leave empty slots. The mask is
  passed as an extra input channel so the network can tell "stroma here" from
  "nothing here"; without it, tissue edges look like low-signal tissue.
- **Slide crossing.** Lattices are built per slide. Combined with a patch-level
  split this would be catastrophic leakage. Split at sample level.
- **Lattice collisions.** Two patches in one slot means the stride is wrong or
  the tiling overlapped. The builder warns; do not ignore it.

## Reporting the benchmark honestly

Report macro-F1 and Cohen's kappa with bootstrap CIs, not accuracy — background
dominates and accuracy hides per-class collapse.

If `grid3d` beats `grid2d`, check it is not simply parameter count. Rerun
`grid2d` with `grid_channels` raised to match, then compare again. The synthetic
fixture in `--simulate` is deliberately built with oriented bands so the grid
wins; it is a machinery check and must never appear in a manuscript.

## Interpreting the synthetic benchmark

`--simulate` builds a fixture where habitat identity is a property of a local
*region*, and each patch embedding carries only a weak noisy signal of it. That
fixture is designed so context helps. A large gain there proves the machinery
works; it proves nothing about breast tissue. Do not put the synthetic number in
a paper.

---

# SMALL-DATA TRACK — config/pilot.yaml

Use `--config config/pilot.yaml` throughout for a laptop-scale run.

| Stage | Full | Pilot |
|---|---|---|
| CN discovery | Danenberg, 693 tumours | **Keren 2018 MIBI-TOF, 41 TNBC, one CSV** |
| CNs | 10 | 8 (sweep and confirm) |
| Paired sections | 10 WSI | 1 Xenium breast section |
| Encoder | MUSK, fine-tuned | Phikon, **frozen** |
| Resolution | 40× (0.25 µm/px) | 20× (0.50 µm/px) |
| Patches/slide | uncapped | 4,000 (grid-subsampled) |
| TCGA slides | ~1,100 | 60, PAM50-stratified |
| Interaction perms | 1,000 | 200 |

Stages 1, 5, 6 and all of Method 2 run entirely on the laptop. Only patch
encoding needs Colab, and only once — cache the embeddings and never re-encode.

**The pilot answers "does the pipeline work", not "what is the effect size".**
Do not report hazard ratios from 60 slides as a finding. `config/pilot.yaml`
ends with an honesty ledger listing every reduction; those belong in the
limitations section, not quietly absorbed.
