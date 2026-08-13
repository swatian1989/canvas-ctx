## Title

**Cellular neighbourhoods reproduce across spatial proteomics platforms but do not transfer between marker panels: a validated pipeline for habitat inference from routine histology**

*Author list, affiliations and corresponding author to be completed.*

*Scope note for the authors, to be deleted before submission. The title above states what the present data support: cross-platform reproducibility of cellular neighbourhoods, a verified registration and label-transfer pipeline, and a diagnosed failure mode when a taxonomy is carried across marker panels of different depth. The architectural contribution (CANVAS-CTX) is reported here as method and machinery, with its controlled ablation validated on synthetic data. The claim that spatial context improves habitat inference on real tissue requires the multi-specimen benchmark described in Ongoing work and is deliberately NOT made in this manuscript. Making it now would rest on a fixture constructed to reward context.*

## Abstract

**Background.** Tumour ecological habitats, defined from spatial proteomics as recurrent cellular neighbourhoods (CNs), can be inferred from haematoxylin and eosin (H&E) histology alone [ref: canvas]. The published approach defines each habitat label from a 40 um cellular neighbourhood but predicts it from a single 224x224 patch viewed in isolation, so the label is contextual while the predictor is not.

**Methods.** We re-implemented the CANVAS pipeline end to end and extended it with CANVAS-CTX, which predicts the habitat of a patch from that patch plus its spatial neighbours through three interchangeable context encoders (k-nearest-neighbour deep set, two-dimensional feature grid, multi-scale three-dimensional grid) feeding the unchanged CANVAS classification head. Because k = 0 reduces CANVAS-CTX exactly to the published per-patch model, the neighbour sweep is itself a controlled ablation. Owing to the scarcity of paired same-section spatial proteomics and H&E in breast tissue, CNs and the habitat classifier were derived in colorectal cancer [ref: schurch; ref: orion] and applied unchanged to breast, mirroring the cross-tumour-type transfer reported in the original work.

**Results.** Applying the neighbourhood-discovery procedure to 250,476 single cells profiled by 56-plex CODEX across 135 colorectal regions from 35 patients recovered ten spatially coherent cellular neighbourhoods spanning tumour, boundary, lymphoid, myeloid, stromal and vascular compartments. Scored against the neighbourhood labels published with the source cohort, agreement was moderate and structurally complete (adjusted Rand index 0.377; every published neighbourhood type represented; bulk tumour recovered at 89.5 percent overlap), despite the two procedures differing in how each window is defined. In an independent cohort imaged by 18-plex immunofluorescence with same-section haematoxylin and eosin, registration was verified rather than assumed: 97.1 percent of 1,620,375 segmented cells fell within the tissue mask and the global residual was bounded below 2.6 microns against a 5 micron acceptance threshold. Transferring the 56-plex neighbourhood taxonomy onto the 18-plex panel failed in a reproducible and diagnosable way: collapsing both taxonomies onto the eight lineages the thinner panel can express left one neighbourhood with a near-uniform composition profile, which acted as a nearest-centroid attractor and absorbed 64 percent of all cells at a median cosine similarity of 0.653. Deriving neighbourhoods de novo on the paired cohort resolved this, yielding ten habitats of which eight matched a discovery-cohort neighbourhood at cosine similarity above 0.89, and 13,321 patches passing the purity rules with every habitat represented.

**Conclusions.** Cellular neighbourhood structure is recoverable across spatial proteomics platforms and cohorts, but a neighbourhood taxonomy is not portable between marker panels of different depth: the limiting factor is the shared cell-type vocabulary, not the clustering. Habitats should therefore be derived on the panel that is paired with the histology, which is also what the original method does. We release a verified, configuration-driven implementation covering neighbourhood discovery through survival modelling, including a registration check that reports a residual bound rather than assuming a supplied alignment, and a calibration procedure that revealed in-sample optimism in the prognostic signature chain.

## Introduction

The spatial organisation of the tumour microenvironment carries prognostic information that is invisible to bulk molecular assays. Multiplexed imaging platforms have established that tumours are built from recurrent, discrete cellular neighbourhoods rather than from randomly mixed cell populations: co-detection by indexing (CODEX) identified conserved neighbourhoods at the colorectal cancer invasive front whose organisation stratified survival [ref: schurch]; imaging mass cytometry defined analogous structures across 693 breast tumours [ref: danenberg] and a single-cell pathology landscape across 352 further cases [ref: jackson]; and multiplexed ion beam imaging showed that the degree of immune compartmentalisation in triple-negative breast cancer is itself prognostic [ref: keren].

These platforms are, however, expensive, low-throughput and absent from routine practice, whereas H&E histology is universal. Recent work demonstrated that habitat identity defined from spatial proteomics can be transferred onto co-registered H&E and predicted from morphology alone using a pathology foundation model, enabling spatial habitat profiling of large archival cohorts [ref: canvas]. This depends on pathology foundation models that have advanced rapidly, from self-supervised encoders trained on pan-cancer tiles [ref: phikon; ref: uni] to vision-language models trained on paired image and text corpora [ref: musk].

A methodological gap remains. The habitat label assigned to a patch is defined by the composition of a 40 um neighbourhood surrounding it, yet the classifier that must reproduce that label sees only the patch itself. Information that defines the target is withheld from the predictor. The wider field has moved toward spatial aggregation for precisely this reason, arranging patch features on a two-dimensional lattice so that positional encoding and relative distance are preserved [ref: titan]. Directionality matters biologically: invasive fronts, duct walls and tumour-stroma interfaces are oriented structures, and a permutation-invariant summary of what is nearby cannot represent which direction it lies in.

We therefore re-implemented the published pipeline faithfully and added CANVAS-CTX, a context-aware variant in which the unchanged classification head receives, in addition to the index patch embedding, a representation of its spatial neighbourhood produced by one of three encoders. Setting the neighbour count to zero recovers the published model exactly, so any difference is attributable to spatial context rather than to a different backbone, a different loss, or additional capacity in the head.

A second constraint shaped the design. The pipeline requires spatial proteomics and H&E from the same tissue section. In breast cancer this pairing is scarce, and the available resource documents its own image alignment as suitable for regional annotation rather than sub-cellular correspondence. Colorectal cancer, by contrast, offers same-section 18-plex immunofluorescence with matched H&E across 74 resections [ref: orion], alongside the CODEX cohort in which cellular neighbourhoods were originally defined [ref: schurch]. We consequently derive habitats and train the classifier in colorectal tissue and apply the trained model unchanged to breast, treating breast as the generalisation test rather than the foundation. The original study performed the structurally equivalent experiment, applying a lung-trained model across multiple tumour types [ref: canvas].

## Results

### Cellular neighbourhoods in colorectal cancer

We analysed 258,385 single cells profiled by 56-plex CODEX across 140 imaged regions from 35 patients with colorectal carcinoma [ref: schurch]. Coordinates in the source table are recorded in image pixels on a 1920x1440 grid and were converted to microns at the 377.44 nm/pixel lateral resolution documented for this collection, giving a 724 x 543 um field of view per region. The conversion was verified against the data itself: median nearest-neighbour centroid spacing after conversion is 6.8 um, consistent with densely packed tissue. After excluding 7,357 cells (2.8 percent) assigned to an imaging-artefact class and regions falling below the minimum cell count, 250,476 cells across 135 regions and 28 cell types entered neighbourhood discovery.

At the 40 um radius specified by the method, neighbourhoods contained a median of 33 cells, above the approximately 25 reported for the lung cohort in which the method was developed. This reflects tissue density rather than a scaling error: the colorectal invasive front in this cohort carries a median of 4,890 cells per square millimetre. We note explicitly that adopting the 0.325 um/pixel resolution quoted for other installations of this imaging platform would reproduce approximately 25 neighbours, and that we rejected this adjustment because calibrating a documented instrument constant to reproduce a value observed in a different tissue would confound density with scale. The radius and the documented pixel size were both retained unchanged.

The three selection criteria did not converge on a single solution. Silhouette width was maximised at nine neighbourhoods (0.319), the Davies-Bouldin index minimised at eleven (1.180), and the adjusted Rand index between adjacent partitions peaked at seventeen (0.978) with a further strong local maximum at eleven (0.968). We report all three rather than optimising any one, and retained ten neighbourhoods as specified by the protocol. The silhouette optimum at nine coincides with the nine neighbourhoods reported by the original investigators of this cohort.

Neighbourhoods were named from cell-type enrichment within this cohort (Figure 1C, Supplementary Table S2) and comprise bulk tumour (12.1 percent of cells), a tumour boundary myeloid compartment (9.5 percent), cytotoxic-infiltrated stroma (11.7 percent), lymphoid follicle (6.4 percent), pan-immune cytotoxic (11.1 percent), memory CD4 T-cell (5.0 percent), plasma-cell rich (7.9 percent), macrophage-adipose (16.6 percent), smooth muscle-lymphatic (12.9 percent) and granulocyte-enriched (6.7 percent) structures. Plotting cells at their measured coordinates confirms that these are spatially coherent tissue domains rather than dispersed label assignments (Figure 1D).

Because the source table also carries the neighbourhood assignments published by the original investigators, the rediscovered neighbourhoods could be validated externally without additional data. Agreement across 250,476 cells was moderate (adjusted Rand index 0.377, normalised mutual information 0.470), with every published neighbourhood type represented among the rediscovered set: bulk tumour was recovered at 89.5 percent overlap, granulocyte-enriched at 82.2 percent, a memory T-cell compartment at 82.0 percent and lymphoid follicle at 72.6 percent (Supplementary Table S11). Partial rather than complete agreement is expected by construction, since the published analysis defined each window from the ten nearest neighbours whereas the present method applies a fixed 40 um radius followed by a topic decomposition. The comparison therefore establishes that a CANVAS-style procedure recovers comparable tissue architecture from the same cells, not that the original implementation was reproduced.

### Registration verification in the paired cohort

The paired cohort images 18-plex immunofluorescence and haematoxylin and eosin from the same tissue section and distributes the histology already registered to the multiplex frame, so the expected transform is the identity. We verified this rather than relying on it, because a sub-cellular offset would corrupt every patch label while leaving the output superficially plausible. Segmented cell centroids and the histology share a coordinate frame: centroid extents reach 78,371 and 55,447 pixels against image dimensions of 78,417 by 57,360, and the median segmented cell area of 406 square pixels corresponds to a 7.4 micron diameter at the stated resolution of 0.325 microns per pixel.

Across 1,620,375 segmented cells in the exemplar specimen, 97.1 percent of centroids fell inside a tissue mask derived independently from the histology. Cross-correlating that mask against the cell-density map placed the peak at zero displacement (Figure 2). We report the residual as a bound rather than as zero, because the check cannot resolve below one pixel at the pyramid level used: the global residual is below 2.6 microns, within the 5 micron acceptance threshold. The check is global and rigid, and would not detect local warping.

### A neighbourhood taxonomy does not transfer between marker panels

We first attempted to carry the 56-plex discovery taxonomy onto the 18-plex paired cohort, which is what the overall design requires. This failed, and the failure is informative rather than incidental.

The two panels resolve different numbers of cell types, 28 phenotypes against 16 biological markers, so the taxonomies cannot be matched cell for cell. Collapsing both onto the eight lineages the thinner panel can express makes them formally comparable but destroys the distinctions that define several neighbourhoods. One neighbourhood, defined in the full taxonomy by plasma cells and unassigned populations that both fall into catch-all buckets after collapse, was left with the flattest composition profile of all ten, no lineage exceeding 28.5 percent. A near-uniform vector is close to everything under cosine distance, and that neighbourhood duly absorbed 64 percent of all cells at a median cosine similarity of 0.653, while the smallest class retained two patches.

Grouping real histology patches by dominant gated lineage (Figure 2C) localised the limiting step. Tumour epithelium is called correctly, showing malignant glandular architecture with enlarged hyperchromatic nuclei, as is smooth muscle stroma; the immune and vascular gates reach only 28 to 46 percent dominant-lineage purity and are visibly mixed. The gated tumour fraction of 57.4 percent exceeds the 19 percent of the discovery cohort, which we attribute to sampling rather than threshold calibration: the paired cohort images whole tumour-rich resections, whereas the discovery cores were positioned at the invasive front to balance compartments. The morphology supports that reading.

### De novo derivation on the paired panel recovers the same structure

Deriving neighbourhoods directly on the paired cohort removes the vocabulary mismatch, because discovery and transfer then share one panel. This is also closer to the original method, which discovers neighbourhoods in the spatial modality paired with the histology rather than importing them from elsewhere.

Applying the identical procedure over 1,620,375 cells produced ten habitats with distinct identities: pure tumour, a second tumour compartment, myeloid and immune, two smooth-muscle stromal compartments, a B and T cell lymphoid aggregate, vasculature, mixed immune infiltrate, and a tumour-adjacent mixed compartment. No centroid acted as an attractor.

These independently derived habitats recapitulate the discovery-cohort taxonomy. Matching the two by lineage composition, eight of ten aligned at cosine similarity above 0.89, including bulk tumour at 0.999, smooth muscle at 0.970 and the lymphoid compartment at 0.971. The two exceptions were the habitat dominated by unassigned cells (0.559) and one vascular and stromal pairing (0.725). Two cohorts, two platforms and two independent derivations therefore converge on comparable tissue architecture, even though the taxonomy itself is not portable between them.

Under the patch purity rules this yielded 13,321 labelled patches with every habitat and the background class represented, the smallest at 87 patches, against two patches for the smallest class under the rejected transfer (Supplementary Table S3).

### Calibration of the spatial-feature and survival machinery

Before any real prognostic estimate is attempted, the downstream statistical chain was calibrated against an outcome carrying no signal: the 262-feature matrix computed from simulated habitat maps was joined to independently drawn exponential survival times. Univariate Cox testing behaved correctly, returning no features below the false-discovery threshold at n = 200 (Figure 5B, Supplementary Table S7), and Kaplan-Meier curves stratified by habitat tertile overlapped as expected (Figure 5C).

The terminal signature model did not behave correctly, and we report this as a methodological finding rather than a footnote. Fitted concordance indices of 0.60 to 0.80 were obtained on pure noise across four independent draws, decreasing as sample size rose. This is in-sample optimism: feature selection by resampled LASSO-Cox, filtering by random survival forest permutation importance, and evaluation of the final multivariable model all share the same subjects, with no held-out split at any point. Any concordance index produced by this chain, here or elsewhere, must therefore come from held-out data or carry an explicit optimism correction. We note in passing that the cross-validation fold parameter is declared but unreferenced in the reference implementation, which may contribute to the instability at small sample size.

### Ongoing work: the context ablation on real tissue

The architectural contribution described in Methods is implemented, tested and validated as machinery, but the claim that spatial context improves habitat inference is deliberately not made here. On a synthetic fixture the four-mode ablation executes correctly across six seeds with no class collapse, confirming that the training loop, the slide-level split and the paired comparison behave as specified (Figure 3). That fixture is constructed with oriented bands that reward spatial context by design, so the margin it produces characterises the implementation and carries no information about tissue. Reporting it as evidence would be circular.

The corresponding test on real histology requires habitat-labelled patch embeddings across enough specimens to support a patient-level split, since a patch-level split combined with spatial context places neighbouring patches on both sides of the partition and renders the comparison meaningless. Twelve specimens of the paired cohort have been acquired and processed to a shared habitat taxonomy for this purpose. The benchmark will report macro-F1 and Cohen's kappa with bootstrap confidence intervals and per-class recall across at least six seeds with a paired signed-rank test, noting that six pairs bound the attainable p value at 0.0312, and will include a capacity-matched comparison between the two-dimensional and multi-scale grid encoders so that any advantage of the latter is not confounded with parameter count.

Deployment to independent histology cohorts, including the cross-cancer application, follows the same sequence and is likewise not claimed here.

## Discussion

Cellular neighbourhood structure is reproducible across cohorts and platforms. Applying a fixed-radius, topic-decomposition procedure to a 56-plex colorectal cohort recovered every neighbourhood type reported by the original investigators of that cohort, with bulk tumour matched at 89.5 percent overlap, despite the two analyses defining each window differently: ten nearest neighbours there against a 40 micron radius here. An adjusted Rand index of 0.377 is moderate, and we read it as convergence on comparable tissue architecture rather than as reproduction of a specific partition. The same convergence appeared again, independently, when habitats were derived on an entirely separate cohort imaged with a different platform, where eight of ten habitats matched a discovery-cohort neighbourhood above cosine 0.89.

The taxonomy itself, however, is not portable. Our attempt to carry a 56-plex neighbourhood definition onto an 18-plex panel failed in a specific and diagnosable way, and we report it because the failure mode generalises. Any transfer of this kind requires a shared cell-type vocabulary, and the shared vocabulary is bounded by the thinner panel. Neighbourhoods distinguished only by phenotypes the thinner panel cannot resolve collapse toward a near-uniform composition, and a near-uniform centroid is nearest to everything under the usual distance metrics, so it absorbs the cohort. The practical implication is that habitats should be defined on the panel that is paired with the histology. That is what the original method does, and our results indicate it is a requirement rather than a convenience.

Two methodological points recur throughout this work and apply beyond it. First, supplied alignments should be verified. The paired cohort is imaged from a single section and distributed pre-registered, and our check confirmed it, but the check is cheap, needs no landmarks, and reports a residual bound; a sub-cellular offset would otherwise corrupt every label while leaving the output plausible. Second, selection and evaluation on the same subjects inflates apparent prognostic performance. Our calibration against a null outcome returned concordance indices as high as 0.80 on pure noise, decaying with sample size, which is the signature of in-sample optimism rather than of any defect in the individual estimators.

The architectural question that motivated this work remains open by design. Habitat labels are defined from a 40 micron neighbourhood while the published predictor observes a single patch, so information that determines the target is withheld from the model, and the wider field has moved toward spatially aware aggregation for related reasons [ref: titan]. Our implementation makes the corresponding test a controlled ablation rather than a model comparison, since the context branch degenerates exactly to the per-patch baseline at zero neighbours. We report the machinery as validated and the biological question as unanswered, because the only data on which we have run the ablation is a fixture built to reward the hypothesis.

The clinical scope of this work is correspondingly limited. We describe a verified pipeline and two reproducibility results, not a prognostic finding. The deployment cohorts contemplated in the design are small enough that stratified hazard ratios would not be interpretable, and we have deliberately reported none.

## Limitations

The cellular neighbourhoods underpinning every downstream analysis are derived from colorectal tissue and applied to breast without recalibration. Colorectal and breast neighbourhoods are not biologically identical, and habitat composition shift between the two cohorts is reported rather than minimised or concealed.

The habitat classifier is trained on a frozen encoder. The published method fine-tunes the final two layers of its backbone; freezing is required here for CPU feasibility and, more importantly, to keep the comparison between the per-patch and context-aware models fair, since the latter necessarily trains on cached embeddings. A frozen comparison is internally valid but establishes a lower ceiling than the published configuration.

The multi-scale grid fills its scale axis by average-pooling the fine lattice rather than by re-encoding each magnification. Because the encoder is non-linear, the mean of embeddings is not the embedding of the mean, so the coarse scales are approximations. Faithful re-encoding triples encoding cost and was not performed.

Deployment cohorts of 60 slides per cancer type, stratified across five intrinsic subtypes, are adequate to demonstrate that the pipeline runs and inadequate to estimate effect sizes. Hazard ratios from strata containing few events are not interpretable and are not reported as findings.

Interaction permutation counts are reduced relative to the published protocol for tractability, which widens the uncertainty on weak habitat-pair associations while leaving strong ones stable.

Two deliberate departures from the published statistical procedure are documented in Methods and quantified in the supplementary material: a toroidal-shift null in place of label shuffling, and edge-corrected spatial dispersion estimators. Both are more conservative than the procedures they replace, and both change reported significance.

## Methods

*The parameters below marked [PAPER] follow the published protocol [ref: canvas] and were not altered. Departures are stated explicitly, with rationale, and are itemised in full in Supplementary Table S10.*

### Cellular neighbourhood discovery

Single-cell tables with spatial coordinates and cell-type assignments were used to construct, for each index cell, the local neighbourhood of all cells within a 40 um radius [PAPER], which yields approximately 25 neighbours. Coordinate units were verified before analysis; the implementation refuses to proceed when the coordinate span is implausible for microns, because applying a micron radius to pixel coordinates silently collapses every neighbourhood while still producing clean-looking clusters. Neighbourhood composition vectors were decomposed by spatial latent Dirichlet allocation and the resulting topic proportions clustered by k-means. The number of clusters was swept from 5 to 20 [PAPER] and scored by silhouette width, Davies-Bouldin index and the adjusted Rand index between adjacent partitions; all three are reported and none is optimised alone.

**Departure.** Neighbourhoods were derived from a 56-plex colorectal CODEX cohort [ref: schurch] rather than the 41-plex lung cohort of the original study, for the paired-data reasons given in the Introduction.

### Habitat label transfer

Spatial-omics cell centroids were mapped into H&E pixel space by an affine transform. Each H&E-segmented nucleus was assigned the CN of its nearest transformed spatial cell, subject to a centroid-to-centroid threshold of 5 um [PAPER]; unmatched nuclei were discarded. Registration residuals were measured and reported before label transfer proceeded, rather than assumed from the pre-registered status of the source data.

Patches of 224x224 pixels [PAPER] were labelled by the published purity rules, applied exactly: five or fewer annotated cells assigns the background class; more than fifteen annotated cells with a dominant CN comprising at least 60 percent of local composition assigns that dominant CN; all intermediate cases are discarded [PAPER]. Train, validation and test partitions were formed at the patient level and never at the patch level [PAPER]. This is not a stylistic preference: patch-level splitting combined with spatial context places neighbouring patches on both sides of the split and renders the resulting accuracy meaningless.

### Whole-slide processing and patch embedding

Slide resolution was read per slide rather than assumed. Tissue was segmented on a downsampled thumbnail by saturation thresholding with morphological closing, a simplified surrogate for the published segmentation step [ref: clam]. Non-overlapping 224x224 patches [PAPER] were extracted on a regular grid at the target resolution; the published analysis shows that overlapping strides confer no meaningful benefit. Patches were screened for pen ink, tissue folds and blur by colour and focus criteria, and stain-normalised by the Macenko method [ref: macenko] against a fixed reference matrix so that all patches are normalised to a common target irrespective of source slide.

Patches were encoded in streaming batches and cached to disk as one columnar shard per slide, allowing the classification head to be retrained without re-encoding and allowing an interrupted run to resume without recomputing completed slides. Each encoder is applied with its own published preprocessing and normalisation statistics; a generic ImageNet transform is never substituted, as this degrades embeddings silently rather than raising an error.

**Departure.** Phikon [ref: phikon] was used as a frozen encoder in place of the fine-tuned MUSK backbone [ref: musk] of the original study, for compute reasons and to preserve a frozen-versus-frozen comparison between Method 1 and Method 2. Slides were processed at 0.50 um per pixel rather than 0.25 um per pixel; the published analysis reports comparable accuracy between these resolutions.

### Habitat classification and the CANVAS-CTX extension

The classification head follows the published architecture exactly [PAPER]: a 256-unit layer and a 128-unit layer, each with rectified linear activation and dropout, followed by a linear map to K+1 logits comprising the habitat classes and background. Class imbalance is addressed by both weighted random sampling with weights inversely proportional to class frequency and focal loss, as specified, rather than by either alone [PAPER].

CANVAS-CTX augments the input to this unchanged head with a representation of the index patch's spatial neighbourhood, computed by one of three encoders. The graph encoder pools the k nearest neighbouring patches by a deep set with distance-biased attention and is permutation invariant. The two-dimensional grid encoder arranges neighbours on their true lattice positions and applies two-dimensional convolution with learned positional encoding, and is therefore orientation aware. The multi-scale grid encoder extends this to a scale axis spanning magnifications, allowing a kernel to combine evidence across resolutions at nearby positions.

Neighbour indices are constructed strictly within a slide, so context never crosses slide boundaries. Positions falling outside the slide are masked, and lattice gaps left by artefact-filtered patches are passed to the network as an explicit occupancy channel so that absent tissue is distinguishable from low-signal tissue. Setting the neighbour count to zero reduces the model exactly to the published per-patch classifier; this equivalence is enforced by an automated test rather than asserted.

### Spatial feature engineering

Each slide's habitat map yields 262 features [PAPER] in six blocks: habitat composition (10), ecological diversity treating habitats as species (6), intra-habitat spatial dispersion as planar point patterns (90, being nine metrics for each of ten habitats), pairwise habitat interaction scored against a spatial null (100), nearest-neighbour distances within and between habitats (55), and spatial transition entropy computed over a patch-level six-nearest-neighbour graph (1) [PAPER]. The total is asserted at runtime so that a block producing the wrong count fails loudly rather than silently altering the feature space.

**Departure, interaction null.** Habitat-pair interaction is scored against a toroidal-shift null rather than by shuffling habitat labels. Label shuffling destroys all spatial autocorrelation, so the resulting null describes randomly scattered habitats, a configuration real tissue never adopts. Tested against it, nearly every habitat pair is declared significant, and the test effectively asks whether the tissue is spatially organised at all rather than whether two specific habitats are associated. The toroidal shift preserves each habitat's own autocorrelation and domain structure and randomises only the relative registration between the habitat field and the point pattern. The magnitude of this difference is quantified in the supplementary material.

**Departure, edge correction.** Ripley's K and L are computed with border correction and the Clark-Evans index with Donnelly's perimeter correction. Uncorrected estimators are biased on small tissue regions because boundary points have artificially few observed neighbours. Where too few eligible centres remain at a given radius the estimate is returned as missing rather than as a misleading finite value.

### Clinical modelling

Slides were partitioned into tumour bulk and leading edge compartments by unsupervised spatial clustering of smoothed patch-level tumour probability and its local gradient, and habitat profiles computed within each compartment [PAPER]. Univariate Cox proportional hazards models were fitted per habitat and compartment, reporting hazard ratios per standard deviation with 95 percent confidence intervals, Wald p values and Benjamini-Hochberg adjusted q values. Endpoints follow the curated pan-cancer clinical resource [ref: tcgacdr]. Ecotypes were derived by consensus clustering of z-scored habitat-by-compartment profiles using partitioning around medoids with Canberra distance [PAPER], and characterised by multivariable Cox models adjusted for age, stage and cancer-appropriate covariates.

The prognostic signature follows the published sequence [PAPER]: collinear features are reduced by community detection on the Spearman correlation graph, retaining one representative per community; surviving features are ranked by selection frequency across resampled LASSO-Cox fits and independently by random survival forest permutation importance; and the intersection of the two rankings enters a multivariable Cox model reporting the concordance index and time-dependent area under the curve.

**Validation caution.** Applying this chain to independently randomised survival outcomes, univariate testing behaved correctly, returning essentially no features below the adjusted significance threshold. The terminal signature model did not: fitted concordance indices of 0.60 to 0.80 were obtained on pure noise, decreasing with increasing sample size. This is in-sample optimism arising because feature selection and model evaluation share the same subjects. Concordance indices reported from this pipeline must therefore derive from held-out data or carry an explicit optimism correction.

### Statistics

All tests are two-sided with alpha of 0.05 and Benjamini-Hochberg control of the false discovery rate [PAPER]. Two-group comparisons use the t test or Mann-Whitney U test and multi-group comparisons analysis of variance or the Kruskal-Wallis test, according to distribution. Classifier performance is reported as macro-F1 and Cohen's kappa with bootstrap confidence intervals and per-class recall; overall accuracy is never reported alone, because the background class dominates and accuracy conceals collapse of a minority habitat. Model comparisons across context modes use at least six random seeds, each reshuffling both the slide-level split and model initialisation, summarised as mean and standard deviation and tested by paired Wilcoxon signed-rank test. Effect sizes are reported with dispersion or confidence intervals in all cases; p values are never reported alone.

## Data availability

All primary data are public. Colorectal CODEX single-cell data are available from Mendeley Data (mpjzbtfgfr) [ref: schurch]. Paired immunofluorescence and H&E whole-slide images are available from the public Orion release [ref: orion]. Diagnostic whole-slide images and clinical outcomes are available from the Genomic Data Commons and the curated pan-cancer clinical resource [ref: tcgacdr]. Breast imaging mass cytometry resources referenced for comparison are available from their respective repositories [ref: danenberg; ref: jackson; ref: keren].

## Code availability

Analysis code, configuration files and the automated test suite are available at [REPOSITORY URL PENDING]. All stages are configuration driven, seeded and resumable, and each records a manifest of configuration hash, package versions and input checksums. *Note for the authors: the working directory is not currently under version control, so no commit identifier can be cited. Initialise a repository before submission.*

## Figures

### Figure 1. Study design and cellular neighbourhood discovery

![Figure 1 panel](figures/F1_study_design.png)

![Figure 1 panel](figures/F2_k_sweep.png)

![Figure 1 panel](figures/F3_cn_marker_heatmap.png)

![Figure 1 panel](figures/F4_cn_spatial_maps.png)

![Figure 1 panel](figures/F5_cn_composition.png)

**Figure 1. Study design and cellular neighbourhood discovery.** (A) Schematic of the six-stage pipeline, the two model variants and the substitutions made relative to the published protocol. (B) Selection diagnostics across candidate cluster numbers, showing silhouette width, Davies-Bouldin index and adjacent-k adjusted Rand index on shared axes with the selected solution marked. (C) Marker enrichment per cellular neighbourhood, z-scored across neighbourhoods. (D) Representative tissue regions coloured by neighbourhood assignment. (E) Neighbourhood composition per sample, ordered by dominant neighbourhood. Panels B to E require the colorectal CODEX single-cell table.

### Figure 2. Habitat label transfer onto matched H&E

![Figure 2 panel](figures/F6_registration_qc.png)

![Figure 2 panel](figures/F7_patch_labels.png)

**Figure 2. Habitat label transfer onto matched H&E.** (A) Immunofluorescence and H&E overlay for a representative specimen after affine alignment. (B) Distribution of centroid-to-centroid registration residuals in microns with the 5 um acceptance threshold indicated. (C) Patch label distribution per class and per split, with representative patches for each habitat. All panels require the paired Orion specimens.

### Figure 3. Validation of the context ablation on a synthetic fixture

![Figure 3 panel](figures/F8_training_curves.png)

![Figure 3 panel](figures/F9_confusion_matrix.png)

![Figure 3 panel](figures/F10_benchmark.png)

![Figure 3 panel](figures/F11_params_vs_f1.png)

**Figure 3. Validation of the context ablation on a synthetic fixture.** (A) Training loss and validation macro-F1 by epoch for all four context modes. (B) Row-normalised confusion matrix with per-class recall annotated. (C) Macro-F1 and Cohen's kappa by context mode, mean and standard deviation across six seeds with individual seeds overlaid and paired test results annotated; the per-patch baseline is marked as the reference method. (D) Parameter count against macro-F1, testing whether any advantage of the grid encoders reflects capacity rather than spatial structure. THESE PANELS ARE MACHINERY VALIDATION, NOT A BIOLOGICAL RESULT. The fixture is constructed with oriented bands that reward spatial context by design, so the separation shown demonstrates that the training loop, the slide-level split and the paired comparison behave as specified, and nothing about tissue. The equivalent test on real histology is described under Ongoing work.

### Figure 4. Planned deployment to independent histology cohorts

![Figure 4 panel](figures/F13_wsi_habitat_maps.png)

![Figure 4 panel](figures/F12_attention_maps.png)

**Figure 4. Planned deployment to independent histology cohorts.** (A) Whole-slide habitat maps overlaid on H&E with the tumour bulk and leading edge boundary drawn. (B) Attention weights showing which neighbouring patches the context model consulted, an output the per-patch model cannot produce. (C) Habitat composition shift between the training and transfer cancer types. This display item is reserved for the deployment described under Ongoing work and is not populated in the present manuscript; the panels shown state the data each requires.

### Figure 5. Spatial features and clinical association

![Figure 5 panel](figures/F14_feature_matrix.png)

![Figure 5 panel](figures/F15_feature_correlation.png)

![Figure 5 panel](figures/F16_cox_forest.png)

![Figure 5 panel](figures/F17_km_tertile.png)

![Figure 5 panel](figures/F18_consensus.png)

![Figure 5 panel](figures/F20_signature.png)

**Figure 5. Spatial features and clinical association.** (A) The 262-feature matrix, clustered, annotated by feature block. (B) Univariate Cox hazard ratios with 95 percent confidence intervals per habitat and compartment, with false-discovery-significant associations highlighted. (C) Kaplan-Meier curves by habitat tertile with log-rank tests. (D) Consensus clustering diagnostics and the resulting ecotypes with clinical annotation tracks. All panels require habitat maps joined to clinical outcomes.

### Supplementary Figure S1. Null-model comparison

![Supplementary Figure S1 panel](figures/F21_null_comparison.png)

**Supplementary Figure S1. Null-model comparison.** Habitat pairs declared significant under a toroidal-shift null against label shuffling on identical input, with the corresponding distributions of interaction z-scores. Demonstrates why the toroidal null is used throughout. Computable without further data acquisition.

### Supplementary Figure S2. Edge correction in spatial dispersion

![Supplementary Figure S2 panel](figures/F22_edge_correction.png)

**Supplementary Figure S2. Edge correction in spatial dispersion.** Border-corrected and uncorrected Ripley's K across the radius grid against the expectation under complete spatial randomness, showing the magnitude and direction of boundary bias. Computable without further data acquisition.

## Table legends

**Table 1.** Habitat classification performance by context mode: macro-F1 and Cohen's kappa as mean and standard deviation across seeds, difference from the per-patch baseline, paired test result, parameter count and training time.

**Table 2.** Univariate Cox associations between habitat abundance and survival, per compartment: hazard ratio per standard deviation, 95 percent confidence interval, p value, adjusted q value, sample count and EVENT count.

**Supplementary Table S1.** Dataset inventory: cohort, platform, sample size, role in the study, accession, and the published resource each replaces.

**Supplementary Table S2.** Cellular neighbourhood definitions: label, assigned name, most enriched cell types, and frequency.

**Supplementary Table S3.** Patch counts by habitat label and split, with patient-level split composition.

**Supplementary Table S4.** Classifier performance with bootstrap confidence intervals.

**Supplementary Table S6.** Per-class precision, recall and F1 for every habitat and context mode.

**Supplementary Table S7.** Full univariate Cox results across all features.

**Supplementary Table S8.** Ecotype clinical associations with tests used and adjusted q values.

**Supplementary Table S9.** Prognostic signature: selected features, coefficients, selection frequency and permutation importance.

**Supplementary Table S10.** Complete itemised list of departures from the published method: parameter, published value, value used here, rationale and expected impact.

## References

1. [canvas] Li et al. Virtual spatial profiling of tumour ecological habitats from routine histology. Cell (2026). doi: 10.1016/j.cell.2026.05.031. **UNVERIFIED - DOI as supplied in the project brief; could not be independently confirmed against PubMed at the time of writing. VERIFY BEFORE SUBMISSION.**

2. [schurch] Schurch CM, Bhate SS, Barlow GL, et al. Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. Cell. 2020;182(5):1341-1359.e19. PMID: 32763154. doi: 10.1016/j.cell.2020.10.021.

3. [orion] Lin JR, Chen YA, Campton D, et al. High-plex immunofluorescence imaging and traditional histology of the same tissue section for discovering image-based biomarkers. Nat Cancer. 2023;4(7):1036-1052. doi: 10.1038/s43018-023-00576-1. (DOI verified against the Nature Cancer article page; PMID not captured during verification.)

4. [danenberg] Danenberg E, Bardwell H, Zanotelli VRT, et al. Breast tumor microenvironment structures are associated with genomic features and clinical outcome. Nat Genet. 2022;54(5):660-669. PMID: 35437329. doi: 10.1038/s41588-022-01041-y.

5. [jackson] Jackson HW, Fischer JR, Zanotelli VRT, et al. The single-cell pathology landscape of breast cancer. Nature. 2020;578(7796):615-620. PMID: 31959985. doi: 10.1038/s41586-019-1876-x.

6. [keren] Keren L, Bosse M, Marquez D, et al. A structured tumor-immune microenvironment in triple negative breast cancer revealed by multiplexed ion beam imaging. Cell. 2018;174(6):1373-1387.e19. PMID: 30193111. doi: 10.1016/j.cell.2018.08.039.

7. [tcgacdr] Liu J, Lichtenberg T, Hoadley KA, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell. 2018;173(2):400-416.e11. PMID: 29625055. doi: 10.1016/j.cell.2018.02.052.

8. [musk] Xiang J, Wang X, Zhang X, et al. A vision-language foundation model for precision oncology. Nature. 2025;638(8051):769-778. doi: 10.1038/s41586-024-08378-w. (Volume/pages verified against the Nature article page.)

9. [uni] Chen RJ, Ding T, Lu MY, et al. Towards a general-purpose foundation model for computational pathology. Nat Med. 2024;30(3):850-862. doi: 10.1038/s41591-024-02857-3.

10. [phikon] Filiot A, Ghermi R, Olivier A, et al. Scaling self-supervised learning for histopathology with masked image modeling. medRxiv (2023). doi: 10.1101/2023.07.21.23292757. (Preprint. Check for a peer-reviewed version before submission.)

11. [clam] Lu MY, Williamson DFK, Chen TY, et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng. 2021;5(6):555-570. PMID: 33649564. doi: 10.1038/s41551-020-00682-w.

12. [titan] Ding T, Wagner SJ, Song AH, et al. A multimodal whole-slide foundation model for pathology. Nat Med. 2025;31(11). doi: 10.1038/s41591-025-03982-3. (Also available as arXiv:2411.19666.)

13. [macenko] Macenko M, Niethammer M, Marron JS, et al. A method for normalizing histology slides for quantitative analysis. IEEE ISBI. 2009:1107-1110. doi: 10.1109/ISBI.2009.5193250. (Conference paper; no PMID.)


---

*Citation keys appear inline as [ref: key] and must be replaced with the journal's numbering during submission preparation. Every reference above was checked against PubMed or the publisher page except where marked UNVERIFIED.*
