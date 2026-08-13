# Quickstart

## Day 1 — prove the stats work, no data needed

    pip install -r requirements.txt
    python -m pytest tests/ -q
    python scripts/run_stage5_features.py --simulate --n-samples 300

Should print `shape=(300, 262)`. If it does not, stop and fix stage 5 before
downloading anything.

## Day 2 to 3 — real CNs from IMC

1. Download Danenberg 2022 from https://doi.org/10.5281/zenodo.5850952
2. Find the single-cell table. Inspect its columns:

       python -c "import pandas as pd; d=pd.read_csv('...'); print(d.columns.tolist()); print(d.head())"

3. **Confirm coordinate units before anything else.** Check the max span. A TMA
   core is roughly 600 to 1000 µm across. If the span is tens of thousands, the
   table is in pixels; convert with the acquisition pixel size (IMC is typically
   1 µm/px, but verify against the paper rather than assuming).
4. Map columns and run:

       python scripts/run_stage1_cn.py \
         --cells data/raw/danenberg_cells.csv \
         --colmap '{"ImageNumber":"image_id","ObjectNumber":"cell_id","Location_Center_X":"x_um","Location_Center_Y":"y_um","cellPhenotype":"cell_type"}'

5. Read `data/processed/k_sweep.csv` and `cn_lineage_enrichment.csv`. Name the
   ten breast CNs from the enrichment table.

Concordance check worth doing: Danenberg published their own 10 TME structures.
Compute the adjusted Rand index between your CNs and theirs. High agreement
validates the pipeline; disagreement is interesting and worth explaining. Either
way it is a result, and it costs one function call.

## Week 2 — paired Xenium and H&E

Download one 10x Xenium breast dataset with post-Xenium H&E and the alignment
file. Get registration working on that single sample and report the residual
before touching anything else.

## Week 3 onwards — model and TCGA

Only after all of the above.
