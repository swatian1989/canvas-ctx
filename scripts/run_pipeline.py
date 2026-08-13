#!/usr/bin/env python
"""Single end-to-end entry point for the whole pipeline.

Runs every stage in order, in VS Code, a plain terminal, or Google Colab.
Each stage is delegated to the same standalone script documented in
docs/PIPELINE.md, so there is one implementation and no duplicated logic
here: this file is an orchestrator, not a second pipeline.

    python scripts/run_pipeline.py                    # everything
    python scripts/run_pipeline.py --dry-run          # show the plan only
    python scripts/run_pipeline.py --from 4           # resume at stage 4
    python scripts/run_pipeline.py --stages 1 2 3     # selected stages
    python scripts/run_pipeline.py --specimens CRC01 CRC02 CRC03
    python scripts/run_pipeline.py --quick            # small, fast smoke run

Idempotent. A stage whose output already exists is skipped unless --force,
so an interrupted run resumes where it stopped rather than starting again.

Stages that need many hours of CPU (patch encoding, the benchmark) are
marked SLOW in the plan. `--dry-run` prints the plan with those estimates
before anything executes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass
class Stage:
    number: int
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    slow: str = ""
    note: str = ""
    expect_files: tuple[str, int] | None = None

    def satisfied(self) -> bool:
        """True when every declared output already exists.

        A stage that writes one shard per specimen is only complete when all
        of them are present. Testing the containing directory instead would
        report a half-finished encoding run as done, and the pipeline would
        march on to train against partial data.
        """
        if not self.produces or not all((ROOT / p).exists() for p in self.produces):
            return False
        if self.expect_files:
            pattern, expected = self.expect_files
            return len(list((ROOT / self.produces[0]).glob(pattern))) >= expected
        return True

    def progress(self) -> str:
        """Human-readable partial state, for stages that build up shards."""
        if not self.expect_files or not (ROOT / self.produces[0]).exists():
            return ""
        pattern, expected = self.expect_files
        have = len(list((ROOT / self.produces[0]).glob(pattern)))
        return f"{have}/{expected} shards present" if have < expected else ""

    def missing_inputs(self) -> list[str]:
        return [p for p in self.needs if not (ROOT / p).exists()]


def build_plan(specimens: list[str], quick: bool) -> list[Stage]:
    seeds = ["1", "2", "3"] if quick else ["1", "2", "3", "4", "5", "6"]
    epochs = "5" if quick else "30"
    n_sim = "50" if quick else "200"
    spec_args = ["--specimens", *specimens]

    return [
        Stage(0, "Verify the test suite",
              "-m", ["pytest", "tests/", "-q"],
              note="Fails fast if the environment is broken before any compute."),

        Stage(1, "Download the discovery cohort (Schurch CODEX, 223 MB)",
              "scripts/download_schurch.py", [],
              produces=["data/raw/CRC_clusters_neighborhoods_markers.csv"]),

        Stage(2, "Prepare the discovery cohort (pixels to microns)",
              "scripts/prepare_schurch.py", [],
              produces=["data/interim/schurch_crc_cells.parquet"],
              needs=["data/raw/CRC_clusters_neighborhoods_markers.csv"]),

        Stage(3, "Discover cellular neighbourhoods",
              "scripts/run_stage1_cn.py",
              ["--config", "config/crc_train_brca_apply.yaml",
               "--cells", "data/interim/schurch_crc_cells.parquet",
               "--outdir", "data/processed"],
              produces=["data/processed/cn_assignments.parquet",
                        "data/processed/k_sweep.csv"],
              needs=["data/interim/schurch_crc_cells.parquet"],
              slow="~10 min (k sweep dominates)"),

        Stage(4, "Validate neighbourhoods against the published labels",
              "scripts/validate_cn_vs_published.py", [],
              produces=["results/tables/T11_cn_vs_published.csv"],
              needs=["data/processed/cn_assignments.parquet"]),

        Stage(5, f"Download the paired cohort ({len(specimens)} specimens)",
              "scripts/download_orion.py", spec_args,
              produces=[f"data/raw/orion_crc/{specimens[0]}/"
                        f"{specimens[0]}-HE-registered.ome.tif"]
                       if specimens else [],
              slow="~1.5 GB per specimen"),

        Stage(6, "Verify registration (residual bound vs the 5 um threshold)",
              "scripts/run_orion_registration_qc.py", ["--level", "3"],
              produces=["results/orion_registration_qc.json"],
              needs=["data/raw/orion_crc"]),

        Stage(7, "Derive one shared habitat taxonomy and label patches",
              "scripts/run_orion_cohort.py", [],
              produces=["data/interim/orion_cohort/cohort_patch_labels.parquet"],
              needs=["data/raw/orion_crc"],
              slow="~50 min for 12 specimens"),

        Stage(8, "Extract example H&E patches (visual gating audit)",
              "scripts/extract_orion_patches.py", [],
              produces=["figures/F7_patch_labels.png"],
              needs=["data/raw/orion_crc"]),

        Stage(9, "Encode patches to cached embeddings",
              "scripts/encode_orion_patches.py",
              ["--encoder", "phikon", "--batch-size", "32"],
              produces=["data/interim/orion_embeddings"],
              needs=["data/interim/orion_cohort/cohort_patch_labels.parquet"],
              expect_files=("*.parquet", len(specimens)),
              slow="SLOW: ~2.9 h on CPU for 24k patches, ~10 min on a T4 GPU"),

        Stage(10, "Method 1 vs Method 2: the controlled context ablation",
              "scripts/run_final_benchmark.py",
              ["--embeddings", "data/interim/orion_embeddings",
               "--modes", "none", "graph", "grid2d", "grid3d",
               "--seeds", *seeds, "--epochs", epochs, "--window", "7",
               "--outdir", "results/real_benchmark"],
              produces=["results/real_benchmark/final_benchmark.csv"],
              needs=["data/interim/orion_embeddings"],
              slow="SLOW: ~3 h on CPU at 6 seeds x 4 modes"),

        Stage(11, "Spatial features on simulated maps (machinery check)",
              "scripts/run_stage5_features.py",
              ["--simulate", "--n-samples", n_sim],
              produces=["data/interim/sim_features.parquet"],
              slow="~17 min at n=200"),

        Stage(12, "Calibrate the survival chain against a null outcome",
              "scripts/validate_stage6_null.py",
              ["--features", "data/interim/sim_features.parquet",
               "--endpoint", "OS"],
              produces=["results/null_cox_OS.csv"],
              needs=["data/interim/sim_features.parquet"]),

        Stage(13, "Generate the report (figures, tables, md/html/docx)",
              "scripts/run_report.py", [],
              produces=["reports/analysis_report.md"],
              slow="~6 min"),

        Stage(14, "Generate the manuscript",
              "scripts/run_manuscript.py", [],
              produces=["reports/manuscript.md"]),
    ]


def run_stage(st: Stage, force: bool, dry: bool) -> str:
    """Returns one of: ok, skipped, blocked, failed."""
    head = f"[{st.number:>2}] {st.name}"

    if st.missing_inputs():
        print(f"{head}\n     BLOCKED, missing: {', '.join(st.missing_inputs())}")
        return "blocked"

    if st.satisfied() and not force:
        print(f"{head}\n     skipped, output exists ({st.produces[0]})")
        return "skipped"

    partial = st.progress()
    if partial:
        print(f"{head}\n     resuming, {partial}")

    cmd = ([PY, "-m", *st.args] if st.script == "-m"
           else [PY, "-u", str(ROOT / st.script), *st.args])

    if dry:
        extra = f"  [{st.slow}]" if st.slow else ""
        print(f"{head}{extra}\n     would run: {' '.join(cmd[1:])}")
        return "ok"

    print(f"{head}" + (f"  [{st.slow}]" if st.slow else ""))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"     FAILED (exit {proc.returncode}) after {dt:.1f} min")
        return "failed"
    print(f"     done in {dt:.1f} min")
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--stages", type=int, nargs="+", help="run only these stage numbers")
    ap.add_argument("--from", dest="start", type=int, default=0, help="resume from here")
    ap.add_argument("--to", dest="end", type=int, default=99)
    ap.add_argument("--specimens", nargs="+",
                    default=[f"CRC{i:02d}" for i in range(1, 13)])
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds/epochs/samples for a fast smoke run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun completed stages")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    plan = build_plan(args.specimens, args.quick)
    if args.stages:
        plan = [s for s in plan if s.number in args.stages]
    plan = [s for s in plan if args.start <= s.number <= args.end]

    print("=" * 74)
    print(f"canvas-ctx pipeline  |  {len(plan)} stages  |  "
          f"{len(args.specimens)} specimens" + ("  |  QUICK" if args.quick else ""))
    if args.dry_run:
        print("DRY RUN, nothing will execute")
    print("=" * 74)

    counts = {"ok": 0, "skipped": 0, "blocked": 0, "failed": 0}
    t0 = time.time()
    for st in plan:
        status = run_stage(st, args.force, args.dry_run)
        counts[status] += 1
        if status == "failed" and not args.continue_on_error:
            print("\nStopping. Fix the failure above, then rerun: completed stages "
                  "are skipped automatically.")
            break

    print("=" * 74)
    print(f"{counts['ok']} run, {counts['skipped']} skipped, "
          f"{counts['blocked']} blocked, {counts['failed']} failed "
          f"in {(time.time() - t0) / 60:.1f} min")
    if counts["blocked"]:
        print("Blocked stages are waiting on data. Run the download stages first, "
              "or pass --stages to target what you can run.")
    print("=" * 74)
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
