"""Tests for the reporting infrastructure.

No downloads, no training, no real cohort data. These check the report's
structural contracts -- the ones that fail silently and produce a
plausible-looking but incomplete document.
"""
from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from canvas_brca.reporting import figures as figs
from canvas_brca.reporting import report as rpt
from canvas_brca.reporting import tables as tbls
from canvas_brca.reporting.style import placeholder_figure, save_figure


def test_all_figures_and_tables_registered():
    """22 figures as specified; 11 tables = the 10 specified plus T11, the
    CN-versus-published validation, which became possible once the Schurch
    table (which ships the authors' own CN labels) was acquired.
    """
    assert len(figs.ALL_FIGURES) == 22
    assert len(tbls.ALL_TABLES) == 11


def test_every_generated_item_is_placed_in_a_section():
    """The bug this guards: a figure/table function exists and runs, but no
    Section references its id, so it never reaches the report at all.
    """
    fig_ids = [f"F{i}" for i in range(1, 23)]
    tab_ids = [f"T{i}" for i in range(1, 12)]
    fig = {i: {"id": i, "title": i, "source": "SIMULATED", "caption": "c",
              "paths": {"png": "x.png", "pdf": "x.pdf"}} for i in fig_ids}
    fig["F20"]["stats"] = {"c_index": 0.6, "n_selected": 4, "n": 200}
    # T5 must carry the rows build_sections() actually reads for the summary
    # numbers (mode x metric), not just any DataFrame.
    t5_df = pd.DataFrame([
        {"metric": "macro_f1", "mode": m, "mean": v, "sd": 0.01, "n_seeds": 6}
        for m, v in [("none", 0.72), ("graph", 0.90), ("grid2d", 0.99), ("grid3d", 0.998)]
    ])
    tab = {i: {"id": i, "title": i, "source": "SIMULATED", "caption": "c",
              "csv_path": "x.csv",
              "df": t5_df if i == "T5" else pd.DataFrame({"a": [1]})} for i in tab_ids}
    stats = {"config_path": "config/x.yaml", "config_hash_sha256": "deadbeef",
            "versions": {"numpy": "1.0"}, "git_sha": "NOT A GIT REPOSITORY",
            "python_version": "3.12.0", "platform": "TestOS 1",
            "project_seed": 42}

    sections = rpt.build_sections(fig, tab, stats)
    placed_figs = {i for s in sections for i in s.figure_ids}
    placed_tabs = {i for s in sections for i in s.table_ids}

    assert set(fig_ids) - placed_figs == set(), "figures generated but never placed"
    assert set(tab_ids) - placed_tabs == set(), "tables generated but never placed"


def test_placeholder_figure_names_the_missing_file(tmp_path):
    fig, ax = placeholder_figure("F99", "Some missing thing",
                                 missing_file="data/raw/needed_file.csv",
                                 unblocks="Phase 9")
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "DATA NOT PROVIDED" in texts
    assert "data/raw/needed_file.csv" in texts
    paths = save_figure(fig, "F99_test", tmp_path)
    assert paths["png"].endswith(".png") and paths["pdf"].endswith(".pdf")


def test_missing_artefact_yields_placeholder_not_crash(tmp_path, monkeypatch):
    """A figure whose input file is absent must degrade to a labelled
    placeholder, never raise and never silently fabricate content.
    """
    monkeypatch.setattr(figs, "RESULTS", tmp_path / "nonexistent")
    meta = figs.f08_training_curves(str(tmp_path))
    assert meta["source"] == "MISSING DATA"
    assert "training_curves.csv" in meta["caption"]


@pytest.mark.parametrize("source", ["REAL DATA (cohort, n=5)", "SIMULATED (fixture)",
                                    "MISSING DATA"])
def test_source_label_maps_to_a_css_class(source):
    assert rpt._source_class(source) in {"source-real", "source-sim", "source-missing"}
