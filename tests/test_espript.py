import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from modules.espript import (MAX_CELLS, FigureOptions, cell_count, column_classes, consensus_line, figure_caption,
                             layout_blocks, map_ss_to_alignment, region_columns, render_alignment_figure,
                             secondary_structure, ss_elements, ss_elements_by_column)
from modules.figures import export_bytes
from modules.structures import StructureFile

DATA = Path(__file__).parent / "data"
ROWS = [("a", "MKV-LIDE"), ("b", "MKVALVDE"), ("c", "MKIALIEQ")]


def test_column_classes_identical_similar_none():
    kinds = [k for k, _, _ in column_classes([s for _, s in ROWS], 0.7)]
    assert kinds[:2] == ["ident", "ident"]          # M, K everywhere
    assert kinds[2] == "sim"                        # V, V, I: one ILVM group
    assert kinds[3] == "none"                       # -, A, A: 2/3 = 0.67 < 0.7
    # with a gap the column is never identical
    assert column_classes(["A", "-"], 0.5)[0][0] == "sim"


def test_similarity_threshold_boundary():
    seqs = ["DK", "EK", "QK"]           # D, E in DE (2/3), Q not
    assert column_classes(seqs, 0.66)[0][0] == "sim"
    assert column_classes(seqs, 0.7)[0][0] == "none"


def test_consensus_line_case():
    seqs = [s for _, s in ROWS]
    line = consensus_line(seqs, column_classes(seqs, 0.6))
    assert line[:2] == "MK" and line[2] in "vi" and line[2].islower()


def test_ss_elements_labels_and_minimum_lengths():
    ss = "-HHHH--HHH--GGG-EE-TT-IIIIHH-E-"
    els = ss_elements(ss)
    assert [(k, label) for k, _, _, label in els] == [("alpha", "α1"), ("eta", "η1"), ("beta", "β1"), ("turn", "TT"),
                                                      ("alpha", "α2")]
    assert els[0][1:3] == (1, 4)                  # 0-based inclusive
    assert all(label != "α" for *_, label in els)


def test_map_ss_through_gaps_and_offset():
    ref_aligned = "M-KVLA--W"                     # reference residues MKVLAW
    chain_seq, chain_ss = "KVLAW", "HHHE-"        # chain lacks the initiator M
    assert map_ss_to_alignment(ref_aligned, chain_seq, chain_ss) == "  HHHE   "


def test_layout_and_region():
    assert layout_blocks(0, 130, 60) == [(0, 60), (60, 120), (120, 130)]
    assert region_columns("M-KVLA--W", 2, 4) == (2, 5)
    assert region_columns("M-KVLA--W", 0, 99) == (0, 9)
    assert cell_count(3, 2, 5) == 9


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None, reason="mkdssp not installed")
@pytest.mark.parametrize("name,fmt", [("1crn.pdb", "PDB"), ("1crn.cif", "mmCIF")])
def test_dssp_crambin(name, fmt):
    sf = StructureFile(name=name, data=(DATA / name).read_bytes(), fmt=fmt)
    seq, ss, source, reason, _ = secondary_structure(sf, "A")
    assert source == "DSSP" and len(seq) == len(ss) == 46 and not reason
    kinds = [k for k, *_ in ss_elements(ss)]
    assert kinds.count("alpha") == 2 and kinds.count("beta") == 2


def test_file_records_fallback(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "find_mkdssp", lambda: None)
    text = (DATA / "1crn.pdb").read_text()
    records = ("HELIX    1  H1 ILE A    7  PRO A   19  1                                  13\n"
               "SHEET    1   S1 2 THR A   1  CYS A   4  0\n")
    sf = StructureFile(name="x.pdb", data=(records + text).encode(), fmt="PDB")
    seq, ss, source, reason, _ = secondary_structure(sf, "A")
    assert source == "file records" and "mkdssp" in reason
    assert ss[6:19] == "H" * 13 and ss[:4] == "EEEE"


def test_no_secondary_structure_explains_why(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "find_mkdssp", lambda: None)
    text = "\n".join(l for l in (DATA / "1crn.pdb").read_text().splitlines() if not l.startswith(("HELIX", "SHEET")))
    sf = StructureFile(name="x.pdb", data=text.encode(), fmt="PDB")
    _, ss, source, reason, _ = secondary_structure(sf, "A")
    assert source == "none" and set(ss) == {"-"} and "not available" in reason


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None, reason="mkdssp not installed")
def test_dssp_failure_reason_is_reported(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "_dssp", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    text = "\n".join(l for l in (DATA / "1crn.pdb").read_text().splitlines() if not l.startswith(("HELIX", "SHEET")))
    sf = StructureFile(name="x.pdb", data=text.encode(), fmt="PDB")
    _, ss, source, reason, _ = secondary_structure(sf, "A")
    assert source == "none" and set(ss) == {"-"} and "boom" in reason


def test_render_and_export():
    rows = [("seq_one", "MKVLAWGKV" * 8), ("seq_two", "MKVIAWGRV" * 8)]
    ss = "  HHHHH  " * 8
    fig = render_alignment_figure(rows, "seq_one", ss, FigureOptions(block=30, consensus=True),
                                  markers={"Consensus suggestion": {3}, "Motif site": {10, 11}}, ss_label="1ABC chain A")
    texts = [t for t in fig.axes[0].texts]
    assert sum(t.get_gid() == "residue" for t in texts) == 2 * 72
    assert sum(t.get_gid() == "marker" for t in texts) == 3
    assert any(t.get_text() == "α1" for t in texts)
    svg = export_bytes(fig, "svg")
    assert svg.lstrip().startswith(b"<?xml") and b"Courier New" in svg and b">W<" in svg
    assert export_bytes(fig, "pdf").startswith(b"%PDF")
    assert Image.open(BytesIO(export_bytes(fig, "png"))).info["dpi"][0] == pytest.approx(600, abs=1)


def test_region_render_numbers_follow_reference():
    rows = [("r", "MKVLAWGKVE" * 3), ("s", "MKVLAWGRVE" * 3)]
    fig = render_alignment_figure(rows, "r", None, FigureOptions(), columns=region_columns(rows[0][1], 11, 30))
    ruler = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "ruler"]
    assert ruler == ["20", "30"]


def test_styles_and_caption():
    for style in ("classic", "lab", "grey"):
        render_alignment_figure(ROWS, "a", None, FigureOptions(style=style))
    caption = figure_caption(FigureOptions(threshold=0.8), "DSSP", "1CRN chain A")
    assert "80%" in caption and "DSSP" in caption and "1CRN chain A" in caption
    assert "No secondary structure" in figure_caption(FigureOptions(), "none")
    assert MAX_CELLS == 30000


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None, reason="mkdssp not installed")
def test_secondary_structure_every_chain_of_a_complex():
    # Review Focus: multi-chain mmCIF (author chain IDs); every protein chain maps residue for residue
    from modules.structure_view import cached_residue_table
    sf = StructureFile(name="1a02.cif", data=(DATA / "1a02.cif").read_bytes(), fmt="mmCIF")
    table = cached_residue_table(sf.data, sf.fmt, sf.name)
    assert len(table) >= 2
    for chain_id, residues in table.items():
        seq, ss, source, _, _ = secondary_structure(sf, chain_id)
        assert len(seq) == len(ss) == len(residues) and source in ("DSSP", "file records", "none")


def test_dna_rows_and_single_column_region_render():
    # Review Focus: DNA alignments use the same rules (identity still highlighted); a one-residue region renders
    rows = [("x", "ATGC-A"), ("y", "ATGCTA")]
    fig = render_alignment_figure(rows, "x", None, FigureOptions(), columns=region_columns("ATGC-A", 3, 3))
    assert sum(t.get_gid() == "residue" for t in fig.axes[0].texts) == 2


def test_ss_elements_split_at_record_breaks():
    els = ss_elements("-HHHHHHHHHHHH-", breaks={7})
    assert [(a, b, label) for _, a, b, label in els] == [(1, 6, "α1"), (7, 12, "α2")]


def test_ss_elements_by_column_bridges_reference_gaps():
    # Final review: a helix interrupted by a reference gap is one element spanning the gap columns
    els = ss_elements_by_column("MKV--LLAEW", "MKVLLAEW", "-HHHHHH-")
    assert els == [("alpha", 1, 8, "α1")]


def test_render_draws_one_element_across_reference_gap():
    rows = [("r", "MKV--LLAEW"), ("s", "MKVAALLAEW")]
    for ss in (ss_elements_by_column(rows[0][1], "MKVLLAEW", "-HHHHHH-"),
               map_ss_to_alignment(rows[0][1], "MKVLLAEW", "-HHHHHH-")):
        fig = render_alignment_figure(rows, "r", ss, FigureOptions())
        labels = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "ss_element"]
        assert labels == ["α1"] and len(fig.axes[0].lines) == 1
        x = [t.get_position()[0] for t in fig.axes[0].texts if t.get_gid() == "residue"][:10]   # row r
        helix = fig.axes[0].lines[0].get_xdata()
        assert min(helix) < x[1] < x[3] < x[4] < x[8] < max(helix) < x[9]   # K..E, across the gap columns


# 4HHB chain B: three touching HELIX records (A, B, C helices of beta-globin)
HBB_4HHB = "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESF"
HELIX_4HHB_B = ("HELIX    9  BA THR B    4  VAL B   18  1                                  15\n"
                "HELIX   10  BB ASN B   19  VAL B   34  1                                  16\n"
                "HELIX   11  BC TYR B   35  PHE B   41  1                                   7\n")


def _ca_only_pdb(chain, seq):
    from Bio.SeqUtils import seq3
    lines = [f"ATOM  {i:5d}  CA  {seq3(aa).upper()} {chain}{i:4d}    {i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00"
             f"           C" for i, aa in enumerate(seq, 1)]
    return "\n".join(lines) + "\nEND\n"


def test_touching_helix_records_stay_separate(monkeypatch):
    import modules.espript as espript
    monkeypatch.setattr(espript, "find_mkdssp", lambda: None)
    sf = StructureFile(name="hbb.pdb", data=(HELIX_4HHB_B + _ca_only_pdb("B", HBB_4HHB)).encode(), fmt="PDB")
    seq, ss, source, reason, breaks = secondary_structure(sf, "B")
    assert seq == HBB_4HHB and source == "file records"
    assert ss[3:41] == "H" * 38 and set(breaks) == {3, 18, 34}
    els = ss_elements_by_column(HBB_4HHB, seq, ss, breaks)
    assert [(a + 1, b + 1, label) for _, a, b, label in els] == [(4, 18, "α1"), (19, 34, "α2"), (35, 41, "α3")]


def test_dna_columns_identity_only():
    seqs = ["ATGA", "TTGA", "ATGT"]
    kinds = [k for k, _, _ in column_classes(seqs, 0.5, seq_type="DNA")]
    assert kinds == ["none", "ident", "ident", "none"]
    assert column_classes(seqs, 0.5)[0][0] == "sim"      # protein rules: A/T share the STA group
    caption = figure_caption(FigureOptions(seq_type="DNA"), "none")
    assert "DNA: identity only" in caption and "FYW" not in caption


def test_columns_gapped_in_every_shown_row_are_dropped():
    # Final review: a hidden row filled columns 3-4; the shown rows are all gaps there
    rows = [("r", "MKV--LAWGKVE"), ("s", "MKV--LAWGRVE")]
    fig = render_alignment_figure(rows, "r", None, FigureOptions(), markers={"m": {7}})
    texts = fig.axes[0].texts
    residues = [t for t in texts if t.get_gid() == "residue"]
    assert len(residues) == 2 * 10 and not any(t.get_text() == "." for t in residues)
    assert [t.get_text() for t in texts if t.get_gid() == "ruler"] == ["10"]
    marker = next(t for t in texts if t.get_gid() == "marker")
    assert marker.get_position()[0] == pytest.approx(residues[5].get_position()[0])   # column 7 -> 6th shown W


def test_legend_lists_only_marker_sets_inside_the_region():
    rows = [("r", "MKVLAWGKVE"), ("s", "MKVLAWGRVE")]
    fig = render_alignment_figure(rows, "r", None, FigureOptions(), markers={"Out": {9}, "In": {2}},
                                  columns=(0, 5))
    legend = [t.get_text() for t in fig.axes[0].texts if t.get_gid() == "legend"]
    assert "In" in legend and "Out" not in legend
    triangle = next(t for t in fig.axes[0].texts if t.get_gid() == "marker")
    assert triangle.get_text() == "▲"          # the second set keeps its symbol
