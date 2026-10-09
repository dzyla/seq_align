import math

import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from modules.phylogeny import (
    MAX_DISTANCE, alignment_distance_matrix, build_phylogenetic_tree, build_tree_from_alignment,
    compute_distance, correct_distance, p_distance,
)


def rec(seq_id, seq):
    return SeqRecord(Seq(seq), id=seq_id, description="")


def test_p_distance_ignores_gaps():
    assert p_distance("AC-T", "AGGT") == pytest.approx(1 / 3)
    assert p_distance("--", "AA") is None


def test_jukes_cantor_and_kimura():
    assert correct_distance(0.0, "DNA") == 0.0
    assert correct_distance(0.1, "DNA") == pytest.approx(-0.75 * math.log(1 - 0.4 / 3))
    assert correct_distance(0.8, "DNA") == MAX_DISTANCE
    assert correct_distance(0.1, "Protein") == pytest.approx(-math.log(1 - 0.1 - 0.2 * 0.01))


def test_alignment_distance_matrix_symmetric():
    d = alignment_distance_matrix(["ACGT", "ACGA", "TCGA"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(0.25)
    assert d[1, 0] == d[0, 1]
    assert d[0, 0] == 0.0


def test_tree_lowercase_input():
    assert compute_distance(Seq("mkvlaa"), Seq("MKVLAA"), "Protein") == 0.0


def test_nj_tree_has_all_leaves():
    seqs = [rec("a", "MKVLAAGIVG"), rec("b", "MKVLAAGIVA"), rec("c", "MRVLSAGLVA"), rec("d", "MRILSAGLVA")]
    tree = build_phylogenetic_tree(seqs, "Protein", method="nj")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c", "d"]


def test_tree_from_alignment_upgma():
    aln = MultipleSeqAlignment([rec("a", "ACGT-A"), rec("b", "ACGTTA"), rec("c", "TCGTTA")])
    tree = build_tree_from_alignment(aln, "DNA", method="upgma")
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]


@pytest.mark.parametrize("method", ["nj", "upgma"])
def test_identical_sequences_give_tree_with_all_leaves(method):
    tree = build_phylogenetic_tree([rec("a", "MKV"), rec("b", "MKV"), rec("c", "MKV")], "Protein", method=method)
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]


def test_tree_to_newick_roundtrip():
    from io import StringIO
    from Bio import Phylo
    from modules.phylogeny import tree_to_newick
    tree = Phylo.read(StringIO("((A:0.1,B:0.2):0.3,C:0.5);"), "newick")
    text = tree_to_newick(tree)
    assert sorted(t.name for t in Phylo.read(StringIO(text), "newick").get_terminals()) == ["A", "B", "C"]


from modules.phylogeny import bootstrap_tree, tree_splits

GROUPED = [("a", "A" * 19 + "C"), ("b", "A" * 20), ("c", "T" * 19 + "G"), ("d", "T" * 20)]


def test_vectorized_distance_matches_definition():
    d = alignment_distance_matrix(["AC-T", "AGGT", "TC-T"], "DNA", correction=False)
    assert d[0, 1] == pytest.approx(1 / 3)
    assert d[0, 2] == pytest.approx(1 / 3)


def test_bootstrap_supports_true_split():
    tree = bootstrap_tree(GROUPED, "DNA", method="nj", correction=False, replicates=50, seed=1)
    supports = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    assert 100 in supports


def test_tree_splits_rooting_invariant():
    from io import StringIO
    from Bio import Phylo
    t1 = Phylo.read(StringIO("((a,b),(c,d));"), "newick")
    t2 = Phylo.read(StringIO("(a,(b,(c,d)));"), "newick")
    names = frozenset("abcd")
    assert tree_splits(t1, names, "a") == tree_splits(t2, names, "a") == {frozenset("cd")}


def test_bootstrap_tree_empty_alignment_returns_tree():
    tree = bootstrap_tree([("a", ""), ("b", ""), ("c", "")], "DNA", replicates=10)
    assert sorted(t.name for t in tree.get_terminals()) == ["a", "b", "c"]


def test_bootstrap_tree_same_seed_same_supports():
    def supports():
        t = bootstrap_tree(GROUPED, "DNA", correction=False, replicates=20, seed=3)
        return [c.confidence for c in t.get_nonterminals()]
    assert supports() == supports()


def test_bootstrap_supports_survive_newick_round_trip():
    from io import StringIO
    from Bio import Phylo
    from modules.phylogeny import tree_to_newick
    tree = bootstrap_tree(GROUPED, "DNA", method="nj", correction=False, replicates=20, seed=1)
    back = Phylo.read(StringIO(tree_to_newick(tree)), "newick")
    inner = [c for c in back.get_nonterminals() if c is not back.root]
    assert inner
    for clade in back.get_nonterminals():
        assert clade.name is None
    assert all(isinstance(c.confidence, (int, float)) for c in inner)


import itertools
import textwrap
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
GLOBIN_NEWICK = ("(('sp|P69905|HBA_HUMAN':0.02,'sp|P01942|HBA_MOUSE':0.06)100:0.3,((HBB_HUMAN:0.01,"
                 "'HBB GORGO':0.014)100:0.07,(HBB_MOUSE:0.05,HBB_HORSE:0.08)72:0.03)99:0.28,MYG_HUMAN:0.6);")


def tree_page(tmp_path, newick=GLOBIN_NEWICK):
    script = tmp_path / "tree_page.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from io import StringIO
        from Bio import Phylo
        from modules.phylogeny import phylogenetic_tree_section
        phylogenetic_tree_section(Phylo.read(StringIO({newick!r}), "newick"), "t")
    """))
    at = AppTest.from_file(str(script), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def control(at, key):
    return next(w for w in at.get("button_group") if w.key == key)


def test_tree_section_every_layout_and_style(tmp_path):
    at = tree_page(tmp_path)
    for layout, style in itertools.product(["Rectangular", "Circular", "Unrooted"],
                                           ["Classic ink", "Support dots", "Clade colour"]):
        control(at, "t_layout").set_value(layout).run()
        control(at, "t_style").set_value(style).run()
        assert not at.exception, (layout, style, at.exception)
        assert [d.label for d in at.get("download_button")] == ["SVG", "PDF", "PNG (600 dpi)", "Newick"]


def test_tree_section_outgroup_and_fallback(tmp_path):
    at = tree_page(tmp_path)
    at.selectbox(key="t_rooting").set_value("Outgroup").run()
    at.multiselect(key="t_outgroup").set_value(["MYG_HUMAN"]).run()
    assert not at.exception and not at.warning
    at.multiselect(key="t_outgroup").set_value([]).run()
    assert not at.exception and not at.warning
    assert any("Pick one or more outgroup leaves" in i.value for i in at.info)
    every_leaf = at.multiselect(key="t_outgroup").options
    at.multiselect(key="t_outgroup").set_value(every_leaf).run()
    assert not at.exception
    assert any("midpoint rooting instead" in w.value for w in at.warning)


def test_tree_section_explore_view(tmp_path):
    at = tree_page(tmp_path)
    control(at, "t_view").set_value("Explore").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1


def _style_items():
    from dataclasses import asdict
    from modules.tree_render import TreeStyle
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in asdict(TreeStyle()).items())


def test_tree_outputs_keep_odd_leaf_names():
    # Review Focus 1: names with |, spaces and quotes survive the Newick round trip in the cached renderer
    from modules.phylogeny import tree_figure_outputs
    out = tree_figure_outputs(GLOBIN_NEWICK, "rectangular", False, "midpoint", (), True, 1, _style_items())
    shown = {t.hover for t in out["drawing"].texts if t.kind == "leaf"}
    assert {"sp|P69905|HBA_HUMAN", "HBB GORGO", "MYG_HUMAN"} <= shown
    assert out["svg"].lstrip().startswith(b"<?xml") and out["pdf"].startswith(b"%PDF")
    assert out["preview"].startswith(b"\x89PNG")


def test_tree_outputs_handle_deep_caterpillar_tree():
    from Bio.Phylo.BaseTree import Clade, Tree
    from modules.phylogeny import tree_figure_outputs, tree_to_newick
    from modules.tree_layout import recursion_headroom
    node = Clade(branch_length=0.1, name="L0")
    for i in range(1, 300):
        node = Clade(branch_length=0.1, clades=[node, Clade(branch_length=0.1, name=f"L{i}")])
    tree = Tree(root=node, rooted=True)
    with recursion_headroom(tree):
        newick = tree_to_newick(tree)
    out = tree_figure_outputs(newick, "rectangular", False, "as_built", (), False, 3, _style_items())
    assert sum(t.kind == "leaf" for t in out["drawing"].texts) == 300


def test_tree_preview_matches_export_framing():
    import re
    from io import BytesIO
    from PIL import Image
    from modules.phylogeny import tree_figure_outputs
    items = tuple((k, "clades" if k == "style" else v) for k, v in _style_items())
    out = tree_figure_outputs(GLOBIN_NEWICK, "circular", False, "midpoint", (), True, 3, items)
    w, h = Image.open(BytesIO(out["preview"])).size
    svg_w, svg_h = (float(v) for v in re.search(rb'width="([\d.]+)pt" height="([\d.]+)pt"', out["svg"]).groups())
    assert w / 150 == pytest.approx(svg_w / 72, abs=0.03) and h / 150 == pytest.approx(svg_h / 72, abs=0.03)


def test_tree_section_clade_colour_on_two_leaf_tree(tmp_path):
    at = tree_page(tmp_path, "(A:0.1,B:0.2);")
    control(at, "t_style").set_value("Clade colour").run()
    assert not at.exception, at.exception
    assert not [w for w in at.slider if w.key == "t_clades"]


def test_tree_section_clade_names_persist_across_k(tmp_path):
    at = tree_page(tmp_path)
    at.session_state["t_clade_name_store"] = ["Alpha", "", "Gamma"]
    control(at, "t_style").set_value("Clade colour").run()
    assert not at.exception
    assert at.session_state["t_clade_name_store"][:3] == ["Alpha", "", "Gamma"]
    at.slider(key="t_clades").set_value(2).run()
    at.slider(key="t_clades").set_value(4).run()
    assert not at.exception
    assert at.session_state["t_clade_name_store"][:4] == ["Alpha", "", "Gamma", "Clade 4"]


def test_tree_section_notes_when_figure_is_wider_than_asked(tmp_path):
    at = tree_page(tmp_path)
    control(at, "t_layout").set_value("Circular").run()
    at.slider(key="t_width").set_value(8.0).run()
    assert not at.exception
    assert any("cm wide so labels fit" in c.value for c in at.caption)


def dark_tree_page(tmp_path, monkeypatch, charts):
    import modules.ui
    monkeypatch.setattr(modules.ui, "theme_type", lambda: "dark")  # AppTest runs the script in this process
    script = tmp_path / "dark_tree_page.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from io import StringIO
        import streamlit as st
        from Bio import Phylo
        st.session_state["appearance_charts"] = {charts!r}
        from modules.phylogeny import phylogenetic_tree_section
        phylogenetic_tree_section(Phylo.read(StringIO({GLOBIN_NEWICK!r}), "newick"), "t")
    """))
    at = AppTest.from_file(str(script), default_timeout=120)
    at.run()
    control(at, "t_view").set_value("Explore").run()
    assert not at.exception, at.exception
    return at.get("plotly_chart")[0].proto.spec


@pytest.mark.parametrize("charts,cream", [("Match theme", True), ("Publication white", False)])
def test_explore_view_is_dark_only_when_charts_match_night_theme(tmp_path, monkeypatch, charts, cream):
    assert ("#ece8df" in dark_tree_page(tmp_path, monkeypatch, charts)) is cream


def test_asr_tree_figure_keeps_node_names_supports_and_highlight():
    from dataclasses import asdict
    from modules.ancestral import parse_tree
    from modules.phylogeny import tree_figure_outputs, tree_to_newick
    from modules.tree_layout import join_node_labels
    from modules.tree_render import TreeStyle
    tree = parse_tree((Path(__file__).parent / "data" / "asr" / "globins.treefile").read_text())
    style = TreeStyle(internal_labels=True, highlight="Node4")
    items = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in asdict(style).items())
    out = tree_figure_outputs(tree_to_newick(join_node_labels(tree)), "rectangular", False, "midpoint", (), True, 1,
                              items)
    names = {t.text for t in out["drawing"].texts if t.kind == "node"}
    assert {"Node2", "Node3", "Node4"} <= names
    assert {t.text for t in out["drawing"].texts if t.kind == "support"} >= {"82", "93"}
    assert b">Node4<" in out["svg"] and "Node4/48" in out["newick"]
    assert sum("Node4" in d[5] for d in out["drawing"].dots) == 1
